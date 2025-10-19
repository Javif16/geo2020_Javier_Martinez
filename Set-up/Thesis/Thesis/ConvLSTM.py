'''
Thermal data:
This file will contain the main Convolutional Long Short-Term Memory model, with which thermal image classification will
be performed and achieved.
Here, the program will also be trained and tested, with parameter tunning taking place here as well.

Geological Maps:
· Pre-labelled geological maps
· Converted into segmentation greyscale masks matching the input shape of images in the CNN with same CRS
· Each pixel corresponds to a specific class (rock/soil) with numbers (sand=1, clay=2, chalk=3, etc.)

Evaluation:
    - overlay output of models over the geological maps and compare pixel by pixel
    - Accuracy, F1 score, IoU inside model (during training)
'''

import numpy as np
import os
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import rasterio
import random
import tensorflow as tf
from tensorflow.keras import backend
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Conv3D, MaxPooling2D, MaxPooling3D, Dense, Flatten, Dropout, SpatialDropout3D
from tensorflow.keras.layers import Conv2DTranspose, concatenate, BatchNormalization
from tensorflow.keras.layers import ConvLSTM2D, TimeDistributed, UpSampling2D, UpSampling3D
from tensorflow.keras.regularizers import l2
from tensorflow.keras.initializers import HeNormal
from tensorflow.keras.metrics import Precision, Recall, BinaryAccuracy, CategoricalAccuracy
from tensorflow.keras.models import load_model
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import precision_score, recall_score, accuracy_score, f1_score
from sklearn.model_selection import KFold
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import optuna
from optuna.trial import TrialState
import pickle
from math import ceil
from collections import defaultdict
from utils import get_map_size_from_path, reconstruct_map_from_patches

# -------------- LOADING DATA ------------------------------------------------------------------------------------------
data_path = r"E:/Studies/Thesis/Outputs/Puertollano overlap/thermal_only/"
X_train = np.load(data_path + 'X_train_seq.npy')
X_valid = np.load(data_path + 'X_valid_seq.npy')
X_test = np.load(data_path + 'X_test_seq.npy')
y_train = np.load(data_path + 'y_train_seq.npy')
y_valid = np.load(data_path + 'y_valid_seq.npy')
y_test = np.load(data_path + 'y_test_seq.npy')
pos_test = np.load(data_path + 'pos_test_seq.npy')
# positions
pos_train = np.load(data_path + 'pos_train_seq.npy')
pos_valid = np.load(data_path + 'pos_valid_seq.npy')
pos_test = np.load(data_path + 'pos_test_seq.npy')
# ndvi
ndvi_test_agg = np.load(data_path + 'ndvi_test_agg.npy')
has_ndvi_test = np.load(data_path + 'has_ndvi_test_seq.npy')
ndvi_test_counts = np.load(data_path + 'ndvi_test_counts.npy')
print(f"Loaded NDVI aggregated data: {ndvi_test_agg.shape}")
print(f"Sequences with NDVI: {np.sum(has_ndvi_test)}/{len(has_ndvi_test)}")
print(f"Mean NDVI frames per sequence: {np.mean(ndvi_test_counts):.1f}")


# Cleaning for rgb and sar
X_train[:, :, :, :, 2:7] = np.nan_to_num(X_train[:, :, :, :, 2:7], nan=0.0)
X_valid[:, :, :, :, 2:7] = np.nan_to_num(X_valid[:, :, :, :, 2:7], nan=0.0)
X_test[:, :, :, :, 2:7] = np.nan_to_num(X_test[:, :, :, :, 2:7], nan=0.0)
for channel in range(X_train.shape[-1]):
    channel_data = X_train[:, :, :, channel]
    nan_count = np.isnan(channel_data).sum()
    total = channel_data.size
    print(f"Channel {channel}: {nan_count}/{total} NaN values ({nan_count/total*100:.2f}%)")

#X_train = X_train[:, :, :, :, :2]
#X_valid = X_valid[:, :, :, :, :2]
#X_test = X_test[:, :, :, :, :2]


# -------------- ConvLSTM U-Net ----------------------------------------------------------------------------------------
def clstm_unet_model(size_input=(5, 128, 128, 2), filters_base=16, classes=14, l2_reg=0.01):
    '''
    U-Net architecture with encoder and decoder.
    '''
    def Encoder(inputs, filters, dropout, l2factor, max_pooling=True):
        '''
        Convolutional and pooling layers, in tandem with ReLU activation function to learn.
        Dropout additional to prevent overfitting problems.
        '''
        # 2 convolutional layers with ReLU and HeNormal initialization
        # Proper initialization to prevent gradient problems
        # Padding 'same' means that the dimensionality of the images is not
        # reduced after each convolution (useful for U-Net arhitecture)
        convolution = ConvLSTM2D(filters, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal',
                             kernel_regularizer=l2(l2factor), return_sequences=True)(inputs)
        convo = BatchNormalization()(convolution)

        # Conv2D(number of filters, kernel size, activation function, padding, initialization) - input shape established in thermal.py
        # L2 regularization = weight decay, prevents overfitting by adding a penalty to the loss function
        # based on sum of squared values of the model’s weights.
        # If value reduced, model is more flexible. If increased might underfit.
        convolution2 = ConvLSTM2D(filters, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal',
            kernel_regularizer=l2(l2factor), return_sequences=True)(convo)
        convo2 = BatchNormalization()(convolution2)
        # batch normalization to normalize output
        # BN best before dropout, as dropout disrupts the stability (removes neurons) BN needs to normalize a stable distribution of activations
        # BN best before activation function and dropout as it prevents instability issues and ensures proper gradient flow

        # If overfitting, dropout regularizes loss and gradient
        # to minimize influence of weights on the output
        if dropout > 0:
            convo2 = SpatialDropout3D(dropout)(convo2)

        # Pooling - reduces image size while maintaining channel number
        if max_pooling:
            next_layer = MaxPooling3D(pool_size=(1, 2, 2))(convo2)
        else:
            next_layer = convo2
        skip_connection = convo2

        return next_layer, skip_connection

    def Decoder(previous_layer, skip, filters, l2factor):
        '''
        Transpose convolutions to make images bigger (upscale) back to original size and merges with skip layer from Encoder.
        2 convolutional layers with 'same' padding increases depth of architecture for better predictions.
        '''
        upscale = tf.keras.layers.Reshape(
            (1, previous_layer.shape[1], previous_layer.shape[2], previous_layer.shape[3]))(previous_layer)
        upscale = UpSampling3D(size=(1, 2, 2))(upscale)

        # Take only the last timestep from skip connection for concatenation
        skip_last = tf.keras.layers.Lambda(lambda x: x[:, -1:, :, :, :])(skip)
        combination = concatenate([upscale, skip_last], axis=4)

        # Conv2D instead of ConvLSTM2D in decoder since we're no longer processing sequences
        combination = tf.keras.layers.Reshape((combination.shape[2], combination.shape[3], combination.shape[4]))(
            combination)
        convolution = Conv2D(filters, (3, 3), activation='relu', padding='same',
                             kernel_initializer='he_normal', kernel_regularizer=l2(l2factor))(combination)
        convo = BatchNormalization()(convolution)

        return convo

    # input size of the image
    inputs = tf.keras.layers.Input(shape=size_input)

    # encoder
    conv1, skip1 = Encoder(inputs, filters_base, dropout=0.0, l2factor=l2_reg, max_pooling=True)  # 372 x 585
    conv2, skip2 = Encoder(conv1, filters_base * 2, dropout=0.0, l2factor=l2_reg, max_pooling=True)  # 186 x 292
    conv3, skip3 = Encoder(conv2, filters_base * 4, dropout=0.1, l2factor=l2_reg, max_pooling=True)  # 93 x 146
    conv4, skip4 = Encoder(conv3, filters_base * 8, dropout=0.15, l2factor=l2_reg, max_pooling=True)  # 46 x 73
    conv5, skip5 = Encoder(conv4, filters_base * 16, dropout=0.2, l2factor=l2_reg, max_pooling=True)  # 23 x 36

    # bottleneck (last layer before upscaling)
    bottleneck = ConvLSTM2D(filters_base * 32, (3, 3), activation='relu', padding='same',
                            kernel_initializer='he_normal', dropout=0.3, kernel_regularizer=l2(l2_reg),
                            return_sequences=False)(conv5)    # maxpooling in last convolution as upscaling starts here

    # decoder with reducing filters with skip connections from encoder given as input
    # second output of encoder block is skip connection, so conv1[1] used
    upsc1 = Decoder(bottleneck, skip5, filters_base * 16, l2factor=l2_reg)
    upsc2 = Decoder(upsc1, skip4, filters_base * 8, l2factor=l2_reg)
    upsc3 = Decoder(upsc2, skip3, filters_base * 4, l2factor=l2_reg)
    upsc4 = Decoder(upsc3, skip2, filters_base * 2, l2factor=l2_reg)
    upsc5 = Decoder(upsc4, skip1, filters_base, l2factor=l2_reg)

    # layer to get to proper size - single segmentation map
    conv10 = Conv2D(filters_base, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(upsc5)
    # Conv2D used to pass from 5D tensor to 4D tensor
    # better for single map output - no need for sequence-for-sequence predictions

    # final layer
    output = Conv2D(classes, (1, 1), padding='same', activation='softmax', dtype='float32')(conv10)

    model = tf.keras.Model(inputs=inputs, outputs=output, name='unet_segmentation_convlstm')
    model.summary()
    return model


class EpochTracker(tf.keras.callbacks.Callback):
    """Track predictions and metrics at each epoch"""

    def __init__(self, X_test, y_test, compute_metrics_fn):
        super().__init__()
        self.X_test = X_test
        self.y_test = y_test
        self.compute_metrics_fn = compute_metrics_fn
        self.epoch_predictions = []
        self.epoch_metrics = []
        self.selected_epoch_predictions = {}
        self.selected_epoch_metrics = {}
        self.epochs_to_store = None
        self.total_epochs = None

    def on_train_begin(self, logs=None):
        self.total_epochs = self.params.get("epochs", 30)
        self.epochs_to_store = self._get_epochs_to_store()
        print(f"Will store predictions for epochs: {self.epochs_to_store}")

    def _get_epochs_to_store(self):
        """Epochs from which to store predictions."""
        if self.total_epochs < 6:
            return list(range(1, self.total_epochs + 1))
        else:
            epochs = [1, 10, self.total_epochs]
            epochs = sorted(list(set(epochs)))
            return epochs

    def on_epoch_end(self, epoch, logs=None):
        current_epoch = epoch + 1
        print(f"Computing metrics for epoch {current_epoch}...")

        predictions_list = []
        batch_size = 4
        for i in range(0, len(self.X_test), batch_size):
            batch_X = self.X_test[i:i + batch_size]
            batch_pred = self.model.predict(batch_X, verbose=0, batch_size=4)
            predictions_list.append(batch_pred)

        predictions = np.concatenate(predictions_list, axis=0)
        pred_classes = np.argmax(predictions, axis=-1)
        metrics = self.compute_metrics_fn(self.y_test, pred_classes)

        self.epoch_metrics.append(metrics)

        if current_epoch in self.epochs_to_store:
            print(f"  → Storing predictions for epoch {current_epoch} (selected epoch)")
            self.selected_epoch_predictions[current_epoch] = pred_classes.copy()
            self.selected_epoch_metrics[current_epoch] = metrics.copy()
        else:
            print(f"  → Skipping prediction storage for epoch {current_epoch}")

        print(f"Epoch {current_epoch} - Test Accuracy: {metrics['accuracy']:.4f}, "
              f"Mean IoU: {metrics['mean_iou']:.4f}")

        del predictions, pred_classes
        import gc
        gc.collect()


# -------------- LOSS FUNCTION -----------------------------------------------------------------------------------------
def simple_weighted_loss(class_weights, gamma=2.0, alpha=0.25):
    """
    Combines focal loss (for hard examples) with class weights (for imbalance)
    """
    def loss_fn(y_true, y_pred):
        y_true = tf.squeeze(y_true, axis=-1)  # [batch, H, W]
        y_true = tf.cast(y_true, tf.int32)
        y_pred = tf.clip_by_value(y_pred, 1e-7, 1.0 - 1e-7)

        y_true_one_hot = tf.one_hot(y_true, depth=14)
        cross_entropy = -y_true_one_hot * tf.math.log(y_pred)
        focal_weight = tf.pow(1 - y_pred, gamma)
        focal_loss = alpha * focal_weight * cross_entropy

        class_weight_tensor = tf.constant([
            class_weights.get(0, 1.0), class_weights.get(1, 1.0),
            class_weights.get(2, 1.0), class_weights.get(3, 1.0),
            class_weights.get(4, 1.0), class_weights.get(5, 1.0),
            class_weights.get(6, 1.0), class_weights.get(7, 1.0),
            class_weights.get(8, 1.0), class_weights.get(9, 1.0),
            class_weights.get(10, 1.0), class_weights.get(11, 1.0),
            class_weights.get(12, 1.0), class_weights.get(13, 1.0)
        ], dtype=tf.float32)

        sample_weights = tf.gather(class_weight_tensor, y_true)
        sample_weights = tf.expand_dims(sample_weights, axis=-1)
        weighted_focal_loss = focal_loss * sample_weights

        return tf.reduce_mean(tf.reduce_sum(weighted_focal_loss, axis=-1))

    return loss_fn


# -------------- TRAINING ----------------------------------------------------------------------------------------------
def train_evaluate_model(X_train, y_train, X_val, y_val, X_test, y_test, use_optuna=True, n_trials=30, epochs=30):
    """
    Training and evaluation of model with optional Optuna hyperparameter optimization.
    """

    print("=== ConvLSTM U-Net Training ===")
    if use_optuna:
        print("=== STARTING OPTUNA HYPERPARAMETER OPTIMIZATION ===")

        def objective(trial):
            tf.keras.backend.clear_session()

            # Suggest hyperparameters
            learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-3, log=True)
            batch_size = trial.suggest_categorical('batch_size', [16, 32, 64])
            dropout_rate = trial.suggest_float('dropout_rate', 0.1, 0.5)
            l2_reg = trial.suggest_float('l2_regularization', 0.01, 0.2)
            weight_strength = trial.suggest_float('weight_strength', 0.5, 2.0)

            try:
                # MODEL
                model = clstm_unet_model(
                    size_input=(5, 64, 64, 1),
                    filters=32,
                    classes=14)

                # OPTIMIZER and LOSS
                optimizer = tf.keras.optimizers.Adam(learning_rate=best_params['learning_rate'])
                loss = tf.keras.losses.SparseCategoricalCrossentropy()

                model.compile(
                    optimizer=optimizer,
                    loss=loss,
                    metrics=['accuracy', 'precision', 'recall']
                )

                # CALLBACKS
                early_stopping = tf.keras.callbacks.EarlyStopping(
                    monitor='val_loss', patience=5, restore_best_weights=True, verbose=0
                )
                pruning_callback = optuna.integration.TFKerasPruningCallback(trial, 'val_loss')

                # TRAINING
                history = model.fit(
                    X_train, y_train,
                    validation_data=(X_val, y_val),
                    epochs=15,
                    batch_size=batch_size,
                    callbacks=[early_stopping, pruning_callback],
                    verbose=0
                )

                val_loss = min(history.history['val_loss'])
                return val_loss

            except Exception as e:
                print(f"Trial failed: {e}")
                return float('inf')

        study = optuna.create_study(
            direction='minimize',
            pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=5)
        )

        study.optimize(objective, n_trials=n_trials)

        print(f"\nOptimization completed!")
        print(f"Best parameters: {study.best_params}")
        print(f"Best validation loss: {study.best_value:.4f}")
        best_params = study.best_params

    else:
        # base params
        best_params = {
            'learning_rate': 0.0001,
            'l2factor': 0.05,
            'batch_size': 4,
            'filters_base': 16
        }
        print("=== USING BASE PARAMETERS (NO OPTUNA) ===")

    print("\n=== TRAINING FINAL ConvLSTM MODEL ===")
    tf.keras.backend.clear_session()

    # MODEL
    model = clstm_unet_model(
        size_input=(5, 128, 128, 2),
        filters_base=best_params['filters_base'],
        classes=14,
        l2_reg=best_params['l2factor'])

    print("Model output shape:", model.output_shape)

    class_weights = {
        0: 1.5,  # NaNs (background)
        1: 10.0,  # Sand
        2: 10.0,  # Clay -
        3: 1.0,  # Chalk
        4: 7.5,  # Silt -
        5: 1.0,  # Peat
        6: 4.0,  # Loam -
        7: 1.0,  # Detritic -
        8: 1.5,  # Carbonate -
        9: 3.0,  # Volcanic -
        10: 1.0,  # Plutonic
        11: 15.0,  # Foliated -
        12: 2.5,  # Non-Foliated -
        13: 1.0  # Water
    }
    # Puertollano - 1, 2, 4, 6, 7, 8, 9, 11, 12
    # Santa Olalla - 1, 3, 4, 7, 8, 9, 10, 11, 12
    # Villoslada - 1, 2, 4, 5, 7, 8

    # OPTIMIZER and LOSS
    optimizer = tf.keras.optimizers.Adam(learning_rate=best_params['learning_rate'])
    # loss = tf.keras.losses.SparseCategoricalCrossentropy()
    loss = simple_weighted_loss(class_weights, gamma=2.0, alpha=1.0)

    model.compile(
        optimizer=optimizer,
        loss=loss,
        metrics=['accuracy', tf.keras.metrics.SparseCategoricalAccuracy()]
    )

    # TRACKER
    epoch_tracker = EpochTracker(X_test, y_test, compute_metrics)

    # CALLBACKS
    lr_callback = tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=5,
        min_lr=1e-7,
        verbose=1
    )
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=15,
        restore_best_weights=True
    )

    # TRAINING
    print("Training final ConvLSTM model...")
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=150,
        batch_size=best_params['batch_size'],
        callbacks=[lr_callback, early_stopping, epoch_tracker],
        verbose=1
    )

    print("\n=== FINAL EVALUATION ===")
    evaluation_results = model.evaluate(X_test, y_test, batch_size=4, verbose=0)
    result_names = ['Loss', 'Accuracy', 'Sparse Categorical Accuracy', 'F1 Score']
    for i, value in enumerate(evaluation_results):
        name = result_names[i] if i < len(result_names) else f'Metric_{i}'
        print(f"Test {name}: {value:.4f}")

    print("Computing final predictions...")
    predictions_list = []
    for i in range(len(X_test)):
        pred = model.predict(X_test[i:i + 1], batch_size=4, verbose=0)
        predictions_list.append(pred)
    final_predictions = np.concatenate(predictions_list, axis=0)
    predictions = np.argmax(final_predictions, axis=-1)
    confidences = np.max(final_predictions, axis=-1)

    return (model, history, predictions, best_params,
            epoch_tracker.selected_epoch_predictions,
            epoch_tracker.selected_epoch_metrics,
            epoch_tracker.epoch_metrics, confidences)


# -------------- PERFORMANCE -------------------------------------------------------------------------------------------
def compute_metrics(y_true, y_pred, class_names=None, print_results=True):
    """
    Metric computation for segmentation.
    """
    print("Computing metrics for full images...")
    # Convert predictions to class indices if they're probabilities
    # Flatten for metric calculation
    y_true_flat = y_true.flatten()
    y_pred_flat = y_pred.flatten()

    unique_classes = np.unique(np.concatenate([y_true_flat, y_pred_flat]))
    n_classes = len(unique_classes)

    # Confusion matrix
    cm = confusion_matrix(y_true_flat, y_pred_flat, labels=unique_classes)
    per_class_iou = {}
    iou_scores = []

    for i, class_id in enumerate(unique_classes):
        # IoU = TP / (TP + FP + FN)
        tp = cm[i, i]  # True positives
        fp = cm[:, i].sum() - tp  # False positives
        fn = cm[i, :].sum() - tp  # False negatives

        if tp + fp + fn == 0:
            iou = 0.0
        else:
            iou = tp / (tp + fp + fn)

        class_name = class_names[class_id] if class_names else f"Class_{class_id}"
        per_class_iou[class_name] = iou
        iou_scores.append(iou)
        if print_results:
            print(f"{class_name} (ID: {class_id}): IoU = {iou:.4f} | TP={tp}, FP={fp}, FN={fn}")

    # Macro F1 (average of per-class F1 scores)
    f1_macro = f1_score(y_true_flat, y_pred_flat, labels=unique_classes, average='macro')
    # Weighted F1 (weighted by support)
    f1_weighted = f1_score(y_true_flat, y_pred_flat, labels=unique_classes, average='weighted')
    # Per-class F1 score
    f1_per_class = f1_score(y_true_flat, y_pred_flat, labels=unique_classes, average=None)

    # Mean IoU
    if 0 in unique_classes and len(unique_classes) > 1:
        non_bg_ious = [iou for i, iou in enumerate(iou_scores) if unique_classes[i] != 0]
        mean_iou = np.mean(non_bg_ious) if non_bg_ious else 0.0
        mean_iou_all = np.mean(iou_scores)
    else:
        mean_iou = np.mean(iou_scores)
        mean_iou_all = mean_iou

    accuracy = np.sum(y_true_flat == y_pred_flat) / len(y_true_flat)

    if print_results:
        print(f"\nOverall Metrics:")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Mean IoU (excluding background): {mean_iou:.4f}")
        print(f"Mean IoU (all classes): {mean_iou_all:.4f}")
        print(f"F1 Score (Macro): {f1_macro:.4f}")
        print(f"F1 Score (Weighted): {f1_weighted:.4f}")
        print(f"\nPer-class F1 scores:")
        for i, class_id in enumerate(unique_classes):
            class_name = class_names[class_id] if class_names else f"Class_{class_id}"
            print(f"{class_name}: {f1_per_class[i]:.4f}")
        print("-" * 50)

    return {
        'accuracy': accuracy,
        'mean_iou': mean_iou,
        'mean_iou_all': mean_iou_all,
        'per_class_iou': per_class_iou,
        'iou_scores': iou_scores,
        'unique_classes': unique_classes,
        'f1_macro': f1_macro,
        'f1_weighted': f1_weighted,
        'f1_per_class': f1_per_class
    }


from tensorflow.keras import backend as K


def f1_score_keras(y_true, y_pred):
    """
    F1 score metric for Keras - works with multi-class segmentation
    """
    def recall_m(y_true, y_pred):
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall = true_positives / (possible_positives + K.epsilon())
        return recall

    def precision_m(y_true, y_pred):
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        return precision

    precision = precision_m(y_true, y_pred)
    recall = recall_m(y_true, y_pred)
    return 2 * ((precision * recall) / (precision + recall + K.epsilon()))


# -------------- EVALUATION --------------------------------------------------------------------------------------------
def plot_training(history):
    """
    Plotting function for metric evolution viewing
    """
    available_metrics = list(history.history.keys())
    has_f1 = any('f1' in metric.lower() for metric in available_metrics)

    if has_f1:
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    else:
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Loss
    axes[0].plot(history.history['loss'], label='Training Loss')
    if 'val_loss' in history.history:
        axes[0].plot(history.history['val_loss'], label='Validation Loss')
    axes[0].set_title('Model Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].legend()
    axes[0].grid(True)

    # Accuracy
    axes[1].plot(history.history['accuracy'], label='Training Accuracy')
    if 'val_accuracy' in history.history:
        axes[1].plot(history.history['val_accuracy'], label='Validation Accuracy')
    axes[1].set_title('Model Accuracy')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy')
    axes[1].legend()
    axes[1].grid(True)

    # F1 Score
    if has_f1:
        f1_keys = [key for key in available_metrics if 'f1' in key.lower() and not key.startswith('val_')]
        val_f1_keys = [key for key in available_metrics if 'f1' in key.lower() and key.startswith('val_')]

        for f1_key in f1_keys:
            axes[2].plot(history.history[f1_key], label=f'Training {f1_key.replace("_", " ").title()}')

        for val_f1_key in val_f1_keys:
            axes[2].plot(history.history[val_f1_key],
                         label=f'Validation {val_f1_key.replace("val_", "").replace("_", " ").title()}')

        axes[2].set_title('Model F1 Score')
        axes[2].set_xlabel('Epoch')
        axes[2].set_ylabel('F1 Score')
        axes[2].legend()
        axes[2].grid(True)
    plt.tight_layout()
    plt.show()


def visualize_convlstm_results(X_test_seq, y_test_seq, predictions_seq, data_path, positions_test_seq=None, num_patches=3,
                               patch_indices=None, timestep_to_visualize=-1, figsize=(15, 5), cmap_thermal='hot',
                               cmap_segmentation='tab20'):
    """
    Visualize ConvLSTM results showing:
    1. Individual sequences (thermal images, ground truth labels, and predictions)
    2. Reconstructed full maps (ground truth and predictions)
    """

    if positions_test_seq is not None and len(positions_test_seq) != len(X_test_seq):
        print(f"Warning: Aligning position data ({len(positions_test_seq)}) to sequences ({len(X_test_seq)})")
        positions_test_seq = positions_test_seq[:len(X_test_seq)]

    if patch_indices is None:
        patch_indices = random.sample(range(len(X_test_seq)), num_patches)
    else:
        num_patches = len(patch_indices)

    fig, axes = plt.subplots(num_patches, 3, figsize=figsize, constrained_layout=True)
    if num_patches == 1:
        axes = axes.reshape(1, -1)

    if cmap_segmentation == 'tab20':
        colors = plt.cm.tab20(np.linspace(0, 1, 14))
        seg_cmap = ListedColormap(colors)
    else:
        seg_cmap = cmap_segmentation

    for i, patch_idx in enumerate(patch_indices):
        thermal_sequence = X_test_seq[patch_idx]  # (timesteps, 64, 64, channels)
        true_labels = y_test_seq[patch_idx]
        pred_labels = predictions_seq[patch_idx]

        if len(true_labels.shape) == 3 and true_labels.shape[-1] == 1:
            true_labels = true_labels.squeeze(-1)

        thermal_frame = thermal_sequence[timestep_to_visualize]  # (64, 64, channels)
        thermal_display = thermal_frame[:, :, 0]

        # Thermal Image (selected timestep)
        im1 = axes[i, 0].imshow(thermal_display, cmap=cmap_thermal, aspect='equal')
        timestep_label = f"t={timestep_to_visualize}" if timestep_to_visualize >= 0 else "t=final"
        axes[i, 0].set_title(f'Sequence {patch_idx + 1}: Thermal ({timestep_label})')
        axes[i, 0].axis('off')

        # Ground Truth Labels
        im2 = axes[i, 1].imshow(true_labels, cmap=seg_cmap, vmin=0, vmax=13, aspect='equal')
        axes[i, 1].set_title(f'Sequence {patch_idx + 1}: Ground Truth')
        axes[i, 1].axis('off')

        # Predictions
        im3 = axes[i, 2].imshow(pred_labels, cmap=seg_cmap, vmin=0, vmax=13, aspect='equal')
        axes[i, 2].set_title(f'Sequence {patch_idx + 1}: Predictions')
        axes[i, 2].axis('off')

    cbar1 = plt.colorbar(im1, ax=axes[:, 0].ravel().tolist(), shrink=0.8, aspect=20)
    cbar1.set_label('Thermal Intensity', rotation=270, labelpad=15)

    cbar2 = plt.colorbar(im2, ax=axes[:, 1:].ravel().tolist(), shrink=0.8, aspect=20)
    cbar2.set_label('Class Labels', rotation=270, labelpad=15)
    cbar2.set_ticks(range(14))

    plt.show()

    print(f"\nMetrics for visualized sequences:")
    for i, patch_idx in enumerate(patch_indices):
        true_labels = y_test_seq[patch_idx]
        if len(true_labels.shape) == 3 and true_labels.shape[-1] == 1:
            true_labels = true_labels.squeeze(-1)
        pred_labels = predictions_seq[patch_idx]

        true_flat = true_labels.flatten()
        pred_flat = pred_labels.flatten()
        accuracy = np.mean(true_flat == pred_flat)
        f1_macro = f1_score(true_flat, pred_flat, average='macro', zero_division=0)
        print(f"Sequence {patch_idx + 1}: Accuracy = {accuracy:.4f}, F1 (macro) = {f1_macro:.4f}")

    if positions_test_seq is None:
        print("\nNote: No position data provided. Skipping map reconstruction.")
        print("To enable reconstruction, load positions: pos_test_seq = np.load('pos_test_seq.npy')")
        return

    print(f"\nReconstructing full maps for ConvLSTM sequences...")
    map_size = get_map_size_from_path(data_path)
    print(f"Map size: {map_size[0]}x{map_size[1]}")

    unique_dates = np.unique(positions_test_seq[:, 2])
    print(f"Found {len(unique_dates)} unique date(s) in test sequences")

    for date_idx in unique_dates:
        date_mask = positions_test_seq[:, 2] == date_idx
        date_y = y_test_seq[date_mask]
        date_pred = predictions_seq[date_mask]
        date_pos = positions_test_seq[date_mask]

        if len(date_y.shape) == 4 and date_y.shape[-1] == 1:
            date_y = date_y.squeeze(-1)

        print(f"\nDate {int(date_idx)}: {len(date_y)} sequences")

        # Reconstructed maps from positions
        gt_map = reconstruct_map_from_patches(
            date_y, date_pos, map_size,
            patch_size=128,
            handle_overlap=True)
        pred_map = reconstruct_map_from_patches(
            date_pred, date_pos, map_size,
            patch_size=128,
            handle_overlap=True)
        fig, axes = plt.subplots(1, 2, figsize=(20, 10), constrained_layout=True)

        # Ground Truth Map
        im1 = axes[0].imshow(gt_map, cmap=seg_cmap, vmin=0, vmax=13, aspect='equal')
        axes[0].set_title(f'Date {int(date_idx)}: Ground Truth (ConvLSTM Reconstruction)', fontsize=14)
        axes[0].axis('off')

        # Prediction Map
        im2 = axes[1].imshow(pred_map, cmap=seg_cmap, vmin=0, vmax=13, aspect='equal')
        axes[1].set_title(f'Date {int(date_idx)}: Predictions (ConvLSTM Reconstruction)', fontsize=14)
        axes[1].axis('off')

        cbar = plt.colorbar(im2, ax=axes.ravel().tolist(), shrink=0.6, aspect=20)
        cbar.set_label('Class Labels', rotation=270, labelpad=15)
        cbar.set_ticks(range(14))

        plt.show()

        gt_flat = gt_map.flatten()
        pred_flat = pred_map.flatten()

        valid_mask = gt_flat != 0
        if np.sum(valid_mask) > 0:
            accuracy = np.mean(gt_flat[valid_mask] == pred_flat[valid_mask])
            f1_macro = f1_score(gt_flat[valid_mask], pred_flat[valid_mask],
                                average='macro', zero_division=0)

            print(f"  Accuracy: {accuracy:.4f}")
            print(f"  F1 (macro): {f1_macro:.4f}")
            print(f"  Valid pixels: {np.sum(valid_mask):,} / {len(gt_flat):,}")
        else:
            print("  Warning: No valid pixels in reconstructed map")


def visualize_ndvi_correlation_convlstm(y_test_seq, predictions_seq, confidences_seq,
                                        ndvi_test_agg, has_ndvi_test, ndvi_test_counts,
                                        positions_test_seq, data_path, date_idx=None, min_ndvi_frames=3):
    """
    ConvLSTM version: Uses aggregated NDVI values per sequence
    """
    n_sequences = len(y_test_seq)
    if len(positions_test_seq) != n_sequences:
        print(f"Warning: Aligning position data ({len(positions_test_seq)}) to sequences ({n_sequences})")
        positions_test_seq = positions_test_seq[:n_sequences]
        has_ndvi_test = has_ndvi_test[:n_sequences]
        ndvi_test_agg = ndvi_test_agg[:n_sequences]
        ndvi_test_counts = ndvi_test_counts[:n_sequences]

    map_size = get_map_size_from_path(data_path)
    patch_size = 128
    if date_idx is not None:
        date_mask = positions_test_seq[:, 2] == date_idx
        valid_mask = date_mask & has_ndvi_test & (ndvi_test_counts >= min_ndvi_frames)
    else:
        valid_mask = has_ndvi_test & (ndvi_test_counts >= min_ndvi_frames)

    if np.sum(valid_mask) == 0:
        print(f"No sequences with ≥{min_ndvi_frames} NDVI frames found!")
        return
    print(f"\nAnalyzing {np.sum(valid_mask)} sequences with NDVI data...")
    print(f"  (filtered to sequences with ≥{min_ndvi_frames} NDVI frames)")
    valid_y = y_test_seq[valid_mask]
    if len(valid_y.shape) == 4 and valid_y.shape[-1] == 1:
        valid_y = valid_y.squeeze(-1)

    valid_pred = predictions_seq[valid_mask]
    valid_conf = confidences_seq[valid_mask]
    valid_ndvi = ndvi_test_agg[valid_mask]  # aggregated NDVI
    valid_pos = positions_test_seq[valid_mask]

    # Reconstruct maps
    print("Reconstructing maps...")
    conf_map = reconstruct_map_from_patches(
        valid_conf, valid_pos, map_size,
        patch_size=128, handle_overlap=True)
    ndvi_map = reconstruct_map_from_patches(
        valid_ndvi, valid_pos, map_size,
        patch_size=128, handle_overlap=True)
    gt_map = reconstruct_map_from_patches(
        valid_y, valid_pos, map_size,
        patch_size=128, handle_overlap=True)
    pred_map = reconstruct_map_from_patches(
        valid_pred, valid_pos, map_size,
        patch_size=128, handle_overlap=True)

    # Accuracy map
    accuracy_map = (gt_map == pred_map).astype(float)
    accuracy_map[gt_map == 0] = np.nan
    ndvi_map_display = ndvi_map.copy()
    ndvi_map_display[ndvi_map == 0] = np.nan
    conf_map_display = conf_map.copy()
    conf_map_display[gt_map == 0] = np.nan

    # Confidence + NDVI
    fig, axes = plt.subplots(1, 2, figsize=(20, 10), constrained_layout=True)

    im1 = axes[0].imshow(conf_map_display, cmap='RdYlGn', vmin=0, vmax=1, aspect='equal')
    axes[0].set_title('ConvLSTM Confidence (Max Softmax Probability)', fontsize=14)
    axes[0].axis('off')
    cbar1 = plt.colorbar(im1, ax=axes[0], shrink=0.6)
    cbar1.set_label('Confidence', rotation=270, labelpad=15)

    im2 = axes[1].imshow(ndvi_map_display, cmap='YlGn', vmin=-0.2, vmax=0.8, aspect='equal')
    axes[1].set_title('NDVI (Aggregated per Sequence)', fontsize=14)
    axes[1].axis('off')
    cbar2 = plt.colorbar(im2, ax=axes[1], shrink=0.6)
    cbar2.set_label('NDVI', rotation=270, labelpad=15)

    date_str = f"Date {int(date_idx)}" if date_idx is not None else "All Dates"
    fig.suptitle(f'{date_str}: ConvLSTM Confidence vs NDVI', fontsize=16, y=0.98)
    plt.show()

    # Accuracy Map
    fig, ax = plt.subplots(1, 1, figsize=(10, 10), constrained_layout=True)
    im = ax.imshow(accuracy_map, cmap='RdYlGn', vmin=0, vmax=1, aspect='equal')
    ax.set_title(f'{date_str}: ConvLSTM Pixel-wise Accuracy', fontsize=14)
    ax.axis('off')
    cbar = plt.colorbar(im, ax=ax, shrink=0.6)
    cbar.set_label('Correct (1) / Incorrect (0)', rotation=270, labelpad=15)
    plt.show()

    from scipy.stats import pearsonr

    conf_flat = conf_map.flatten()
    ndvi_flat = ndvi_map.flatten()
    acc_flat = accuracy_map.flatten()

    valid_pixels = (ndvi_flat != 0) & (~np.isnan(ndvi_flat)) & (~np.isnan(conf_flat))

    conf_values = conf_flat[valid_pixels]
    ndvi_values = ndvi_flat[valid_pixels]
    acc_values = acc_flat[valid_pixels]
    print(f"Valid pixels for correlation: {len(conf_values):,}")
    if len(conf_values) < 10:
        print("Not enough valid pixels for correlation!")
        return

    corr_conf_ndvi, p_conf = pearsonr(conf_values, ndvi_values)
    corr_acc_ndvi, p_acc = pearsonr(acc_values, ndvi_values)
    print(f"\nCorrelation Results:")
    print(f"  Confidence vs NDVI: r={corr_conf_ndvi:.3f}, p={p_conf:.4f}")
    print(f"  Accuracy vs NDVI: r={corr_acc_ndvi:.3f}, p={p_acc:.4f}")

    # Correlation Plots
    fig, axes = plt.subplots(1, 2, figsize=(16, 6), constrained_layout=True)

    # Confidence vs NDVI
    axes[0].hexbin(ndvi_values, conf_values, gridsize=50, cmap='Blues', mincnt=1)
    axes[0].set_xlabel('NDVI (Aggregated)', fontsize=12)
    axes[0].set_ylabel('ConvLSTM Confidence', fontsize=12)
    axes[0].set_title(f'Confidence vs NDVI\nr={corr_conf_ndvi:.3f}, p={p_conf:.4f}', fontsize=14)
    axes[0].grid(True, alpha=0.3)

    z = np.polyfit(ndvi_values, conf_values, 1)
    p = np.poly1d(z)
    ndvi_sorted = np.sort(ndvi_values)
    axes[0].plot(ndvi_sorted, p(ndvi_sorted), "r--", linewidth=2,
                 label=f'Trend: y={z[0]:.3f}x+{z[1]:.3f}')
    axes[0].legend()

    # Accuracy vs NDVI
    ndvi_bins = np.linspace(ndvi_values.min(), ndvi_values.max(), 20)
    bin_indices = np.digitize(ndvi_values, ndvi_bins)

    bin_means = []
    bin_centers = []
    for i in range(1, len(ndvi_bins)):
        mask = bin_indices == i
        if np.sum(mask) > 0:
            bin_means.append(np.mean(acc_values[mask]))
            bin_centers.append((ndvi_bins[i - 1] + ndvi_bins[i]) / 2)

    axes[1].scatter(ndvi_values, acc_values, alpha=0.1, s=1, c='blue')
    axes[1].plot(bin_centers, bin_means, 'ro-', linewidth=2, markersize=8,
                 label='Mean Accuracy per Bin')
    axes[1].set_xlabel('NDVI (Aggregated)', fontsize=12)
    axes[1].set_ylabel('Pixel Accuracy', fontsize=12)
    axes[1].set_title(f'Accuracy vs NDVI\nr={corr_acc_ndvi:.3f}, p={p_acc:.4f}', fontsize=14)
    axes[1].set_ylim([-0.1, 1.1])
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    fig.suptitle(f'{date_str}: ConvLSTM NDVI Correlation', fontsize=16, y=1.02)
    plt.show()

    print("\n=== ConvLSTM Performance by NDVI Range ===")
    ndvi_ranges = [
        (-1.0, 0.0, "Bare Soil/Water"),
        (0.0, 0.2, "Low Vegetation"),
        (0.2, 0.4, "Moderate Vegetation"),
        (0.4, 0.6, "Healthy Vegetation"),
        (0.6, 1.0, "Very Healthy Vegetation")
    ]

    for ndvi_min, ndvi_max, label in ndvi_ranges:
        mask = (ndvi_values >= ndvi_min) & (ndvi_values < ndvi_max)
        if np.sum(mask) > 0:
            mean_conf = np.mean(conf_values[mask])
            mean_acc = np.mean(acc_values[mask])
            n_pixels = np.sum(mask)
            print(f"{label:25s} (NDVI {ndvi_min:.1f}-{ndvi_max:.1f}): "
                  f"Conf={mean_conf:.3f}, Acc={mean_acc:.3f}, N={n_pixels:,}")


# -------------- WORKFLOW ----------------------------------------------------------------------------------------------
(model, history, predictions, best_params, epoch_predictions, epoch_metrics, all_epoch_metrics, confidences) = train_evaluate_model(
    X_train, y_train,
    X_valid, y_valid,
    X_test, y_test,
    use_optuna=False, n_trials=5, epochs=30)

class_names = [
    "NaNs",      # Class 0
    "Sand",         # Class 1
    "Clay",         # Class 2
    "Chalk",         # Class 3
    "Silt",         # Class 4
    "Peat",         # Class 5
    "Loam",         # Class 6
    "Detritic",         # Class 7
    "Carbonate",         # Class 8
    "Volcanic",         # Class 9
    "Plutonic",        # Class 10
    "Foliated",        # Class 11
    "Non-Foliated",        # Class 12
    "Water"         # Class 13
]

final_metrics = compute_metrics(y_test, predictions, class_names=class_names)


print("\n=== FINAL PERFORMANCE METRICS ===")
for metric_name, metric_value in final_metrics.items():
    if metric_name == 'per_class_iou':
        print(f"{metric_name}:")
        for class_name, iou_value in metric_value.items():
            print(f"  {class_name}: {iou_value:.4f}")
    elif isinstance(metric_value, (int, float, np.floating)):
        print(f"{metric_name}: {metric_value:.4f}")
    else:
        print(f"{metric_name}: {metric_value}")

print("\n=== BEST PARAMETERS ===")
for param_name, param_value in best_params.items():
    print(f"{param_name}: {param_value}")

plot_training(history)

visualize_convlstm_results(X_test, y_test, predictions, data_path=data_path, positions_test_seq=pos_test, patch_indices=[15, 34, 36], timestep_to_visualize=-1)

print("\n=== CONVLSTM NDVI CORRELATION ANALYSIS ===")
# Per date
unique_dates = np.unique(pos_test[:, 2])
for date_idx in unique_dates:
    print(f"\n--- Analyzing Date {int(date_idx)} ---")
    visualize_ndvi_correlation_convlstm(
        y_test, predictions, confidences,
        ndvi_test_agg, has_ndvi_test, ndvi_test_counts,
        pos_test, data_path, date_idx=date_idx, min_ndvi_frames=3)
# Overall
print(f"\n--- Overall Analysis (All Dates) ---")
visualize_ndvi_correlation_convlstm(
    y_test, predictions, confidences,
    ndvi_test_agg, has_ndvi_test, ndvi_test_counts,
    pos_test, data_path, date_idx=None, min_ndvi_frames=3)

'''
Maximum patch index:
    Villoslada:
        · thermal only - 
        · thermal_sar - 13
        · thermal_day - 82
        · thermal_night - 277
        · thermal_winter - 165
        · thermal_summer - 198
    Santa:
        · thermal only - 83
        · thermal_sar - 15
        · thermal_day - 57
        · thermal_night - 26
        · thermal_winter - 35
        · thermal_summer - 46
    Puertollano:
        · thermal only - 100
        · thermal_sar - 25
        · thermal_day - 63
        · thermal_night - 52
        · thermal_winter - 56
        · thermal_summer - 55
'''

# ---------------- SAVING ----------------------------------------------------------------------------------------------
model.save(os.path.join('models', 'ConvLSTM.keras'))
model.save(os.path.join('models', 'ConvLSTM.h5'))