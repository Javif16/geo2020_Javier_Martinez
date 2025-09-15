'''
Thermal data:
This file will contain the main Convolutional Neural Network, with which thermal image classification will be performed
and achieved.
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
import json
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import random
import rasterio
import tensorflow as tf
from tensorflow.keras import backend
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout
from tensorflow.keras.layers import Conv2DTranspose, concatenate, BatchNormalization
from tensorflow.keras.regularizers import l2
from tensorflow.keras.initializers import HeNormal
from tensorflow.keras.metrics import Precision, Recall, BinaryAccuracy, CategoricalAccuracy
from tensorflow.keras.models import load_model
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, accuracy_score, f1_score
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import optuna
from optuna.trial import TrialState
import pickle
from math import ceil
from collections import defaultdict


# -------------- LOADING -----------------------------------------------------------------------------------------------
data_path = 'E:/Studies/Thesis/THERMAL/Villoslada/Villoslada_thermal_winter/'
X_train = np.load(data_path + 'X_train.npy')
X_valid = np.load(data_path + 'X_valid.npy')
X_test = np.load(data_path + 'X_test.npy')
y_train = np.load(data_path + 'y_train.npy')
y_valid = np.load(data_path + 'y_valid.npy')
y_test = np.load(data_path + 'y_test.npy')

# Cleaning for RGB and SAR
X_train[:, :, :, 2:7] = np.nan_to_num(X_train[:, :, :, 2:7], nan=0.0)
X_valid[:, :, :, 2:7] = np.nan_to_num(X_valid[:, :, :, 2:7], nan=0.0)
X_test[:, :, :, 2:7] = np.nan_to_num(X_test[:, :, :, 2:7], nan=0.0)
for channel in range(X_train.shape[-1]):
    channel_data = X_train[:, :, :, channel]
    nan_count = np.isnan(channel_data).sum()
    total = channel_data.size
    print(f"Channel {channel}: {nan_count}/{total} NaN values ({nan_count/total*100:.2f}%)")

# for training with only thermal data in SAR and RGB tests
#X_train = X_train[:, :, :, :2]
#X_valid = X_valid[:, :, :, :2]
#X_test = X_test[:, :, :, :2]


# -------------- CNN U-Net ---------------------------------------------------------------------------------------------
def unet_model(size_input=(64, 64, 2), filters_base=16, classes=14, l2_reg=0.01):
    '''
    U-Net architecture with encoder and decoder.
    '''
    # Encoder
    def Encoder(inputs, filters, dropout, l2factor, max_pooling=True):
        '''
        Convolutional and pooling layers, in tandem with ReLU activation function to learn.
        Dropout additional to prevent overfitting problems.
        '''
        # 2 convolutional layers with ReLU and HeNormal initialization
        # Proper initialization to prevent gradient problems
        # Padding 'same' means that the dimensionality of the images is not
        # reduced after each convolution (useful for U-Net arhitecture)
        convolution = Conv2D(filters, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal',
                             kernel_regularizer=l2(l2factor))(inputs)
        convolution = BatchNormalization()(convolution)
        convolution = tf.keras.layers.Activation("relu")(convolution)

        # Conv2D(number of filters, kernel size, activation function, padding, initialization) - input shape established in thermal.py
        # L2 regularization = weight decay, prevents overfitting by adding a penalty to the loss function
        # based on sum of squared values of the model’s weights.
        # If value reduced, model is more flexible. If increased might underfit.
        convo = Conv2D(filters, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal',
                       kernel_regularizer=l2(l2factor))(convolution)
        convo = BatchNormalization()(convo)  # Apply BN first
        convo = tf.keras.layers.Activation("relu")(convo)
        # batch normalization to normalize output
        # BN best before dropout, as dropout disrupts the stability (removes neurons) BN needs to normalize a stable distribution of activations
        # BN best before activation function and dropout as it prevents instability issues and ensures proper gradient flow

        # If overfitting, dropout regularizes loss and gradient
        # to minimize influence of weights on the output
        if dropout > 0:
            convo = tf.keras.layers.Dropout(dropout)(convo)

        # Pooling - reduces image size while maintaining channel number
        if max_pooling:
            next_layer = MaxPooling2D(pool_size=(2, 2))(convo)
        else:
            next_layer = convo

        skip_connection = convo
        return next_layer, skip_connection

    def Decoder(previous_layer, skip, filters, l2factor):
        '''
        Transpose convolutions to make images bigger (upscale) back to original size and merges with skip layer from Encoder.
        2 convolutional layers with 'same' padding increases depth of architecture for better predictions.
        '''
        # increasing size
        upscale = Conv2DTranspose(filters, (3, 3), strides=(2, 2), padding='same')(previous_layer)
        if upscale.shape[1] != skip.shape[1] or upscale.shape[2] != skip.shape[2]:
            upscale = tf.keras.layers.Resizing(skip.shape[1], skip.shape[2])(upscale)

        # combine skip connection to prevent loss of information
        combination = concatenate([upscale, skip], axis=3)

        convolution = Conv2D(filters, (3, 3), padding='same', kernel_initializer='he_normal',
                             kernel_regularizer=l2(l2factor))(combination)
        convolution = BatchNormalization()(convolution)
        convolution = tf.keras.layers.Activation("relu")(convolution)

        convo = Conv2D(filters, (3, 3), padding='same', kernel_initializer='he_normal',
                       kernel_regularizer=l2(l2factor))(convolution)
        convo = BatchNormalization()(convo)
        convo = tf.keras.layers.Activation("relu")(convo)

        return convo

    # input size of the image
    inputs = tf.keras.layers.Input(size_input)

    # encoder with multiple convolutional layers with different maxpooling, dropout and filters
    # filters increase the deeper into the network it reaches to increase channel size
    conv1, skip1 = Encoder(inputs, filters_base, dropout=0.0, l2factor=l2_reg, max_pooling=True)  # 372 x 585
    conv2, skip2 = Encoder(conv1, filters_base * 2, dropout=0.0, l2factor=l2_reg, max_pooling=True)  # 186 x 292
    conv3, skip3 = Encoder(conv2, filters_base * 4, dropout=0.0, l2factor=l2_reg, max_pooling=True)  # 93 x 146
    conv4, skip4 = Encoder(conv3, filters_base * 8, dropout=0.2, l2factor=l2_reg, max_pooling=True)  # 46 x 73
    conv5, skip5 = Encoder(conv4, filters_base * 16, dropout=0.2, l2factor=l2_reg, max_pooling=True)  # 23 x 36

    # bottleneck (last layer before upscaling)
    bottleneck, _ = Encoder(conv5, filters_base*32, dropout=0.3, l2factor=l2_reg, max_pooling=False)
    # maxpooling in last convolution as upscaling starts here

    # decoder with reducing filters with skip connections from encoder given as input
    # second output of encoder block is skip connection, so conv1[1] used
    upsc1 = Decoder(bottleneck, skip5, filters_base * 16, l2factor=l2_reg)
    upsc2 = Decoder(upsc1, skip4, filters_base * 8, l2factor=l2_reg)
    upsc3 = Decoder(upsc2, skip3, filters_base * 4, l2factor=l2_reg)
    upsc4 = Decoder(upsc3, skip2, filters_base * 2, l2factor=l2_reg)
    upsc5 = Decoder(upsc4, skip1, filters_base, l2factor=l2_reg)

    # final convolutional layer to get image to proper size, so nº of channels = nº output classes
    conv10 = Conv2D(filters_base, (3, 3), padding='same', kernel_initializer='he_normal',
                    kernel_regularizer=l2(l2_reg))(upsc5)
    conv10 = BatchNormalization()(conv10)
    conv10 = tf.keras.layers.Activation("relu")(conv10)
    # no need for Flatten() as it would output a single classification label instead of a full segmentation map

    output = Conv2D(classes, (1, 1), activation='softmax', padding='same',
                    kernel_regularizer=l2(l2_reg), name='segmentation_output')(conv10)

    # model defining
    model = tf.keras.Model(inputs=inputs, outputs=output, name='unet_segmentation')
    model.summary()

    return model


# -------------- LOSS FUNCTION ---------------------------------------------------------------------------------
def simple_weighted_loss(class_weights):
    def loss_fn(y_true, y_pred):
        y_true = tf.squeeze(y_true, axis=-1)  # [4,64,64,1] -> [4,64,64]
        loss = tf.keras.losses.sparse_categorical_crossentropy(y_true, y_pred)

        # Class weights
        sample_weights = tf.gather([
            class_weights.get(0, 1.0), class_weights.get(1, 1.0), class_weights.get(2, 1.0),
            class_weights.get(3, 1.0), class_weights.get(4, 1.0), class_weights.get(5, 1.0),
            class_weights.get(6, 1.0), class_weights.get(7, 1.0), class_weights.get(8, 1.0),
            class_weights.get(9, 1.0), class_weights.get(10, 1.0), class_weights.get(11, 1.0),
            class_weights.get(12, 1.0), class_weights.get(13, 1.0)
        ], tf.cast(y_true, tf.int32))

        return tf.reduce_mean(loss * sample_weights)

    return loss_fn


class EpochTracker(tf.keras.callbacks.Callback):
    """Enhanced callback to track predictions and per-class metrics at each epoch"""

    def __init__(self, X_test, y_test, compute_metrics_fn, show_per_class=True):
        super().__init__()
        self.X_test = X_test
        self.y_test = y_test
        self.compute_metrics_fn = compute_metrics_fn
        self.show_per_class = show_per_class
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
        print(f"\n{'=' * 60}")
        print(f"EPOCH {current_epoch} - DETAILED METRICS")
        print(f"{'=' * 60}")

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
            print(f"→ Storing predictions for epoch {current_epoch} (selected epoch)")
            self.selected_epoch_predictions[current_epoch] = pred_classes.copy()
            self.selected_epoch_metrics[current_epoch] = metrics.copy()

        if logs:
            print(f"\nTraining Metrics:")
            print(f"Loss: {logs.get('loss', 'N/A'):.4f} | Val Loss: {logs.get('val_loss', 'N/A'):.4f}")
            print(f"Accuracy: {logs.get('accuracy', 'N/A'):.4f} | Val Accuracy: {logs.get('val_accuracy', 'N/A'):.4f}")

        print(f"\nTest Set Performance Summary:")
        print(f"Overall Accuracy: {metrics['accuracy']:.4f}")
        print(f"Mean IoU (excluding background): {metrics['mean_iou']:.4f}")
        print(f"Mean IoU (all classes): {metrics['mean_iou_all']:.4f}")

        if 'per_class_iou' in metrics:
            iou_items = list(metrics['per_class_iou'].items())
            iou_items.sort(key=lambda x: x[1])

            print(f"\nWorst performing classes:")
            for class_name, iou in iou_items[:3]:
                print(f"  {class_name}: {iou:.4f}")

            print(f"\nBest performing classes:")
            for class_name, iou in iou_items[-3:]:
                print(f"  {class_name}: {iou:.4f}")
        print(f"{'=' * 60}\n")

        del predictions, pred_classes
        import gc
        gc.collect()

    def get_class_performance_summary(self):
        """Summary of class performance across all epochs"""
        if not self.epoch_metrics:
            return None
        latest_metrics = self.epoch_metrics[-1]
        if 'per_class_iou' not in latest_metrics:
            return None

        iou_items = list(latest_metrics['per_class_iou'].items())
        iou_items.sort(key=lambda x: x[1], reverse=True)

        return {
            'sorted_classes': iou_items,
            'mean_iou': latest_metrics['mean_iou'],
            'accuracy': latest_metrics['accuracy']
        }


# -------------- TRAINING & EVALUATION ---------------------------------------------------------------------------------
def train_model(X_train, y_train, X_val, y_val, X_test, y_test, use_optuna=True, n_trials=30):
    """
    Training and evaluation of model with optional Optuna hyperparameter optimization.
    """

    print("=== CNN U-Net Training ===")
    print(f"Training on {len(X_train)} images, validating on {len(X_val)} images, testing on {len(X_test)} images")
    if use_optuna:
        print("=== STARTING OPTUNA HYPERPARAMETER OPTIMIZATION ===")

        def objective(trial):
            tf.keras.backend.clear_session()

            # HYPERPARAMETERS
            learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-3)
            batch_size = trial.suggest_categorical('batch_size', [4, 16, 32])
            dropout_rate = trial.suggest_float('dropout_rate', 0.1, 0.2)
            l2_reg = trial.suggest_categorical('l2_reg', [0.001, 0.01, 0.1])
            filters_base = trial.suggest_categorical('filters_base', [16, 32, 64])

            try:
                # MODEL
                model = unet_model(
                    size_input=(64, 64, 5),
                    classes=14,
                    l2_reg=l2_reg,
                    filters_base=filters_base)

                # OPTIMIZER and LOSS
                optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
                loss = tf.keras.losses.SparseCategoricalCrossentropy()

                model.compile(
                    optimizer=optimizer,
                    loss=loss,
                    metrics=['accuracy', tf.keras.metrics.SparseCategoricalAccuracy()]
                )

                # CALLBACK
                early_stopping = tf.keras.callbacks.EarlyStopping(
                    monitor='val_loss', patience=5, restore_best_weights=True, verbose=0
                )
                pruning_callback = optuna.integration.TFKerasPruningCallback(trial, 'val_loss')

                # TRAINING
                history = model.fit(
                    X_train, y_train,
                    validation_data=(X_val, y_val),
                    epochs=100,
                    batch_size=batch_size,
                    callbacks=[early_stopping, pruning_callback],
                    verbose=0
                )

                val_loss = min(history.history['val_loss'])
                return val_loss

            except Exception as e:
                print(f"Trial failed: {e}")
                return float('inf')

        study = optuna.create_study(direction='minimize')
        study.optimize(objective, n_trials=n_trials)

        best_params = study.best_params
        print(f"Best parameters: {best_params}")

    else:
        # Base params
        best_params = {
            'learning_rate': 0.0001,
            'l2_reg': 0.05,
            'batch_size': 4,
            'filters_base': 16
        }
        print("=== USING BASE PARAMETERS (NO OPTUNA) ===")

    print("\n=== TRAINING FINAL MODEL ===")
    tf.keras.backend.clear_session()

    # MODEL
    model = unet_model(
        size_input=(64, 64, 2),
        filters_base=best_params['filters_base'],
        classes=14,
        l2_reg=best_params['l2_reg'])

    print("Model output shape:", model.output_shape)

    class_weights = {
        0: 1.0,  # NaNs (background)
        1: 4.0,  # Sand
        2: 4.0,  # Clay
        3: 1.0,  # Chalk
        4: 1.5, # Silt - keep high
        5: 1.0,  # Peat
        6: 2.0,  # Loam
        7: 0.5,  # Detritic
        8: 1.25,  # Carbonate
        9: 2.0,  # Volcanic
        10: 1.0,  # Plutonic
        11: 0.5,  # Foliated
        12: 0.75, # Non-Foliated
        13: 1.0  # Water
    }
    # Puertollano - 1, 2, 4, 6, 7, 8, 9, 11, 12
    # Santa Olalla - 1, 3, 4, 7, 8, 9, 10, 11, 12
    # Villoslada - 1, 2, 4, 5, 7, 8

    # OPTIMIZER AND LOSS
    optimizer = tf.keras.optimizers.Adam(learning_rate=best_params['learning_rate'])
    loss = tf.keras.losses.SparseCategoricalCrossentropy()
    # loss = simple_weighted_loss(class_weights)

    model.compile(
        optimizer=optimizer,
        loss=loss,
        metrics=['accuracy', tf.keras.metrics.SparseCategoricalAccuracy()]
    )

    # TRACKER
    epoch_tracker = EpochTracker(X_test, y_test, compute_metrics, show_per_class=True)

    # CALLBACKS
    lr_callback = tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=7,
        min_lr=1e-7,
        verbose=1
    )
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=7,
        restore_best_weights=True)

    # TRAINING
    print("Training final CNN model...")
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=100,
        batch_size=best_params['batch_size'],
        callbacks=[lr_callback, early_stopping, epoch_tracker],
        verbose=1
    )

    print("\n=== FINAL EVALUATION ===")
    test_loss, test_acc, test_sparse_acc = model.evaluate(X_test, y_test, batch_size=4, verbose=0)
    print(f"Test loss: {test_loss}")
    print(f"Test Accuracy: {test_acc}")
    print(f"Test Sparse Categorical Accuracy: {test_sparse_acc:.4f}")

    print("Computing final predictions...")
    predictions_list = []
    for i in range(len(X_test)):
        pred = model.predict(X_test[i:i + 1], batch_size=4, verbose=0)
        predictions_list.append(pred)
    final_predictions = np.concatenate(predictions_list, axis=0)
    predictions = np.argmax(final_predictions, axis=-1)

    return (model, history, predictions, best_params,
            epoch_tracker.selected_epoch_predictions,
            epoch_tracker.selected_epoch_metrics,
            epoch_tracker.epoch_metrics)


# weight_class = tf.constant([1.0] * 12, dtype=tf.float32)
# avoids class imbalances in cases where some rocks or soil types appear less frequently
# if using custom loss function - inside model.compile() - loss=loss_function(weight_class)
# - where weight_class = tf.constant([0.3, 0.7]) adjusting based on the imabalance

# learning rate scheduler good to prevent overshooting or poor local minimum
# factor = how much the learning rate is increased or reduced each time the validation loss stops improving
# patience = how many epochs the model will wait before reducing the learning rate if validation loss does not improve
# early stopping addition, in case loss stops improving before the 30 epochs are completed


# -------------- PERFORMANCE -------------------------------------------------------------------------------------------
def compute_metrics(y_true, y_pred, class_names=None, print_results=True):
    """
    Metric computation for segmentation.
    """
    print("Computing metrics for full images...")
    if len(y_true.shape) > 1:
        y_true_flat = y_true.flatten()
    else:
        y_true_flat = y_true

    if len(y_pred.shape) > 1:
        y_pred_flat = y_pred.flatten()
    else:
        y_pred_flat = y_pred

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


def plot_training(history):
    """
    Plotting function.
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


def visualize_cnn_results(X_test, y_test, predictions, num_patches=3, patch_indices=None, figsize=(20, 10), cmap_thermal='hot', cmap_segmentation='tab20'):
    """
    CNN results showing thermal images, ground truth labels, and predictions
    """

    if patch_indices is None:
        patch_indices = random.sample(range(len(X_test)), num_patches)
    else:
        num_patches = len(patch_indices)

    # One column for each part: input, label and prediciton
    fig, axes = plt.subplots(num_patches, 3, figsize=figsize, constrained_layout=True)
    if num_patches == 1:
        axes = axes.reshape(1, -1)

    if cmap_segmentation == 'tab20':
        colors = plt.cm.tab20(np.linspace(0, 1, 14))
        seg_cmap = ListedColormap(colors)
    else:
        seg_cmap = cmap_segmentation

    for i, patch_idx in enumerate(patch_indices):
        thermal_image = X_test[patch_idx]
        true_labels = y_test[patch_idx].squeeze()
        pred_labels = predictions[patch_idx]
        thermal_display = thermal_image[:, :, 1]

        # Thermal Image
        im1 = axes[i, 0].imshow(thermal_display, cmap=cmap_thermal, aspect='equal')
        axes[i, 0].set_title(f'Patch {patch_idx + 1}: Thermal Image')
        axes[i, 0].axis('off')

        # Ground Truth Labels
        im2 = axes[i, 1].imshow(true_labels, cmap=seg_cmap, vmin=0, vmax=13, aspect='equal')
        axes[i, 1].set_title(f'Patch {patch_idx + 1}: Ground Truth')
        axes[i, 1].axis('off')

        # Predictions
        im3 = axes[i, 2].imshow(pred_labels, cmap=seg_cmap, vmin=0, vmax=13, aspect='equal')
        axes[i, 2].set_title(f'Patch {patch_idx + 1}: Predictions')
        axes[i, 2].axis('off')

    cbar1 = plt.colorbar(im1, ax=axes[:, 0].ravel().tolist(), shrink=0.8, aspect=20)
    cbar1.set_label('Thermal Intensity', rotation=270, labelpad=15)

    cbar2 = plt.colorbar(im2, ax=axes[:, 1:].ravel().tolist(), shrink=0.8, aspect=20)
    cbar2.set_label('Class Labels', rotation=270, labelpad=15)
    cbar2.set_ticks(range(14))

    plt.show()

    print(f"\nMetrics for visualized patches:")
    for i, patch_idx in enumerate(patch_indices):
        true_labels = y_test[patch_idx].squeeze().flatten()
        pred_labels = predictions[patch_idx].flatten()
        accuracy = np.mean(true_labels == pred_labels)
        f1_macro = f1_score(true_labels, pred_labels, average='macro')
        print(f"Patch {patch_idx + 1}: Accuracy = {accuracy:.4f}, F1 (macro) = {f1_macro:.4f}")


# -------------- WORKFLOW ----------------------------------------------------------------------------------------------
(model, history, predictions, best_params,
 epoch_predictions, epoch_metrics, all_epoch_metrics) = train_model(
    X_train, y_train,
    X_valid, y_valid,
    X_test, y_test,
    use_optuna=False, n_trials=15, epochs=100)

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

visualize_cnn_results(X_test, y_test, predictions, figsize=(18, 10), patch_indices=[77, 134, 456])

'''
Maximum patch index:
    Villoslada:
        · thermal only - 900
        · thermal_optical - 76
        · thermal_sar - 263
        · all - 36
        · thermal_day - 864
        · thermal_night - 249
        · thermal_winter - 501
        · thermal_summer - 612
    Santa:
        · thermal only - 60
        · thermal_optical - 23
        · thermal_sar - 60
        · all - 5
        · thermal_day - 175
        · thermal_night - 81
        · thermal_winter - 111
        · thermal_summer - 145
    Puertollano:
        · thermal only - 360
        · thermal_optical - 40
        · thermal_sar - 173
        · all - 18
        · thermal_day - 202
        · thermal_night - 159
        · thermal_winter - 173
        · thermal_summer - 187
'''


model.save(os.path.join('models', 'CNN_categorical.keras'))
model.save(os.path.join('models', 'CNN_categorical.h5'))
print("Model saved successfully!")

print("\n=== TRAINING COMPLETE ===")
