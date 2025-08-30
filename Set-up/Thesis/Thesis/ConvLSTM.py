'''
Here, the algorithm will receive a set of thermal images and output a correct classification
of the type of rock or soil the image contains.

It will order the images in a temporal sequence.

Then, with TimeDistributed, apply the trained CNN model across the sequence.

ConvLSTM layers will be used to analyse spatial and temporal dependencies + include dropouts
to control overfitting + batch normalization for stable training.
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


# -------------- LOADING DATA ------------------------------------------------------------------------------------------
data_path = 'E:/Studies/Thesis/THERMAL/Villoslada/Villoslada_thermal_sar/'
X_train = np.load(data_path + 'X_train_seq.npy')
X_valid = np.load(data_path + 'X_valid_seq.npy')
X_test = np.load(data_path + 'X_test_seq.npy')
y_train = np.load(data_path + 'y_train_seq.npy')
y_valid = np.load(data_path + 'y_valid_seq.npy')
y_test = np.load(data_path + 'y_test_seq.npy')

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

print("Dataset loaded successfully!")
print(f"X_train shape: {X_train.shape}")
print(f"y_train shape: {y_train.shape}")
print(f"X_valid shape: {X_valid.shape}")
print(f"y_valid shape: {y_valid.shape}")
print(f"X_test shape: {X_test.shape}")
print(f"y_test shape: {y_test.shape}")


# -------------- ConvLSTM U-Net ----------------------------------------------------------------------------------------
def clstm_unet_model(size_input=(5, 64, 64, 2), filters_base=16, classes=14, l2_reg=0.01):
    def Encoder(inputs, filters, dropout, l2factor, max_pooling=True):
        convolution = ConvLSTM2D(filters, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal',
                             kernel_regularizer=l2(l2factor), return_sequences=True)(inputs)
        convo = BatchNormalization()(convolution)

        convolution2 = ConvLSTM2D(filters, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal',
            kernel_regularizer=l2(l2factor), return_sequences=True)(convo)
        convo2 = BatchNormalization()(convolution2)
        if dropout > 0:
            convo2 = SpatialDropout3D(dropout)(convo2)
        if max_pooling:
            next_layer = MaxPooling3D(pool_size=(1, 2, 2))(convo2)
        else:
            next_layer = convo2
        skip_connection = convo2

        return next_layer, skip_connection

    def Decoder(previous_layer, skip, filters, l2factor):
        upscale = tf.keras.layers.Reshape(
            (1, previous_layer.shape[1], previous_layer.shape[2], previous_layer.shape[3]))(previous_layer)
        upscale = UpSampling3D(size=(1, 2, 2))(upscale)

        # Take only the last timestep from skip connection for concatenation
        skip_last = tf.keras.layers.Lambda(lambda x: x[:, -1:, :, :, :])(skip)
        combination = concatenate([upscale, skip_last], axis=4)

        # Use Conv2D instead of ConvLSTM2D in decoder since we're no longer processing sequences
        combination = tf.keras.layers.Reshape((combination.shape[2], combination.shape[3], combination.shape[4]))(
            combination)
        convolution = Conv2D(filters, (3, 3), activation='relu', padding='same',
                             kernel_initializer='he_normal', kernel_regularizer=l2(l2factor))(combination)
        convo = BatchNormalization()(convolution)

        return convo

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


'''
TimeDistributed CNN:
    cnn = base_cnn(input_shape)
    cnn_td = TimeDistributed(cnn)(inputs)  # CNN applied to each frame

    # ConvLSTM to process time-sequence
    x = ConvLSTM2D(64, (3, 3), activation='relu', padding='same', return_sequences=False)(cnn_td)
    x = BatchNormalization()(x)
    x = Dropout(0.2)(x)
'''


class EpochTracker(tf.keras.callbacks.Callback):
    """Custom callback to track predictions and metrics at each epoch"""

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
        """Define which epochs to store predictions for"""
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

        # Concatenate all predictions
        predictions = np.concatenate(predictions_list, axis=0)

        # Convert to class indices for metrics
        pred_classes = np.argmax(predictions, axis=-1)

        # Compute metrics
        metrics = self.compute_metrics_fn(self.y_test, pred_classes)

        # Always store metrics (lightweight)
        self.epoch_metrics.append(metrics)

        # Only store predictions for selected epochs (memory intensive)
        if current_epoch in self.epochs_to_store:
            print(f"  → Storing predictions for epoch {current_epoch} (selected epoch)")
            self.selected_epoch_predictions[current_epoch] = pred_classes.copy()
            self.selected_epoch_metrics[current_epoch] = metrics.copy()
        else:
            print(f"  → Skipping prediction storage for epoch {current_epoch}")

        # Print epoch summary
        print(f"Epoch {current_epoch} - Test Accuracy: {metrics['accuracy']:.4f}, "
              f"Mean IoU: {metrics['mean_iou']:.4f}")

        # Memory cleanup
        del predictions, pred_classes
        import gc
        gc.collect()


# -------------- LOSS FUNCTION -----------------------------------------------------------------------------------------
def simple_weighted_loss(class_weights):
    """Much simpler weighted loss function"""

    def loss_fn(y_true, y_pred):
        # Convert to sparse categorical crossentropy with class weights
        y_true = tf.squeeze(y_true, axis=-1)  # [4,64,64,1] -> [4,64,64]
        loss = tf.keras.losses.sparse_categorical_crossentropy(y_true, y_pred)

        # Apply class weights
        sample_weights = tf.gather([
            class_weights.get(0, 1.0), class_weights.get(1, 1.0), class_weights.get(2, 1.0),
            class_weights.get(3, 1.0), class_weights.get(4, 1.0), class_weights.get(5, 1.0),
            class_weights.get(6, 1.0), class_weights.get(7, 1.0), class_weights.get(8, 1.0),
            class_weights.get(9, 1.0), class_weights.get(10, 1.0), class_weights.get(11, 1.0),
            class_weights.get(12, 1.0), class_weights.get(13, 1.0)
        ], tf.cast(y_true, tf.int32))

        return tf.reduce_mean(loss * sample_weights)

    return loss_fn


# -------------- TRAINING ----------------------------------------------------------------------------------------------
def train_evaluate_model(X_train, y_train, X_val, y_val, X_test, y_test, use_optuna=True, n_trials=30, epochs=30):
    """
    Train and evaluate ConvLSTM model with optional Optuna hyperparameter optimization

    Parameters:
    -----------
    X_train, y_train, w_train : Training data and weights
    X_val, y_val, w_val : Validation data and weights
    X_test, y_test, w_test : Test data and weights
    use_optuna : bool, whether to use Optuna optimization
    n_trials : int, number of Optuna trials to run

    Returns:
    --------
    model : trained model
    history : training history
    y_pred : predictions on test set
    best_params : best hyperparameters found (or default if not using Optuna)
    """

    print("=== ConvLSTM U-Net Training ===")
    print("X_train shape:", X_train.shape)
    print("y_train shape:", y_train.shape)

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
                # Create model with suggested hyperparameters
                model = clstm_unet_model(
                    size_input=(5, 64, 64, 1),
                    filters=32,
                    classes=14)

                print("Model output shape:", model.output_shape)

                # Compile model
                optimizer = tf.keras.optimizers.Adam(learning_rate=best_params['learning_rate'])
                loss = tf.keras.losses.SparseCategoricalCrossentropy()

                model.compile(
                    optimizer=optimizer,
                    loss=loss,
                    metrics=['accuracy', 'precision', 'recall']
                )

                # Callbacks for optimization (shorter patience for faster trials)
                early_stopping = tf.keras.callbacks.EarlyStopping(
                    monitor='val_loss', patience=5, restore_best_weights=True, verbose=0
                )
                pruning_callback = optuna.integration.TFKerasPruningCallback(trial, 'val_loss')

                # Train with reduced epochs for optimization
                history = model.fit(
                    X_train, y_train,
                    validation_data=(X_val, y_val),
                    epochs=15,  # Reduced for faster optimization
                    batch_size=batch_size,
                    callbacks=[early_stopping, pruning_callback],
                    verbose=0
                )

                val_loss = min(history.history['val_loss'])
                return val_loss

            except Exception as e:
                print(f"Trial failed: {e}")
                return float('inf')

        # Run optimization
        study = optuna.create_study(
            direction='minimize',
            pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=5)
        )

        study.optimize(objective, n_trials=n_trials)

        print(f"\nOptimization completed!")
        print(f"Best parameters: {study.best_params}")
        print(f"Best validation loss: {study.best_value:.4f}")

        # Use best parameters
        best_params = study.best_params

    else:
        # Use default parameters
        best_params = {
            'learning_rate': 0.00001,
            'l2factor': 0.05,
            'batch_size': 4,
            'filters_base': 16
        }
        print("=== USING DEFAULT PARAMETERS (NO OPTUNA) ===")

    print("\n=== TRAINING FINAL ConvLSTM MODEL ===")

    # Train final model with best/default parameters
    tf.keras.backend.clear_session()

    model = clstm_unet_model(
        size_input=(5, 64, 64, 2),
        filters_base=best_params['filters_base'],
        classes=14,
        l2_reg=best_params['l2factor'])

    print("Model output shape:", model.output_shape)

    class_weights = {
        0: 1.0,  # NaNs (background)
        1: 3.0,  # Sand
        2: 3.0,  # Clay
        3: 1.0,  # Chalk
        4: 2.0,  # Silt - keep high
        5: 1.0,  # Peat
        6: 2.0,  # Loam
        7: 1.0,  # Detritic
        8: 2.0,  # Carbonate
        9: 3.0,  # Volcanic
        10: 1.0,  # Plutonic
        11: 1.0,  # Foliated
        12: 1.0,  # Non-Foliated
        13: 1.0  # Water
    }
    # Puertollano - 1, 2, 4, 6, 7, 8, 9, 11, 12
    # Santa Olalla - 1, 3, 4, 7, 8, 9, 10, 11, 12
    # Villoslada - 1, 2, 4, 5, 7, 8

    optimizer = tf.keras.optimizers.AdamW(learning_rate=best_params['learning_rate'], clipnorm=1.0)
    loss = tf.keras.losses.SparseCategoricalCrossentropy()
    # loss = simple_weighted_loss(class_weights)

    model.compile(
        optimizer=optimizer,
        loss=loss,
        metrics=['accuracy', tf.keras.metrics.SparseCategoricalAccuracy(), f1_score_keras]
    )

    # Initialize the callback
    epoch_tracker = EpochTracker(X_test, y_test, compute_metrics)

    # Callbacks for final training
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

    # Training with full epochs
    print("Training final ConvLSTM model...")
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=100,
        batch_size=best_params['batch_size'],
        callbacks=[lr_callback, early_stopping, epoch_tracker],
        verbose=1
    )

    # Evaluation
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

    return (model, history, predictions, best_params,
            epoch_tracker.selected_epoch_predictions,
            epoch_tracker.selected_epoch_metrics,
            epoch_tracker.epoch_metrics)


# -------------- PERFORMANCE -------------------------------------------------------------------------------------------
def compute_metrics(y_true, y_pred):
    """
    Compute metrics for segmentation task
    Args:
        y_true: Ground truth one-hot encoded labels (samples, height, width, classes)
        y_pred: Model predictions as probabilities (samples, height, width, classes)
    """
    print("Computing metrics for full images...")
    # Convert predictions to class indices if they're probabilities
    # Flatten for metric calculation
    y_true_flat = y_true.flatten()
    y_pred_flat = y_pred.flatten()

    # remove any padding/invalid values
    valid_mask = (y_true_flat >= 0) & (y_true_flat < 14)
    y_true_flat = y_true_flat[valid_mask]
    y_pred_flat = y_pred_flat[valid_mask]

    accuracy = accuracy_score(y_true_flat, y_pred_flat)
    precision = precision_score(y_true_flat, y_pred_flat, average='macro', zero_division=0)
    recall = recall_score(y_true_flat, y_pred_flat, average='macro', zero_division=0)
    f1 = f1_score(y_true_flat, y_pred_flat, average='macro', zero_division=0)

    # Calculate IoU and Dice for each class then average
    unique_classes = np.unique(np.concatenate([y_true_flat, y_pred_flat]))
    iou_values = []
    dice_values = []

    for cls in unique_classes:
        if cls < 0 or cls >= 14:
            continue

        y_true_cls = (y_true_flat == cls)
        y_pred_cls = (y_pred_flat == cls)

        intersection = np.logical_and(y_true_cls, y_pred_cls).sum()
        union = np.logical_or(y_true_cls, y_pred_cls).sum()

        if union > 0:
            iou = intersection/union
            dice = (2 * intersection) / (np.sum(y_true_cls) + np.sum(y_pred_cls) + 1e-7)
        else:
            iou = 0.0
            dice = 0.0
        iou_values.append(iou)
        dice_values.append(dice)

    mean_iou = np.mean(iou_values)
    mean_dice = np.mean(dice_values)

    metrics = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'mean_iou': mean_iou,
        'mean_dice': mean_dice,
    }

    return metrics


from tensorflow.keras import backend as K

def f1_score_keras(y_true, y_pred):
    """
    F1 score metric for Keras - works with multi-class segmentation
    """
    # Convert predictions to class indices
    y_pred_classes = K.argmax(y_pred, axis=-1)

    # Cast to same type
    y_true = K.cast(K.flatten(y_true), tf.float32)
    y_pred_classes = K.cast(K.flatten(y_pred_classes), tf.float32)

    # Calculate overall precision and recall (micro-average)
    true_positives = K.sum(K.cast(K.equal(y_true, y_pred_classes), tf.float32))
    total_samples = K.cast(K.shape(y_true)[0], tf.float32)

    # This gives us accuracy, which for multiclass can serve as a proxy
    accuracy = true_positives / total_samples

    # For multiclass, we'll return accuracy as F1 approximation
    return accuracy


# -------------- EVALUATION --------------------------------------------------------------------------------------------
def plot_training(history):
    """
    Enhanced plotting function that includes F1 score if available
    """
    # Check what metrics are available in history
    available_metrics = list(history.history.keys())
    has_f1 = any('f1' in metric.lower() for metric in available_metrics)

    # Determine number of subplots based on available metrics
    if has_f1:
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    else:
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Plot 1: Loss
    axes[0].plot(history.history['loss'], label='Training Loss')
    if 'val_loss' in history.history:
        axes[0].plot(history.history['val_loss'], label='Validation Loss')
    axes[0].set_title('Model Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].legend()
    axes[0].grid(True)

    # Plot 2: Accuracy
    axes[1].plot(history.history['accuracy'], label='Training Accuracy')
    if 'val_accuracy' in history.history:
        axes[1].plot(history.history['val_accuracy'], label='Validation Accuracy')
    axes[1].set_title('Model Accuracy')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy')
    axes[1].legend()
    axes[1].grid(True)

    # Plot 3: F1 Score
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


def visualize_convlstm_results(X_test, y_test, predictions, num_patches=3, patch_indices=None,
                               timestep_to_visualize=-1, figsize=(15, 5), cmap_thermal='hot',
                               cmap_segmentation='tab20'):
    """
    Visualize ConvLSTM results showing thermal image sequences, ground truth labels, and predictions

    Parameters:
    -----------
    X_test : numpy array, shape (n_samples, 5, 64, 64, 2)
        Test thermal image sequences with 5 timesteps and 2 channels
    y_test : numpy array, shape (n_samples, 64, 64, 1) or (n_samples, 64, 64)
        Ground truth segmentation labels (final timestep)
    predictions : numpy array, shape (n_samples, 64, 64)
        Model predictions
    num_patches : int
        Number of sequences to visualize (default: 3)
    patch_indices : list or None
        Specific sequence indices to visualize. If None, random sequences are selected
    timestep_to_visualize : int
        Which timestep to visualize from the sequence (default: -1 for last timestep)
    figsize : tuple
        Figure size (width, height)
    cmap_thermal : str
        Colormap for thermal images
    cmp_segmentation : str
        Colormap for segmentation maps
    """

    # Select sequences to visualize
    if patch_indices is None:
        patch_indices = random.sample(range(len(X_test)), num_patches)
    else:
        num_patches = len(patch_indices)

    # Create figure with subplots
    fig, axes = plt.subplots(num_patches, 3, figsize=figsize)
    if num_patches == 1:
        axes = axes.reshape(1, -1)

    # Set up colormap for segmentation (14 classes)
    if cmap_segmentation == 'tab20':
        colors = plt.cm.tab20(np.linspace(0, 1, 14))
        seg_cmap = ListedColormap(colors)
    else:
        seg_cmap = cmap_segmentation

    for i, patch_idx in enumerate(patch_indices):
        # Get data for this sequence
        thermal_sequence = X_test[patch_idx]  # Shape: (5, 64, 64, 2)
        true_labels = y_test[patch_idx]
        pred_labels = predictions[patch_idx]

        # Handle y_test shape variations
        if len(true_labels.shape) == 3 and true_labels.shape[-1] == 1:
            true_labels = true_labels.squeeze(-1)  # Remove channel dimension

        # Select timestep to visualize from the sequence
        thermal_frame = thermal_sequence[timestep_to_visualize]  # Shape: (64, 64, 2)

        # Use first thermal channel for visualization
        thermal_display = thermal_frame[:, :, 0]

        # Column 1: Thermal Image (selected timestep)
        im1 = axes[i, 0].imshow(thermal_display, cmap=cmap_thermal, aspect='equal')
        timestep_label = f"t={timestep_to_visualize}" if timestep_to_visualize >= 0 else "t=final"
        axes[i, 0].set_title(f'Sequence {patch_idx + 1}: Thermal ({timestep_label})')
        axes[i, 0].axis('off')

        # Column 2: Ground Truth Labels
        im2 = axes[i, 1].imshow(true_labels, cmap=seg_cmap, vmin=0, vmax=13, aspect='equal')
        axes[i, 1].set_title(f'Sequence {patch_idx + 1}: Ground Truth')
        axes[i, 1].axis('off')

        # Column 3: Predictions
        im3 = axes[i, 2].imshow(pred_labels, cmap=seg_cmap, vmin=0, vmax=13, aspect='equal')
        axes[i, 2].set_title(f'Sequence {patch_idx + 1}: Predictions')
        axes[i, 2].axis('off')

    # Add colorbars
    # Thermal colorbar
    cbar1 = plt.colorbar(im1, ax=axes[:, 0].ravel().tolist(), shrink=0.8, aspect=20)
    cbar1.set_label('Thermal Intensity', rotation=270, labelpad=15)

    # Segmentation colorbar
    cbar2 = plt.colorbar(im2, ax=axes[:, 1:].ravel().tolist(), shrink=0.8, aspect=20)
    cbar2.set_label('Class Labels', rotation=270, labelpad=15)
    cbar2.set_ticks(range(14))

    plt.tight_layout()
    plt.show()

    # Print accuracy for visualized sequences
    print(f"\nAccuracy for visualized sequences:")
    for i, patch_idx in enumerate(patch_indices):
        true_labels = y_test[patch_idx]
        if len(true_labels.shape) == 3 and true_labels.shape[-1] == 1:
            true_labels = true_labels.squeeze(-1)
        pred_labels = predictions[patch_idx]
        accuracy = np.mean(true_labels == pred_labels)
        print(f"Sequence {patch_idx + 1}: {accuracy:.4f}")


# -------------- WORKFLOW ----------------------------------------------------------------------------------------------
(model, history, predictions, best_params,
 epoch_predictions, epoch_metrics, all_epoch_metrics) = train_evaluate_model(
    X_train, y_train,
    X_valid, y_valid,
    X_test, y_test,
    use_optuna=False, n_trials=5, epochs=30)

final_metrics = compute_metrics(y_test, predictions)

print("\n=== FINAL PERFORMANCE METRICS ===")
for metric_name, metric_value in final_metrics.items():
    print(f"{metric_name}: {metric_value:.4f}")

print("\n=== BEST PARAMETERS ===")
for param_name, param_value in best_params.items():
    print(f"{param_name}: {param_value}")

visualize_convlstm_results(
    X_test, y_test, predictions, patch_indices=[5, 10, 11], timestep_to_visualize=-1)

plot_training(history)

'''
Maximum patch index:
    Villoslada:
        · thermal only - 
        · thermal_optical - 65
        · thermal_sar - 13
        · all - 10
        · thermal_day - 82
        · thermal_night - 277
        · thermal_winter - 165
        · thermal_summer - 198
    Santa:
        · thermal only - 83
        · thermal_optical - 7
        · thermal_sar - 15
        · all - 1 
        · thermal_day - 57
        · thermal_night - 26
        · thermal_winter - 35
        · thermal_summer - 46
    Puertollano:
        · thermal only - 63
        · thermal_optical - 6
        · thermal_sar - 25
        · all - 1
        · thermal_day - 118
        · thermal_night - 52
        · thermal_winter - 56
        · thermal_summer - 55
'''

# ---------------- SAVING ----------------------------------------------------------------------------------------------
model.save(os.path.join('models', 'ConvLSTM.keras'))
model.save(os.path.join('models', 'ConvLSTM.h5'))
