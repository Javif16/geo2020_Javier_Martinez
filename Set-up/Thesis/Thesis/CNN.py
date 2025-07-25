'''
This file will contain the main Convolutional Neural Network, with which thermal image
classification will be performed and achieved.

Here, the program will also be trained and tested, with parameter tunning taking place
here as well.

It will be able to work both with a single image or a set of thermal iamges.

Extraction will be tested with different architectures to improve efficiency (ResNet
blocks) and batch normalization.

The program should output the  correct identification of different rock and soil types
when presented with either a single or a set of thermal images.
'''
from typing import List, Any

# GEOLOGICAL MAPS
'''
· Pre-labelled geological maps
· Converted into segmentation greyscale masks matching the input shape of images in the CNN with same CRS
· Each pixel corresponds to a specific class (rock/soil) with numbers (loam=1, peat=2, silt=3, etc.)

Evaluation:
    - overlay output of models over the geological maps and compare pixel by pixel
    - IoU and Dice Coefficient inside model (during training)
    - Confusion matrix and Accuracy outside of model'''

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
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.initializers import HeNormal
from tensorflow.keras.metrics import Precision, Recall, BinaryAccuracy, CategoricalAccuracy
from tensorflow.keras.models import load_model
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
'''
# -------------- LOADING DATA ------------------------------------------------------------------------------------------
# get datasets already prepared from thermal.py + check for their shapes
# thermal
X_train = np.load("X_train.npy")
X_val = np.load("X_val.npy")
X_test = np.load("X_test.npy")
# labels
y_train = np.load("y_train.npy")
y_val = np.load("y_val.npy")
y_test = np.load("y_test.npy")
# weights
w_train = np.load("weights_train.npy")
w_val = np.load("weights_val.npy")
w_test = np.load("weights_test.npy")
# dates
dates_train = np.load("date_indices_train.npy")
dates_val = np.load("date_indices_val.npy")
dates_test = np.load("date_indices_test.npy")

# ------------------------------------------------------------------------------
# X = input data (samples, 744, 1171, 6), y = labels (), w = weights (samples, 744, 1171)
# data has to be 'float32' for tensorflow computations
X_train = X_train.astype('float32')
X_val = X_val.astype('float32')
X_test = X_test.astype('float32')
# int - they are output
y_train = y_train.astype('int32')
y_val = y_val.astype('int32')
y_test = y_test.astype('int32')
# weights
w_train = w_train.astype('float32')
w_val = w_val.astype('float32')
w_test = w_test.astype('float32')

# channel decision
use_lst_only = False
use_emis_only = False
if use_lst_only:
    X_train = X_train[:, :, :, 0:1]
    X_val = X_val[:, :, :, 0:1]
    X_test = X_test[:, :, :, 0:1]
    input_channels = 1
    print("Using LST only (1 channel)")
elif use_emis_only:
    X_train = X_train[:, :, :, 1:0]
    X_val = X_val[:, :, :, 1:0]
    X_test = X_test[:, :, :, 1:0]
    input_channels = 5
else:
    input_channels = 6
    print("Using all channels")
print(f"Final input shapes:")
print(f"X_train: {X_train.shape}")
print(f"y_train: {y_train.shape}")
print(f"dates_train: {dates_train.shape}")
print(f"Label range: {np.min(y_train)} to {np.max(y_train)}")
'''

# -------------- LOADING -----------------------------------------------------------------------------------------------
data_path = "C:/Users/txiki/OneDrive/Documents/Studies/MSc_Geomatics/2Y/Thesis/Outputs/Villoslada/"
# data_path_optical = "C:/Users/txiki/OneDrive/Documents/Studies/MSc_Geomatics/2Y/Thesis/Outputs/Villoslada/"
# data_path_sar = "C:/Users/txiki/OneDrive/Documents/Studies/MSc_Geomatics/2Y/Thesis/Outputs/Villoslada/"
# data_path_all = "C:/Users/txiki/OneDrive/Documents/Studies/MSc_Geomatics/2Y/Thesis/Outputs/Villoslada/"
X_train = np.load(data_path + 'X_train.npy')
X_valid = np.load(data_path + 'X_valid.npy')
X_test = np.load(data_path + 'X_test.npy')
y_train = np.load(data_path + 'y_train.npy')
y_valid = np.load(data_path + 'y_valid.npy')
y_test = np.load(data_path + 'y_test.npy')

print("Dataset loaded successfully!")
print(f"X_train shape: {X_train.shape}")
print(f"y_train shape: {y_train.shape}")
print(f"X_valid shape: {X_valid.shape}")
print(f"y_valid shape: {y_valid.shape}")
print(f"X_test shape: {X_test.shape}")
print(f"y_test shape: {y_test.shape}")


# -------------- CNN U-Net ---------------------------------------------------------------------------------------------
def unet_model(size_input=(64, 64, 2), filters_base=16, classes=14, l2_reg=0.01):
    '''
    U-Net architecture with encoder and decoder.
    :param size_input: Tuple, input size of the image (height, width, channels).
    :param filters: Nº of filters in the first convolutional layer.
    :param classes: Nº of output classes.
    :param l2_reg: L2 regularization factor to apply to all Conv2D layers.
    :return: Compiled U-Net model.
    '''
    # Encoder
    def Encoder(inputs, filters, dropout, l2factor, max_pooling=True):
        '''
        Uses convolutional and pooling layers, in tandem with ReLU activation function to learn.
        Dropout additional to prevent overfitting problems.
        :return:
        Activation function values for next layer.
        Skip connection used in the decoder.
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
        Uses transpose convolutions to make images bigger (upscale) back to original size and merges with skip layer from Encoder.
        2 convolutional layers with 'same' padding increases depth of architecture for better predictions.
        :return:
        Decoded layer output.
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
    conv1, skip1 = Encoder(inputs, filters_base, dropout=0, l2factor=l2_reg, max_pooling=True)  # 372 x 585
    conv2, skip2 = Encoder(conv1, filters_base * 2, dropout=0, l2factor=l2_reg, max_pooling=True)  # 186 x 292
    conv3, skip3 = Encoder(conv2, filters_base * 4, dropout=0, l2factor=l2_reg, max_pooling=True)  # 93 x 146
    conv4, skip4 = Encoder(conv3, filters_base * 8, dropout=0.2, l2factor=l2_reg, max_pooling=True)  # 46 x 73
    conv5, skip5 = Encoder(conv4, filters_base * 16, dropout=0.2, l2factor=l2_reg, max_pooling=True)  # 23 x 36

    # bottleneck (last layer before upscaling)
    bottleneck, _ = Encoder(conv5, filters_base*32, dropout=0.3, l2factor=l2_reg*2, max_pooling=False)
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
    model = tf.keras.Model(inputs=inputs, outputs=output, name='unet_segmentation_cnn')
    model.summary()

    return model


# -------------- LOSS FUNCTION ---------------------------------------------------------------------------------
def combined_loss(class_weights, focal_alpha=0.25, focal_gamma=2.0, ce_weight=0.5):
    """
    Combines weighted categorical crossentropy with focal loss
    """
    class_weights_tensor = tf.constant(class_weights, dtype=tf.float32)

    def loss_fn(y_true, y_pred):
        y_true = tf.cast(y_true, tf.int32)

        # Categorical crossentropy with class weights
        ce_loss = tf.keras.losses.sparse_categorical_crossentropy(y_true, y_pred, from_logits=False)
        weights = tf.gather(class_weights_tensor, y_true)
        weighted_ce = tf.reduce_mean(ce_loss * weights)

        # Focal loss component
        y_pred_softmax = tf.nn.softmax(y_pred, axis=-1)
        y_pred_softmax = tf.clip_by_value(y_pred_softmax, 1e-8, 1.0 - 1e-8)
        y_true_one_hot = tf.one_hot(y_true, depth=14)

        ce_focal = -y_true_one_hot * tf.math.log(y_pred_softmax)
        p_t = tf.reduce_sum(y_true_one_hot * y_pred_softmax, axis=-1)
        focal_weight = focal_alpha * tf.pow(1 - p_t, focal_gamma)
        focal_loss = tf.reduce_mean(focal_weight * tf.reduce_sum(ce_focal, axis=-1))

        # Combine losses
        total_loss = ce_weight * weighted_ce + (1 - ce_weight) * focal_loss
        return total_loss

    return loss_fn


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


# -------------- TRAINING & EVALUATION ---------------------------------------------------------------------------------
def train_model(X_train, y_train, X_val, y_val, X_test, y_test, use_optuna=True, n_trials=30, epochs=30):
    """
    Train and evaluate model with optional Optuna hyperparameter optimization
    Enhanced to track predictions at each epoch for visualization
    """

    print("Starting model training...")
    print(f"Training on {len(X_train)} images, validating on {len(X_val)} images, testing on {len(X_test)} images")

    if use_optuna:
        print("=== STARTING OPTUNA HYPERPARAMETER OPTIMIZATION ===")

        def objective(trial):
            tf.keras.backend.clear_session()

            # Suggest hyperparameters
            learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-3, log=True)
            batch_size = trial.suggest_categorical('batch_size', [1, 2, 4])
            dropout_rate = trial.suggest_float('dropout_rate', 0.1, 0.5)
            l2_reg = trial.suggest_float('l2_regularization', 0.001, 0.01, 0.1, log=True)
            weight_strength = trial.suggest_float('weight_strength', 0.5, 2.0)
            filters_base = trial.suggest_categorical('filters_base', [16, 32, 64])

            try:
                # Create model with suggested hyperparameters
                model = unet_model(
                    size_input=(64, 64, 2),
                    classes=14,
                    l2_reg=l2_reg,
                    filters_base=filters_base)

                # Compile model
                optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
                loss = tf.keras.losses.SparseCategoricalCrossentropy()

                model.compile(
                    optimizer=optimizer,
                    loss=loss,
                    metrics=['accuracy', tf.keras.metrics.SparseCategoricalAccuracy()]
                )

                # Callbacks for optimization
                early_stopping = tf.keras.callbacks.EarlyStopping(
                    monitor='val_loss', patience=5, restore_best_weights=True, verbose=0
                )
                pruning_callback = optuna.integration.TFKerasPruningCallback(trial, 'val_loss')

                # Train with reduced epochs for optimization
                history = model.fit(
                    X_train, y_train,
                    validation_data=(X_val, y_val),
                    epochs=5,
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
        study = optuna.create_study(direction='minimize')
        study.optimize(objective, n_trials=n_trials)

        best_params = study.best_params
        print(f"Best parameters: {best_params}")

    else:
        # Use default parameters
        best_params = {
            'learning_rate': 0.0001,
            'l2factor': 0.001,
            'batch_size': 4,
            'filters_base': 16
        }
        # lr - 0.001 = better for 1 epoch
        # lr - 0.0001 = better for 20 epochs
        print("=== USING DEFAULT PARAMETERS (NO OPTUNA) ===")

    print("\n=== TRAINING FINAL MODEL WITH EPOCH TRACKING ===")

    # Train final model with best/default parameters
    tf.keras.backend.clear_session()

    # Create final model
    model = unet_model(
        size_input=(64, 64, 2),
        filters_base=best_params['filters_base'],
        classes=14,
        l2_reg=best_params['l2factor'])

    print("Model output shape:", model.output_shape)

    # Compile with optimized parameters
    optimizer = tf.keras.optimizers.Adam(learning_rate=best_params['learning_rate'])
    loss = tf.keras.losses.SparseCategoricalCrossentropy()
    # class_weights = {0: 0., 1: 1., 2: 1., 3: 1.,
                     # 4: 1., 5: 1., 6: 1., 7: 1.,
                     # 8: 1., 9: 1., 10: 1., 11: 1.,
                     # 12: 1.0, 13: 1.0}
    # loss = combined_loss(class_weights, focal_alpha=0.25, focal_gamma=2.0)

    model.compile(
        optimizer=optimizer,
        loss=loss,
        metrics=['accuracy', tf.keras.metrics.SparseCategoricalAccuracy()]
    )

    # Initialize the tracker
    epoch_tracker = EpochTracker(X_test, y_test, compute_metrics)

    # Other callbacks
    lr_callback = tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=3,
        min_lr=1e-7,
        verbose=1
    )
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=7,
        restore_best_weights=True
    )

    # Training with epoch tracking
    print("Training final CNN model...")
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=1,
        batch_size=best_params['batch_size'],
        callbacks=[lr_callback, early_stopping, epoch_tracker],
        verbose=1
    )

    # Final evaluation
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


def plot_training(history):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # loss
    axes[0].plot(history.history['loss'], label='Training Loss')
    axes[0].plot(history.history['val_loss'], label='Validation Loss')
    axes[0].set_title('Model Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].legend()
    axes[0].grid(True)

    # accuracy
    axes[1].plot(history.history['accuracy'], label='Training Accuracy')
    axes[1].plot(history.history['val_accuracy'], label='Validation Accuracy')
    axes[1].set_title('Model Accuracy')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy')
    axes[1].legend()
    axes[1].grid(True)

    plt.tight_layout()
    plt.show()


def visualize_cnn_results(X_test, y_test, predictions, num_patches=3, patch_indices=None,
                          figsize=(15, 5), cmap_thermal='hot', cmap_segmentation='tab20'):
    """
    Visualize CNN results showing thermal images, ground truth labels, and predictions

    Parameters:
    -----------
    X_test : numpy array, shape (n_samples, 64, 64, 2)
        Test thermal images with 2 channels
    y_test : numpy array, shape (n_samples, 64, 64, 1)
        Ground truth segmentation labels
    predictions : numpy array, shape (n_samples, 64, 64)
        Model predictions
    num_patches : int
        Number of patches to visualize (default: 3)
    patch_indices : list or None
        Specific patch indices to visualize. If None, random patches are selected
    figsize : tuple
        Figure size (width, height)
    cmap_thermal : str
        Colormap for thermal images
    cmap_segmentation : str
        Colormap for segmentation maps
    """

    # Select patches to visualize
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
        # Get data for this patch
        thermal_image = X_test[patch_idx]
        true_labels = y_test[patch_idx].squeeze()  # Remove channel dimension
        pred_labels = predictions[patch_idx]

        # Use first thermal channel for visualization (you can modify this)
        thermal_display = thermal_image[:, :, 0]

        # Column 1: Thermal Image
        im1 = axes[i, 0].imshow(thermal_display, cmap=cmap_thermal, aspect='equal')
        axes[i, 0].set_title(f'Patch {patch_idx + 1}: Thermal Image')
        axes[i, 0].axis('off')

        # Column 2: Ground Truth Labels
        im2 = axes[i, 1].imshow(true_labels, cmap=seg_cmap, vmin=0, vmax=13, aspect='equal')
        axes[i, 1].set_title(f'Patch {patch_idx + 1}: Ground Truth')
        axes[i, 1].axis('off')

        # Column 3: Predictions
        im3 = axes[i, 2].imshow(pred_labels, cmap=seg_cmap, vmin=0, vmax=13, aspect='equal')
        axes[i, 2].set_title(f'Patch {patch_idx + 1}: Predictions')
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

    # Print accuracy for visualized patches
    print(f"\nAccuracy for visualized patches:")
    for i, patch_idx in enumerate(patch_indices):
        true_labels = y_test[patch_idx].squeeze()
        pred_labels = predictions[patch_idx]
        accuracy = np.mean(true_labels == pred_labels)
        print(f"Patch {patch_idx + 1}: {accuracy:.4f}")


# -------------- WORKFLOW ----------------------------------------------------------------------------------------------
'''
fold_results = k_fold_cross_validation(
    X_train,
    y_train,
    w_train
)
'''

(model, history, predictions, best_params,
 epoch_predictions, epoch_metrics, all_epoch_metrics) = train_model(
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

plot_training(history)

visualize_cnn_results(X_test, y_test, predictions, patch_indices=[50, 690, 943])


model.save(os.path.join('models', 'CNN_categorical.keras'))  # Native Keras format
model.save(os.path.join('models', 'CNN_categorical.h5'))     # HDF5 format for compatibility
print("Model saved successfully!")

print("\n=== TRAINING COMPLETE ===")
