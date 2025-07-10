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
import cv2
import rasterio
import tensorflow as tf
from tensorflow.keras import backend
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Conv3D, MaxPooling2D, MaxPooling3D, Dense, Flatten, Dropout
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


# Set memory growth to avoid memory allocation issues
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(f"GPU memory growth error: {e}")

# Set mixed precision for memory efficiency
tf.keras.mixed_precision.set_global_policy('mixed_float16')


# -------------- LOADING DATA ------------------------------------------------------------------------------------------
# get datasets already prepared from thermal.py + check for their shapes
X_train = np.load("X_train_lstm.npy")
X_val = np.load("X_val_lstm.npy")
X_test = np.load("X_test_lstm.npy")
y_train = np.load("y_train_lstm.npy")
y_val = np.load("y_val_lstm.npy")
y_test = np.load("y_test_lstm.npy")
w_train = np.load("weights_train_lstm.npy")
w_val = np.load("weights_val_lstm.npy")
w_test = np.load("weights_test_lstm.npy")
pos_test = np.load("positions_test_lstm.npy")
date_test = np.load("dates_test_lstm.npy")
'''
print("=== ORIGINAL DATA SHAPES ===")
print(f"X_train shape: {X_train.shape}")
print(f"X_val shape: {X_val.shape}")
print(f"X_test shape: {X_test.shape}")
print(f"y_train shape: {y_train.shape}")
print(f"y_val shape: {y_val.shape}")
print(f"y_test shape: {y_test.shape}")
print(f"w_train shape: {w_train.shape}")
print(f"w_val shape: {w_val.shape}")
print(f"w_test shape: {w_test.shape}")
'''

print("X_train shape before slicing:", X_train.shape)

# Only LST - (sequences, 5, 64, 64, 1)
X_train = X_train[:, :, :, :, 0:1]
X_val = X_val[:, :, :, :, 0:1]
X_test = X_test[:, :, :, :, 0:1]

# Only Emissivity - (sequences, 5, 64, 64, 5)
# X_train = X_train[:, :, :, 1:]
# X_val = X_val[:, :, :, 1:]
# X_test = X_test[:, :, :, 1:]


# -------------- ConvLSTM U-Net ----------------------------------------------------------------------------------------
def clstm_unet_model(size_input=(5, 64, 64, 1), filters=32, classes=14):
    def Encoder(inputs, filters, dropout, l2factor=0.1, max_pooling=True):
        convolution = ConvLSTM2D(filters, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal',
                             kernel_regularizer=l2(l2factor), return_sequences=True)(inputs)
        convo = BatchNormalization()(convolution)
        if dropout > 0:
            convo = Dropout(dropout)(convo)
        if max_pooling:
            next_layer = MaxPooling3D(pool_size=(1, 2, 2))(convo)
        else:
            next_layer = convo

        skip_connection = convo

        return next_layer, skip_connection

    def Decoder(previous_layer, skip, filters):
        upscale = UpSampling3D(size=(1, 2, 2))(previous_layer)
        combination = concatenate([upscale, skip], axis=4)
        # axis=-1 is for taking always last dimension, regardless of dimension nº

        convolution = ConvLSTM2D(filters, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal',
                                 return_sequences=True)(combination)
        convo = BatchNormalization()(convolution)

        return convo

    inputs = tf.keras.layers.Input(shape=size_input)

    # encoder
    conv1 = Encoder(inputs, filters=32, dropout=0, l2factor=0.1, max_pooling=True)
    conv2 = Encoder(conv1[0], filters=64, dropout=0, l2factor=0.1, max_pooling=True)
    conv3 = Encoder(conv2[0], filters=128, dropout=0, l2factor=0.1, max_pooling=True)
    conv4 = Encoder(conv3[0], filters=256, dropout=0.3, l2factor=0.1, max_pooling=True)

    # bottleneck
    bottleneck = Encoder(conv4[0], filters=512, dropout=0.3, l2factor=0.1, max_pooling=False)

    # decoder
    upsc1 = Decoder(bottleneck[0], conv4[1], filters=256)
    upsc2 = Decoder(upsc1, conv3[1], filters=128)
    upsc3 = Decoder(upsc2, conv2[1], filters=64)
    upsc4 = Decoder(upsc3, conv1[1], filters=32)

    # layer to get to proper size - single segmentation map
    conv10 = Conv3D(filters, (1, 3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(upsc4)
    # Conv2D used to pass from 5D tensor to 4D tensor
    # better for single map output - no need for sequence-for-sequence predictions

    # final layer
    conv11 = Conv3D(classes, (1, 1, 1), padding='same', activation='softmax', dtype='float32')(conv10)

    model = tf.keras.Model(inputs=inputs, outputs=conv11)
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


# -------------- TRAINING ----------------------------------------------------------------------------------------------
def loss_function(w_train, weight_strength=1.5):
    def loss(y_true, y_pred):
        batch_size = tf.shape(y_true)[0]
        # Handle batch size properly - use modulo to cycle through weights if needed
        batch_indices = tf.range(batch_size) % tf.shape(w_train)[0]
        w_batch = tf.gather(w_train, batch_indices)

        # Reshape tensors for proper calculation
        num_classes = tf.shape(y_true)[4]
        # Flatten tensors for element-wise operations
        y_true_flat = tf.reshape(y_true, [-1, num_classes])
        y_pred_flat = tf.reshape(y_pred, [-1, num_classes])
        w_flat = tf.reshape(w_batch, [-1])

        # Use epsilon for numerical stability
        epsilon = tf.keras.backend.epsilon()

        # Clip prediction values to avoid log(0)
        y_pred_flat = tf.clip_by_value(y_pred_flat, epsilon, 1 - epsilon)

        tf.print(y_true_flat[0])
        tf.print(y_pred_flat[0])
        # ----------------------------------------------------------------

        # ====== INVERSE CLASS FREQUENCY WEIGHTING ======
        class_counts = tf.reduce_sum(y_true_flat, axis=0)
        class_counts = class_counts + epsilon  # epsilon to avoid division by 0
        # Total samples and class frequencies
        total_samples = tf.reduce_sum(class_counts)
        class_frequencies = class_counts / total_samples

        # inverse weighting (stronger than square root)
        class_weights = 1.0 / (class_frequencies + 0.01)  # Small constant to prevent extreme weights

        # control how aggressive the weighting is (1.0, 0.5, 1.5)
        class_weights = tf.pow(class_weights, weight_strength)

        # clip weights to prevent extreme values + normalization (consistent loss scale)
        class_weights = tf.clip_by_value(class_weights, 0.1, 10.0)
        class_weights = class_weights / tf.reduce_mean(class_weights)

        # weighted cross-entropy
        weighted_cce_per_pixel = -tf.reduce_sum(class_weights * y_true_flat * tf.math.log(y_pred_flat), axis=-1)
        final_weighted_loss = weighted_cce_per_pixel * w_flat  # original sample weights

        # Return mean
        return tf.reduce_mean(final_weighted_loss)

    return loss


def train_evaluate_model(X_train, y_train, w_train, X_val, y_val, w_val,
                         X_test, y_test, w_test, use_optuna=True, n_trials=30):
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
    print("w_train shape:", w_train.shape)

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
                optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate, clipnorm=1.0)
                loss = loss_function(w_train, weight_strength=weight_strength)

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
                    sample_weight=w_train,
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
            'learning_rate': 0.0001,
            'batch_size': 32,
            'weight_strength': 1.0
        }
        print("=== USING DEFAULT PARAMETERS (NO OPTUNA) ===")

    print("\n=== TRAINING FINAL ConvLSTM MODEL ===")

    # Train final model with best/default parameters
    tf.keras.backend.clear_session()

    model = clstm_unet_model(
        size_input=(5, 64, 64, 1),
        filters=32,
        classes=14)

    print("Model output shape:", model.output_shape)

    optimizer = tf.keras.optimizers.Adam(
        learning_rate=best_params['learning_rate'],
        clipnorm=1.0
    )
    loss = loss_function(w_train, weight_strength=best_params['weight_strength'])

    model.compile(
        optimizer=optimizer,
        loss=loss,
        metrics=[
            'accuracy',
            tf.keras.metrics.Precision(name='precision'),
            tf.keras.metrics.Recall(name='recall'),
            tf.keras.metrics.CategoricalAccuracy(name='categorical_accuracy')
        ]
    )

    # Custom callback to store predictions at each epoch
    class EpochPredictionCallback(tf.keras.callbacks.Callback):
        def __init__(self, X_test, y_test):
            super().__init__()
            # first and last 3
            first_3 = X_test[:3]
            last_3 = X_test[-3:]
            self.X_test = np.concatenate([first_3, last_3], axis=0)

            first_3_labels = y_test[:3]
            last_3_labels = y_test[-3:]
            self.y_test = np.concatenate([first_3_labels, last_3_labels], axis=0)

            self.epoch_predictions = []
            self.epoch_metrics = []

        def on_epoch_end(self, epoch, logs=None):
            # Make predictions on test set
            predictions = self.model.predict(self.X_test, verbose=0)
            self.epoch_predictions.append(predictions)

            # Calculate metrics for this epoch
            metrics = compute_metrics(self.y_test, predictions)
            self.epoch_metrics.append(metrics)

            print(f"Epoch {epoch + 1} - Test Accuracy: {metrics['accuracy']:.4f}, "
                  f"Test Mean IoU: {metrics['mean_iou']:.4f}")

    # Initialize the callback
    epoch_callback = EpochPredictionCallback(X_test, y_test)

    # Callbacks for final training
    lr_callback = tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=5,
        min_lr=1e-6,
        verbose=1
    )
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=10,
        restore_best_weights=True
    )

    # Training with full epochs
    print("Training final ConvLSTM model...")
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        sample_weight=w_train,
        epochs=1,
        batch_size=best_params['batch_size'],
        callbacks=[lr_callback, early_stopping, epoch_callback],
        verbose=1
    )

    # Evaluation
    print("\n=== FINAL EVALUATION ===")
    test_results = model.evaluate(X_test, y_test, sample_weight=w_test, verbose=0)
    test_loss, test_accuracy, test_precision, test_recall, test_categorical = test_results

    print(f"Test loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_accuracy:.4f}")
    print(f"Test Precision: {test_precision:.4f}")
    print(f"Test Recall: {test_recall:.4f}")
    print(f"Test Categorical: {test_categorical:.4f}")

    # Predictions
    y_pred = model.predict(X_test)

    print(f"y_test shape: {y_test.shape}")
    print(f"y_pred shape: {y_pred.shape}")

    return model, history, y_pred, best_params, epoch_callback.epoch_predictions, epoch_callback.epoch_metrics


# -------------- PERFORMANCE -------------------------------------------------------------------------------------------
def compute_metrics(y_true, y_pred):

    # Convert predictions to class indices if they're probabilities
    if y_pred.ndim == 5 and y_pred.shape[-1] > 1:  # Multi-class probabilities
        y_pred_classes = np.argmax(y_pred, axis=-1)
    else:
        y_pred_classes = y_pred

    # Convert true labels to class indices if they're one-hot encoded
    if y_true.ndim == 5 and y_true.shape[-1] > 1:  # One-hot encoded
        y_true_classes = np.argmax(y_true, axis=-1)
    else:
        y_true_classes = y_true

    # Flatten for metric calculation
    y_true_flat = y_true_classes.flatten()
    y_pred_flat = y_pred_classes.flatten()

    print(f"Flattened y_true shape: {y_true_flat.shape}")
    print(f"Flattened y_pred shape: {y_pred_flat.shape}")

    # Ensure both arrays have the same shape
    min_length = min(len(y_true_flat), len(y_pred_flat))
    y_true_flat = y_true_flat[:min_length]
    y_pred_flat = y_pred_flat[:min_length]

    # Use macro averaging for multi-class metrics
    precision_value = precision_score(y_true_flat, y_pred_flat, average='macro', zero_division=0)
    recall_value = recall_score(y_true_flat, y_pred_flat, average='macro', zero_division=0)
    accuracy_value = accuracy_score(y_true_flat, y_pred_flat)
    f1 = f1_score(y_true_flat, y_pred_flat, average='macro', zero_division=0)

    # Calculate IoU and Dice for each class then average
    unique_classes = np.unique(np.concatenate([y_true_flat, y_pred_flat]))
    iou_values = []
    dice_values = []

    for cls in unique_classes:
        y_true_cls = (y_true_flat == cls)
        y_pred_cls = (y_pred_flat == cls)

        intersection = np.logical_and(y_true_cls, y_pred_cls).sum()
        union = np.logical_or(y_true_cls, y_pred_cls).sum()

        iou = intersection / (union + 1e-7) if union > 0 else 0
        dice = (2 * intersection) / (np.sum(y_true_cls) + np.sum(y_pred_cls) + 1e-7)

        iou_values.append(iou)
        dice_values.append(dice)

    mean_iou = np.mean(iou_values)
    mean_dice = np.mean(dice_values)

    metrics = {
        'accuracy': accuracy_value,
        'precision': precision_value,
        'recall': recall_value,
        'f1_score': f1,
        'mean_iou': mean_iou,
        'mean_dice': mean_dice,
        'class_iou': iou_values,
        'class_dice': dice_values
    }

    return metrics

'''
# K-Fold Cross Validation Loop
def k_fold_cross_validation(model_func, X_train, y_train, k=5):
    kf = KFold(n_splits=k, shuffle=True, random_state=42)
    X_train = X_train.reshape(-1, 4, 4, 7)
    y_train = y_train.reshape(-1, 4, 4, 7)
    fold_results = {'accuracy': [], 'precision': [], 'recall': [], 'f1_score': [], 'iou': []}

    for fold, (train_idx, val_idx) in enumerate(kf.split(X_train)):
        print(f"\nFold {fold + 1}/{k}")

        X_train_fold, X_val_fold = X_train[train_idx], X_train[val_idx]
        y_train_fold, y_val_fold = y_train[train_idx], y_train[val_idx]

        model = model_func()
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

        history = model.fit(
            X_train_fold, y_train_fold,
            validation_data=(X_val_fold, y_val_fold),
            batch_size=4, epochs=30, verbose=1, callbacks=callbacks
        )

        evaluate_loss(history)

        y_pred_fold = model.predict(X_val_fold)
        precision, recall, accuracy, f1_score, iou, _ = compute_metrics(y_val_fold, y_pred_fold)

        fold_results['accuracy'].append(accuracy)
        fold_results['precision'].append(precision)
        fold_results['recall'].append(recall)
        fold_results['f1_score'].append(f1_score)
        fold_results['iou'].append(iou)

    print("\nCross-Validation Results:")
    for metric, values in fold_results.items():
        print(f"Mean {metric.capitalize()}: {np.mean(values):.4f}")

    return fold_results
'''


# -------------- EVALUATION --------------------------------------------------------------------------------------------
def plot_training(history):
    plt.figure(figsize=(12, 4))
    metrics = ['loss', 'accuracy', 'recall', 'precision']
    for i, metric in enumerate(metrics, 1):
        plt.subplot(1, 4, i)
        plt.plot(history.history[metric], label=f'Train {metric}')
        plt.plot(history.history[f'val_{metric}'], label=f'Validation {metric}')
        plt.title(f'{metric.capitalize()} Over Epochs')
        plt.xlabel('Epoch')
        plt.ylabel(metric.capitalize())
        plt.legend()
    plt.tight_layout()
    plt.show()


def visualize_predictions(X_test, y_test, epoch_predictions, epoch_metrics,
                                               original_shape=(744, 1171),
                                               patch_size=64,
                                               stride=64,
                                               num_images=1,
                                               timestep_to_visualize=-1):
    """
    Enhanced visualization for ConvLSTM that shows predictions from each epoch for comparison.
    Handles sequences of shape (sequences, sequence_length, 64, 64, channels).

    Args:
        X_test: Test sequence patches array (N_sequences, seq_length, H, W, C)
        y_test: Ground truth label sequences (N_sequences, seq_length, H, W, classes) - one-hot encoded
        epoch_predictions_list: List of predictions from each epoch (same shape as y_test)
        original_shape: Shape of the original full image (H, W)
        patch_size: Size of each square patch
        stride: Step size between patches
        num_images: Number of full images to reconstruct and display
        timestep_to_visualize: Which timestep to use for reconstruction (-1 for last, 0 for first, etc.)
    """

    print("=== VISUALIZING ConvLSTM EPOCH-BY-EPOCH PREDICTIONS ===")

    # Load the test positions and date indices for LSTM sequences
    pos_test = np.load("positions_test_lstm.npy")  # Shape: (N_sequences, seq_length, 2)
    date_test = np.load("dates_test_lstm.npy")  # Shape: (N_sequences, seq_length)

    # Handle timestep selection
    if timestep_to_visualize == -1:
        timestep_idx = X_test.shape[1] - 1  # Last timestep
    else:
        timestep_idx = timestep_to_visualize

    print(f"Using timestep {timestep_idx} for visualization")

    # Extract data from the selected timestep
    X_test_2d = X_test[:, timestep_idx, :, :, :]  # (N_sequences, H, W, C)
    y_test_2d = y_test[:, timestep_idx, :, :, :]  # (N_sequences, H, W, classes)
    pos_test_2d = pos_test[:, timestep_idx, :]  # (N_sequences, 2)
    date_test_2d = date_test[:, timestep_idx]  # (N_sequences,)

    # Calculate expected grid dimensions
    h, w = original_shape
    patches_per_row = (w - patch_size) // stride + 1
    patches_per_col = (h - patch_size) // stride + 1
    patches_per_image = patches_per_row * patches_per_col

    print(f"Original image shape: {original_shape}")
    print(f"Patches per row: {patches_per_row}, Patches per column: {patches_per_col}")
    print(f"Total patches per image expected: {patches_per_image}")
    print(f"Available test sequences: {len(X_test)}")
    print(f"Sequence length: {X_test.shape[1]}")
    print(f"Number of epochs tracked: {len(epoch_predictions)}")

    # Group patches by date AND position (using the selected timestep)
    date_position_to_patches = defaultdict(list)
    for idx, ((i, j), date_idx) in enumerate(zip(pos_test_2d, date_test_2d)):
        date_position_to_patches[(date_idx, i, j)].append(idx)

    # Group by date
    patches_per_date = defaultdict(list)
    for idx, date_idx in enumerate(date_test_2d):
        patches_per_date[date_idx].append(idx)

    print(f"Available dates in test set: {sorted(patches_per_date.keys())}")

    # Generate all expected positions for a complete image
    expected_positions = []
    for i in range(0, h - patch_size + 1, stride):
        for j in range(0, w - patch_size + 1, stride):
            expected_positions.append((i, j))

    # Get available dates and limit to requested number of images
    available_dates = sorted(patches_per_date.keys())
    num_images = min(num_images, len(available_dates))

    # Plot metrics evolution first
    print("\n=== PLOTTING METRICS EVOLUTION ===")
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()

    metrics_to_plot = ['accuracy', 'precision', 'recall', 'f1_score', 'mean_iou', 'mean_dice']

    for i, metric in enumerate(metrics_to_plot):
        values = [epoch[metric] for epoch in epoch_metrics]
        axes[i].plot(range(1, len(values) + 1), values, 'b-o', linewidth=2, markersize=6)
        axes[i].set_title(f'{metric.replace("_", " ").title()} Evolution', fontsize=12)
        axes[i].set_xlabel('Epoch')
        axes[i].set_ylabel(metric.replace("_", " ").title())
        axes[i].grid(True, alpha=0.3)
        axes[i].set_ylim(0, 1)

        # Add value annotations
        for j, val in enumerate(values):
            axes[i].annotate(f'{val:.3f}', (j + 1, val), textcoords="offset points",
                             xytext=(0, 10), ha='center', fontsize=8)

    plt.tight_layout()
    plt.savefig('convlstm_epoch_metrics_evolution.png', dpi=300, bbox_inches='tight')
    plt.show()

    # Now visualize reconstructed images for each epoch
    for img_idx in range(num_images):
        date_idx = available_dates[img_idx]
        print(f"\n--- Processing Image {img_idx + 1} from date index {date_idx} (timestep {timestep_idx}) ---")

        # Reconstruct ground truth and thermal image (same for all epochs)
        channels_X = X_test_2d.shape[-1]
        channels_y = y_test_2d.shape[-1] if y_test_2d.ndim == 4 else 1

        reconstructed_thermal = np.zeros((h, w, channels_X), dtype=np.float32)
        reconstructed_labels = np.zeros((h, w, channels_y), dtype=np.float32)

        filled_positions = []

        # Place patches only from this specific date (for ground truth and thermal)
        for (i, j) in expected_positions:
            key = (date_idx, i, j)
            if key in date_position_to_patches:
                patch_indices = date_position_to_patches[key]
                patch_idx = patch_indices[0]

                patch_thermal = X_test_2d[patch_idx]
                patch_labels = y_test_2d[patch_idx]

                reconstructed_thermal[i:i + patch_size, j:j + patch_size, :] = patch_thermal
                reconstructed_labels[i:i + patch_size, j:j + patch_size, :] = patch_labels
                filled_positions.append((i, j))

        coverage = len(filled_positions) / len(expected_positions) * 100
        print(f"Coverage: {coverage:.1f}%")

        # Prepare ground truth visualization
        thermal_img = reconstructed_thermal[:, :, 0]
        thermal_masked = np.ma.masked_where(thermal_img == 0, thermal_img)

        if reconstructed_labels.shape[-1] > 1:
            label_img = np.argmax(reconstructed_labels, axis=-1)
            label_sum = np.sum(reconstructed_labels, axis=-1)
        else:
            label_img = reconstructed_labels[:, :, 0]
            label_sum = reconstructed_labels[:, :, 0]
        label_masked = np.ma.masked_where(label_sum == 0, label_img)

        # Create epoch-by-epoch predictions visualization
        num_epochs = len(epoch_predictions)
        cols = min(4, num_epochs + 2)  # +2 for thermal and ground truth
        rows = max(1, (num_epochs + 2) // cols)
        if (num_epochs + 2) % cols != 0:
            rows += 1

        fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 5 * rows))
        if rows == 1:
            axes = axes.reshape(1, -1)

        plot_idx = 0

        # Plot thermal input
        if plot_idx < len(axes.flat):
            im = axes.flat[plot_idx].imshow(thermal_masked, cmap='hot', aspect='equal', origin='upper')
            axes.flat[plot_idx].set_title(f'Input Thermal\n(Date {date_idx}, t={timestep_idx})', fontsize=10)
            axes.flat[plot_idx].axis('off')
            plt.colorbar(im, ax=axes.flat[plot_idx], shrink=0.8)
            plot_idx += 1

        # Plot ground truth
        if plot_idx < len(axes.flat):
            im = axes.flat[plot_idx].imshow(label_masked, cmap='tab20', aspect='equal',
                                            vmin=0, vmax=13, origin='upper')
            axes.flat[plot_idx].set_title('Ground Truth\nLabels', fontsize=10)
            axes.flat[plot_idx].axis('off')
            plt.colorbar(im, ax=axes.flat[plot_idx], shrink=0.8)
            plot_idx += 1

        # Plot predictions from each epoch
        for epoch_idx, predictions in enumerate(epoch_predictions):
            if plot_idx >= len(axes.flat):
                break

            # Extract predictions from the selected timestep
            predictions_2d = predictions[:, timestep_idx, :, :, :]  # (N_sequences, H, W, classes)

            # Reconstruct predictions for this epoch
            channels_pred = predictions_2d.shape[-1] if predictions_2d.ndim == 4 else 1
            reconstructed_predictions = np.zeros((h, w, channels_pred), dtype=np.float32)

            for (i, j) in expected_positions:
                key = (date_idx, i, j)
                if key in date_position_to_patches:
                    patch_indices = date_position_to_patches[key]
                    patch_idx = patch_indices[0]
                    patch_pred = predictions_2d[patch_idx]
                    reconstructed_predictions[i:i + patch_size, j:j + patch_size, :] = patch_pred

            # Convert predictions to class indices
            if reconstructed_predictions.shape[-1] > 1:
                pred_img = np.argmax(reconstructed_predictions, axis=-1)
                pred_sum = np.sum(reconstructed_predictions, axis=-1)
            else:
                pred_img = reconstructed_predictions[:, :, 0]
                pred_sum = reconstructed_predictions[:, :, 0]
            pred_masked = np.ma.masked_where(pred_sum == 0, pred_img)

            # Plot prediction
            im = axes.flat[plot_idx].imshow(pred_masked, cmap='tab20', aspect='equal',
                                            vmin=0, vmax=13, origin='upper')

            # Get metrics for this epoch
            epoch_metrics = epoch_metrics[epoch_idx]
            axes.flat[plot_idx].set_title(f'Epoch {epoch_idx + 1}\nAcc: {epoch_metrics["accuracy"]:.3f} | '
                                          f'IoU: {epoch_metrics["mean_iou"]:.3f}', fontsize=10)
            axes.flat[plot_idx].axis('off')
            plt.colorbar(im, ax=axes.flat[plot_idx], shrink=0.8)
            plot_idx += 1

        # Hide unused subplots
        for i in range(plot_idx, len(axes.flat)):
            axes.flat[i].axis('off')

        plt.suptitle(f'ConvLSTM Image {img_idx + 1} - Prediction Evolution (t={timestep_idx})', fontsize=16, y=0.98)
        plt.tight_layout()
        plt.savefig(f'convlstm_epoch_evolution_image_{img_idx + 1}_date_{date_idx}_t{timestep_idx}.png',
                    dpi=300, bbox_inches='tight')
        plt.show()

        print(f"✅ Image {img_idx + 1} epoch evolution visualization complete")
        print("-" * 70)


def visualize_temporal_evolution(X_test, y_test, predictions, positions, dates,
                                          original_shape=(744, 1171),
                                          patch_size=64,
                                          stride=64,
                                          sequence_idx=0):
    """
    Visualize how a single spatial location evolves over time in ConvLSTM sequences.
    Shows the temporal dimension that ConvLSTM learns from.

    Args:
        X_test: Test sequence patches array (N_sequences, seq_length, H, W, C)
        y_test: Ground truth sequences (N_sequences, seq_length, H, W, classes)
        predictions: Model predictions (N_sequences, seq_length, H, W, classes)
        positions: Position data (N_sequences, seq_length, 2)
        dates: Date data (N_sequences, seq_length)
        sequence_idx: Which sequence to visualize
    """

    print(f"=== VISUALIZING TEMPORAL EVOLUTION FOR SEQUENCE {sequence_idx} ===")

    # Extract the specific sequence
    seq_thermal = X_test[sequence_idx]  # (seq_length, H, W, C)
    seq_labels = y_test[sequence_idx]  # (seq_length, H, W, classes)
    seq_preds = predictions[sequence_idx]  # (seq_length, H, W, classes)
    seq_dates = dates[sequence_idx]  # (seq_length,)
    seq_pos = positions[sequence_idx]  # (seq_length, 2)

    seq_length = seq_thermal.shape[0]

    print(f"Sequence length: {seq_length}")
    print(f"Position: {seq_pos[0]} (should be same for all timesteps)")
    print(f"Dates: {seq_dates}")

    # Create visualization
    fig, axes = plt.subplots(3, seq_length, figsize=(4 * seq_length, 12))
    if seq_length == 1:
        axes = axes.reshape(3, 1)

    for t in range(seq_length):
        # Thermal input
        thermal_img = seq_thermal[t, :, :, 0]
        im1 = axes[0, t].imshow(thermal_img, cmap='hot', aspect='equal', origin='upper')
        axes[0, t].set_title(f'Thermal t={t}\nDate {seq_dates[t]}', fontsize=10)
        axes[0, t].axis('off')
        plt.colorbar(im1, ax=axes[0, t], shrink=0.8)

        # Ground truth
        if seq_labels.shape[-1] > 1:
            label_img = np.argmax(seq_labels[t], axis=-1)
        else:
            label_img = seq_labels[t, :, :, 0]
        im2 = axes[1, t].imshow(label_img, cmap='tab20', aspect='equal',
                                vmin=0, vmax=13, origin='upper')
        axes[1, t].set_title(f'Ground Truth t={t}', fontsize=10)
        axes[1, t].axis('off')
        plt.colorbar(im2, ax=axes[1, t], shrink=0.8)

        # Predictions
        if seq_preds.shape[-1] > 1:
            pred_img = np.argmax(seq_preds[t], axis=-1)
        else:
            pred_img = seq_preds[t, :, :, 0]
        im3 = axes[2, t].imshow(pred_img, cmap='tab20', aspect='equal',
                                vmin=0, vmax=13, origin='upper')
        axes[2, t].set_title(f'Prediction t={t}', fontsize=10)
        axes[2, t].axis('off')
        plt.colorbar(im3, ax=axes[2, t], shrink=0.8)

    plt.suptitle(f'Temporal Evolution - Sequence {sequence_idx} at Position {seq_pos[0]}',
                 fontsize=16, y=0.98)
    plt.tight_layout()
    plt.savefig(f'convlstm_temporal_evolution_seq_{sequence_idx}.png',
                dpi=300, bbox_inches='tight')
    plt.show()


# -------------- WORKFLOW ----------------------------------------------------------------------------------------------
model, history, predictions, best_params, epoch_predictions, epoch_metrics = train_evaluate_model(
    X_train, y_train, w_train,
    X_val, y_val, w_val,
    X_test, y_test, w_test,
    use_optuna=False, n_trials=10)

metrics = compute_metrics(y_test, predictions)

print("\nModel Performance Metrics:")
print(f"Accuracy: {metrics['accuracy']:.4f}")
print(f"Precision: {metrics['precision']:.4f}")
print(f"Recall: {metrics['recall']:.4f}")
print(f"F1 Score: {metrics['f1_score']:.4f}")
print(f"Mean IoU: {metrics['mean_iou']:.4f}")
print(f"Mean Dice: {metrics['mean_dice']:.4f}")
print("\nPer-Class IoU:")
for cls, iou in enumerate(metrics['class_iou']):
    print(f"Class {cls}: {iou:.4f}")

print("=== INDIVIDUAL PARAMETER VALUES ===")
for param_name, param_value in best_params.items():
    print(f"{param_name}: {param_value}")

visualize_predictions(
    X_test, y_test, epoch_predictions=epoch_predictions,
    epoch_metrics=epoch_metrics, timestep_to_visualize=-1)

visualize_temporal_evolution(X_test, y_test, predictions,
                             pos_test, date_test,
                             sequence_idx=0)  # show first sequence

plot_training(history)

# ---------------- SAVING ----------------------------------------------------------------------------------------------
model.save(os.path.join('models', 'ConvLSTM.keras'))
model.save(os.path.join('models', 'ConvLSTM.h5'))
