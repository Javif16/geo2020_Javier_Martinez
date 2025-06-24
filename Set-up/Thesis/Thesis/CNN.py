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
import matplotlib.pyplot as plt
import cv2
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
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import KFold
from sklearn.metrics import precision_score, recall_score, accuracy_score, f1_score
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import optuna
from optuna.trial import TrialState
import pickle
from math import ceil

# -------------- LOADING DATA ------------------------------------------------------------------------------------------
# get datasets already prepared from thermal.py + check for their shapes
X_train = np.load("X_train.npy")
X_val = np.load("X_val.npy")
X_test = np.load("X_test.npy")
y_train = np.load("y_train.npy")
y_val = np.load("y_val.npy")
y_test = np.load("y_test.npy")
# data has to be 'float32' for tensorflow computations
w_train = np.load("weights_train.npy")
w_val = np.load("weights_val.npy")
w_test = np.load("weights_test.npy")

# ------------------------------------------------------------------------------
# X = input data (samples, 64, 64, 6), y = labels (samples, 64, 64, 14), w = weights (samples, 64, 64, 1)
X_train = X_train.astype('float32')
X_val = X_val.astype('float32')
X_test = X_test.astype('float32')
# int - they are output
y_train = y_train.astype('float32')
y_val = y_val.astype('float32')
y_test = y_test.astype('float32')
#
w_train = w_train.astype('float32')
w_val = w_val.astype('float32')
w_test = w_test.astype('float32')

# Only LST - (samples, 64, 64, 1)
X_train = X_train[:, :, :, 0:1]
X_val = X_val[:, :, :, 0:1]
X_test = X_test[:, :, :, 0:1]

# Only Emis - (samples, 64, 64, 5)
# X_train = X_train[:, :, :, 1:]
# X_val = X_val[:, :, :, 1:]
# X_test = X_test[:, :, :, 1:]


# -------------- CNN U-Net ---------------------------------------------------------------------------------------------
def unet_model(size_input=(64, 64, 1), filters=32, classes=14):
    '''
    U-Net architecture with encoder and decoder.
    :param size_input: Tuple, input size of the image (height, width, channels).
    :param filters: Nº of filters in the first convolutional layer.
    :param classes: Nº of output classes.
    :return: Compiled U-Net model.
    '''
    # Encoder
    def Encoder(inputs, filters, dropout, l2factor=0.1, max_pooling=True):
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

    def Decoder(previous_layer, skip, filters):
        '''
        Uses transpose convolutions to make images bigger (upscale) back to original size and merges with skip layer from Encoder.
        2 convolutional layers with 'same' padding increases depth of architecture for better predictions.
        :return:
        Decoded layer output.
        '''
        # increasing size
        upscale = Conv2DTranspose(filters, (3, 3), strides=(2, 2), padding='same')(previous_layer)  # strides = how much it upsamples
        # combine skip connection to prevent loss of information
        combination = concatenate([upscale, skip], axis=3)

        convolution = Conv2D(filters, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(
            combination)
        convolution = BatchNormalization()(convolution)
        convolution = tf.keras.layers.Activation("relu")(convolution)

        convo = Conv2D(filters, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(convolution)
        convo = BatchNormalization()(convo)
        convo = tf.keras.layers.Activation("relu")(convo)

        return convo

    # input size of the image
    inputs = tf.keras.layers.Input(size_input)

    # encoder with multiple convolutional layers with different maxpooling, dropout and filters
    # filters increase the deeper into the network it reaches to increase channel size
    conv1 = Encoder(inputs, filters=32, dropout=0, l2factor=0.1, max_pooling=True)
    conv2 = Encoder(conv1[0], filters=64, dropout=0, l2factor=0.1, max_pooling=True)
    conv3 = Encoder(conv2[0], filters=128, dropout=0, l2factor=0.1, max_pooling=True)
    conv4 = Encoder(conv3[0], filters=256, dropout=0.3, l2factor=0.1, max_pooling=True)

    # bottleneck (last layer before upscaling)
    bottleneck = Encoder(conv4[0], filters=512, dropout=0.3, l2factor=0.1, max_pooling=False)
    # maxpooling in last convolution as upscaling starts here

    # decoder with reducing filters with skip connections from encoder given as input
    # second output of encoder block is skip connection, so conv1[1] used
    upsc1 = Decoder(bottleneck[0], conv4[1], filters=256)
    upsc2 = Decoder(upsc1, conv3[1], filters=128)
    upsc3 = Decoder(upsc2, conv2[1], filters=64)
    upsc4 = Decoder(upsc3, conv1[1], filters=32)

    # final convolutional layer to get image to proper size, so nº of channels = nº output classes
    conv10 = Conv2D(filters, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal')(upsc4)
    # no need for Flatten() as it would output a single classification label instead of a full segmentation map

    conv11 = Conv2D(classes, (1, 1), activation='softmax', padding='same')(conv10)

    # model defining
    model = tf.keras.Model(inputs=inputs, outputs=conv11)
    model.summary()

    return model


# -------------- TRAINING & EVALUATION ---------------------------------------------------------------------------------
def loss_function(w_train, weight_strength=1.5):
    def loss(y_true, y_pred):
        batch_size = tf.shape(y_true)[0]
        # Handle batch size properly - use modulo to cycle through weights if needed
        batch_indices = tf.range(batch_size) % tf.shape(w_train)[0]
        w_batch = tf.gather(w_train, batch_indices)

        # Reshape tensors for proper calculation
        num_classes = tf.shape(y_true)[3]
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


'''        # Calculate categorical crossentropy manually (better than built-in functions)
        cce = -tf.reduce_sum(y_true_flat * tf.math.log(y_pred_flat), axis=-1)

        # Apply weights
        weighted_cce = cce * w_flat

        # Return mean
        return tf.reduce_mean(weighted_cce)
'''


def train_evaluate_model_with_epoch_tracking(X_train, y_train, w_train, X_val, y_val, w_val, X_test, y_test, w_test,
                                             use_optuna=True, n_trials=30):
    """
    Train and evaluate model with optional Optuna hyperparameter optimization
    Enhanced to track predictions at each epoch for visualization
    """

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
                model = unet_model(
                    size_input=(64, 64, 1),
                    filters=32,
                    classes=14)

                # Compile model
                optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate, clipnorm=1.0)
                loss = loss_function(w_train, weight_strength=weight_strength)

                model.compile(
                    optimizer=optimizer,
                    loss=loss,
                    metrics=['accuracy', 'precision', 'recall']
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
                    sample_weight=w_train,
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

        # Run optimization
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
        # Use default parameters
        best_params = {
            'learning_rate': 0.0001,
            'batch_size': 32,
            'weight_strength': 1.0
        }
        print("=== USING DEFAULT PARAMETERS (NO OPTUNA) ===")

    print("\n=== TRAINING FINAL MODEL WITH EPOCH TRACKING ===")

    # Train final model with best/default parameters
    tf.keras.backend.clear_session()

    # Create final model
    model = unet_model(
        size_input=(64, 64, 1),
        filters=32,
        classes=14)

    print("Model output shape:", model.output_shape)

    # Compile with optimized parameters
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
            self.X_test = X_test
            self.y_test = y_test
            self.epoch_predictions = []
            self.epoch_metrics = []

        def on_epoch_end(self, epoch, logs=None):
            # Make predictions on test set
            predictions = self.model.predict(self.X_test, verbose=0)
            self.epoch_predictions.append(predictions.copy())

            # Calculate metrics for this epoch
            metrics = compute_metrics(self.y_test, predictions)
            self.epoch_metrics.append(metrics)

            print(f"Epoch {epoch + 1} - Test Accuracy: {metrics['accuracy']:.4f}, "
                  f"Test Mean IoU: {metrics['mean_iou']:.4f}")

    # Initialize the callback
    epoch_callback = EpochPredictionCallback(X_test, y_test)

    # Other callbacks
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

    # Training with epoch tracking
    print("Training final CNN model...")
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        sample_weight=w_train,
        epochs=6,
        batch_size=best_params['batch_size'],
        callbacks=[lr_callback, early_stopping, epoch_callback],
        verbose=1
    )

    # Final evaluation
    print("\n=== FINAL EVALUATION ===")
    test_results = model.evaluate(X_test, y_test, sample_weight=w_test, verbose=0)
    test_loss, test_accuracy, test_precision, test_recall, test_categorical = test_results

    print(f"Test loss: {test_loss}")
    print(f"Test Accuracy: {test_accuracy}")
    print(f"Test Precision: {test_precision}")
    print(f"Test Recall: {test_recall}")
    print(f"Test Categorical: {test_categorical}")

    # Final predictions
    final_predictions = model.predict(X_test)

    return model, history, final_predictions, best_params, epoch_callback.epoch_predictions, epoch_callback.epoch_metrics


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

    # Convert predictions to class indices if they're probabilities
    if y_pred.ndim == 4 and y_pred.shape[-1] > 1:  # Multi-class probabilities
        y_pred_classes = np.argmax(y_pred, axis=-1)
    else:
        y_pred_classes = y_pred

    # Convert true labels to class indices if they're one-hot encoded
    if y_true.ndim == 4 and y_true.shape[-1] > 1:  # One-hot encoded
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
def k_fold_cross_validation(X, y, w, k=10):
    kf = KFold(n_splits=k, shuffle=True, random_state=42)
    fold_results = {'accuracy': [], 'precision': [], 'recall': [], 'f1_score': [], 'iou': [], 'dice': []}

    for fold, (train_idx, val_idx) in enumerate(kf.split(X), 1):
        print(f"\nFold {fold}/{k}")

        # Split data for this fold
        X_train_fold, X_val_fold = X[train_idx], X[val_idx]
        y_train_fold, y_val_fold = y[train_idx], y[val_idx]
        w_train_fold, w_val_fold = w[train_idx], w[val_idx]

        # Train the model on this fold's training data
        model = unet_model()
        optimizer = tf.keras.optimizers.Adam(learning_rate=5e-5, clipnorm=1.0)
        loss = loss_function(w_train_fold)

        model.compile(
            optimizer=optimizer,
            loss=loss,
            metrics=['accuracy', tf.keras.metrics.CategoricalAccuracy(), tf.keras.metrics.Precision(), tf.keras.metrics.Recall()]
        )

        # Callbacks
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
        print("X_train shape:", X_train.shape)
        print("y_train shape:", y_train.shape)
        print("Model output shape:", model.output_shape)
        # Training
        history = model.fit(
            X_train_fold, y_train_fold,
            validation_data=(X_val_fold, y_val_fold),
            epochs=30,
            batch_size=32,
            callbacks=[lr_callback, early_stopping],
            sample_weight=w_train_fold,
            verbose=1
        )

        # Evaluate on validation fold
        y_pred = model.predict(X_val_fold)

        # Compute metrics
        precision, recall, accuracy, f1_score, iou, dice = compute_metrics(y_val_fold, y_pred)
        
        # Store results
        fold_results['accuracy'].append(accuracy)
        fold_results['precision'].append(precision)
        fold_results['recall'].append(recall)
        fold_results['f1_score'].append(f1_score)
        fold_results['iou'].append(iou)
        fold_results['dice'].append(dice)

        plot_training(history)

    # Compute and print mean results
    print("\nCross-Validation Results:")
    for metric, values in fold_results.items():
        print(f"{metric.capitalize()}: {np.mean(values):.4f} ± {np.std(values):.4f}")

    return fold_results
'''


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


def visualize_predictions(X_test, y_test, epoch_predictions_list, epoch_metrics_list,
                                      original_shape=(744, 1171),
                                      patch_size=64,
                                      stride=64,
                                      num_images=1):
    """
    Enhanced visualization that shows predictions from each epoch for comparison.
    Shows how the model predictions evolve over training epochs.

    Args:
        X_test: Test image patches array (N, H, W, C)
        y_test: Ground truth label patches (N, H, W, classes) - one-hot encoded
        epoch_predictions_list: List of predictions from each epoch
        epoch_metrics_list: List of metrics from each epoch
        original_shape: Shape of the original full image (H, W)
        patch_size: Size of each square patch
        stride: Step size between patches
        num_images: Number of full images to reconstruct and display
    """
    import numpy as np
    import matplotlib.pyplot as plt
    from collections import defaultdict

    print("=== VISUALIZING EPOCH-BY-EPOCH PREDICTIONS ===")

    # Load the test positions and date indices
    pos_test = np.load("positions_test.npy")
    date_test = np.load("date_indices_test.npy")

    # Calculate expected grid dimensions
    h, w = original_shape
    patches_per_row = (w - patch_size) // stride + 1
    patches_per_col = (h - patch_size) // stride + 1
    patches_per_image = patches_per_row * patches_per_col

    print(f"Original image shape: {original_shape}")
    print(f"Patches per row: {patches_per_row}, Patches per column: {patches_per_col}")
    print(f"Total patches per image expected: {patches_per_image}")
    print(f"Available test patches: {len(X_test)}")
    print(f"Number of epochs tracked: {len(epoch_predictions_list)}")

    # Group patches by date AND position
    date_position_to_patches = defaultdict(list)
    for idx, ((i, j), date_idx) in enumerate(zip(pos_test, date_test)):
        date_position_to_patches[(date_idx, i, j)].append(idx)

    # Group by date
    patches_per_date = defaultdict(list)
    for idx, date_idx in enumerate(date_test):
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
        values = [epoch_metrics[metric] for epoch_metrics in epoch_metrics_list]
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
    plt.savefig('epoch_metrics_evolution.png', dpi=300, bbox_inches='tight')
    plt.show()

    # Now visualize reconstructed images for each epoch
    for img_idx in range(num_images):
        date_idx = available_dates[img_idx]
        print(f"\n--- Processing Image {img_idx + 1} from date index {date_idx} ---")

        # Reconstruct ground truth and thermal image (same for all epochs)
        channels_X = X_test.shape[-1]
        channels_y = y_test.shape[-1] if y_test.ndim == 4 else 1

        reconstructed_thermal = np.zeros((h, w, channels_X), dtype=np.float32)
        reconstructed_labels = np.zeros((h, w, channels_y), dtype=np.float32)

        filled_positions = []

        # Place patches only from this specific date (for ground truth and thermal)
        for (i, j) in expected_positions:
            key = (date_idx, i, j)
            if key in date_position_to_patches:
                patch_indices = date_position_to_patches[key]
                patch_idx = patch_indices[0]

                patch_thermal = X_test[patch_idx]
                patch_labels = y_test[patch_idx]

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
        num_epochs = len(epoch_predictions_list)
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
            axes.flat[plot_idx].set_title(f'Input Thermal\n(Date {date_idx})', fontsize=10)
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
        for epoch_idx, predictions in enumerate(epoch_predictions_list):
            if plot_idx >= len(axes.flat):
                break

            # Reconstruct predictions for this epoch
            channels_pred = predictions.shape[-1] if predictions.ndim == 4 else 1
            reconstructed_predictions = np.zeros((h, w, channels_pred), dtype=np.float32)

            for (i, j) in expected_positions:
                key = (date_idx, i, j)
                if key in date_position_to_patches:
                    patch_indices = date_position_to_patches[key]
                    patch_idx = patch_indices[0]
                    patch_pred = predictions[patch_idx]
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
            epoch_metrics = epoch_metrics_list[epoch_idx]
            axes.flat[plot_idx].set_title(f'Epoch {epoch_idx + 1}\nAcc: {epoch_metrics["accuracy"]:.3f} | '
                                          f'IoU: {epoch_metrics["mean_iou"]:.3f}', fontsize=10)
            axes.flat[plot_idx].axis('off')
            plt.colorbar(im, ax=axes.flat[plot_idx], shrink=0.8)
            plot_idx += 1

        # Hide unused subplots
        for i in range(plot_idx, len(axes.flat)):
            axes.flat[i].axis('off')

        plt.suptitle(f'Image {img_idx + 1} - Prediction Evolution Across Epochs', fontsize=16, y=0.98)
        plt.tight_layout()
        plt.savefig(f'epoch_evolution_image_{img_idx + 1}_date_{date_idx}.png',
                    dpi=300, bbox_inches='tight')
        plt.show()

        print(f"✅ Image {img_idx + 1} epoch evolution visualization complete")
        print("-" * 70)


# -------------- WORKFLOW ----------------------------------------------------------------------------------------------
'''fold_results = k_fold_cross_validation(
    X_train,
    y_train,
    w_train
)'''

model, history, final_predictions, best_params, epoch_predictions, epoch_metrics = train_evaluate_model_with_epoch_tracking(
    X_train, y_train, w_train,
    X_val, y_val, w_val,
    X_test, y_test, w_test,
    use_optuna=False, n_trials=10
)

# Calculate detailed metrics
final_metrics = compute_metrics(y_test, final_predictions)

print("\n=== FINAL MODEL PERFORMANCE METRICS ===")
print(f"Accuracy: {final_metrics['accuracy']:.4f}")
print(f"Precision: {final_metrics['precision']:.4f}")
print(f"Recall: {final_metrics['recall']:.4f}")
print(f"F1 Score: {final_metrics['f1_score']:.4f}")
print(f"Mean IoU: {final_metrics['mean_iou']:.4f}")
print(f"Mean Dice: {final_metrics['mean_dice']:.4f}")

print("=== INDIVIDUAL PARAMETER VALUES ===")
for param_name, param_value in best_params.items():
    print(f"{param_name}: {param_value}")

visualize_predictions(X_test, y_test, epoch_predictions, epoch_metrics,
                                original_shape=(744, 1171),
                                patch_size=64,
                                num_images=2)

# Plot training history
plot_training(history)

# Save model
model.save(os.path.join('models', 'CNN_categorical.keras'))  # Native Keras format
model.save(os.path.join('models', 'CNN_categorical.h5'))     # HDF5 format for compatibility


'''
def predict_geology_map(model, input_data):
    """
    Generate geology classification maps from the model.
    
    Args:
        model: Trained U-Net model
        input_data: Input thermal data
        
    Returns:
        predictions: Classification maps with class labels (0-13)
    """
    logits = model.predict(input_data)
    
    # Convert logits to probabilities
    probabilities = tf.nn.softmax(logits).numpy()
    
    # Get class indices
    class_indices = np.argmax(probabilities, axis=-1)
    
    # Convert to one-hot encoding (categorical format)
    one_hot_predictions = tf.one_hot(class_indices, depth=logits.shape[-1]).numpy()
    
    return one_hot_predictions, class_indices

# Example of generating prediction maps
geology_maps = predict_geology_map(model, X_test)
print(f"Generated geology maps shape: {geology_maps.shape}")
'''