import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.utils import plot_model
import sys
import os

# Load the model WITHOUT importing from CNN.py to avoid re-running training
print("Loading model without compilation to avoid re-running training...")
try:
    model = tf.keras.models.load_model(
        'models/CNN_categorical.keras',
        compile=False  # This skips the custom loss function but loads everything else
    )
    print("Model loaded successfully!")
    print("Note: Custom loss function not loaded, but all architecture and weights are preserved.")

except Exception as e:
    print(f"Failed to load model: {e}")
    exit(1)

# 1. Basic model information
print("=== MODEL SUMMARY ===")
model.summary()

print("\n=== MODEL CONFIGURATION ===")
print(f"Input shape: {model.input_shape}")
print(f"Output shape: {model.output_shape}")
print(f"Total parameters: {model.count_params():,}")
try:
    print(f"Trainable parameters: {sum([tf.keras.backend.count_params(w) for w in model.trainable_weights]):,}")
except:
    print("Could not calculate trainable parameters")

# 2. Layer details
print("\n=== LAYER DETAILS ===")
for i, layer in enumerate(model.layers):
    print(f"Layer {i}: {layer.name} ({layer.__class__.__name__})")

    # Check for common layer attributes
    if hasattr(layer, 'filters'):
        print(f"  - Filters: {layer.filters}")
    if hasattr(layer, 'kernel_size'):
        print(f"  - Kernel size: {layer.kernel_size}")
    if hasattr(layer, 'strides'):
        print(f"  - Strides: {layer.strides}")
    if hasattr(layer, 'padding'):
        print(f"  - Padding: {layer.padding}")
    if hasattr(layer, 'activation'):
        activation_name = layer.activation.__name__ if hasattr(layer.activation, '__name__') else str(layer.activation)
        print(f"  - Activation: {activation_name}")
    if hasattr(layer, 'units'):
        print(f"  - Units: {layer.units}")
    if hasattr(layer, 'pool_size'):
        print(f"  - Pool size: {layer.pool_size}")

    # Handle output shape more carefully
    try:
        if hasattr(layer, 'output_shape'):
            print(f"  - Output shape: {layer.output_shape}")
        elif hasattr(layer, 'output'):
            print(f"  - Output shape: {layer.output.shape}")
        else:
            print(f"  - Output shape: Not available")
    except:
        print(f"  - Output shape: Could not determine")

    # Parameter count
    try:
        param_count = layer.count_params()
        print(f"  - Parameters: {param_count:,}")
    except:
        print(f"  - Parameters: Could not determine")

    print()

# 3. Model configuration (with error handling)
print("=== MODEL COMPILATION ===")
try:
    print(f"Optimizer: {model.optimizer.__class__.__name__}")
    if hasattr(model.optimizer, 'learning_rate'):
        print(f"Learning rate: {model.optimizer.learning_rate.numpy()}")
    print(f"Loss function: {model.loss}")
    print(f"Metrics: {model.metrics_names}")
except Exception as e:
    print(f"Could not retrieve compilation details: {e}")
    print("This is normal if model was loaded with compile=False")

# 4. Examine weights (first few layers)
print("\n=== WEIGHT SHAPES (First 5 layers) ===")
for i, layer in enumerate(model.layers[:5]):
    if layer.weights:
        print(f"Layer {i} ({layer.name}):")
        for j, weight in enumerate(layer.weights):
            print(f"  Weight {j}: {weight.shape}")

# --------------------------------------------------------------------------------------------------------------------------

# 1. Create architecture diagram
try:
    plot_model(
        model,
        to_file='model_architecture.png',
        show_shapes=True,
        show_layer_names=True,
        rankdir='TB',  # Top to bottom
        expand_nested=True,
        dpi=150
    )
    print("Architecture diagram saved as 'model_architecture.png'")
except Exception as e:
    print(f"Could not create architecture diagram: {e}")

# 2. Plot model as a graph (if you have graphviz installed)
try:
    plot_model(
        model,
        to_file='model_detailed.png',
        show_shapes=True,
        show_layer_names=True,
        show_layer_activations=True,
        rankdir='LR',  # Left to right
        expand_nested=True
    )
    print("Detailed diagram saved as 'model_detailed.png'")
except Exception as e:
    print(f"Detailed plot failed: {e}")
    print("Install graphviz: pip install graphviz")


# 3. Simple layer visualization
def plot_layer_summary(model):
    layer_names = []
    layer_types = []
    param_counts = []

    for layer in model.layers:
        layer_names.append(layer.name[:20])  # Truncate long names
        layer_types.append(layer.__class__.__name__)
        param_counts.append(layer.count_params())

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 8))

    # Layer types
    unique_types = list(set(layer_types))
    colors = plt.cm.Set3(np.linspace(0, 1, len(unique_types)))
    type_colors = [colors[unique_types.index(t)] for t in layer_types]

    ax1.barh(range(len(layer_names)), [1] * len(layer_names), color=type_colors)
    ax1.set_yticks(range(len(layer_names)))
    ax1.set_yticklabels(layer_names, fontsize=8)
    ax1.set_xlabel('Layers')
    ax1.set_title('Model Layers by Type')

    # Add legend
    handles = [plt.Rectangle((0, 0), 1, 1, color=colors[i]) for i in range(len(unique_types))]
    ax1.legend(handles, unique_types, loc='upper right', fontsize=8)

    # Parameter counts
    ax2.barh(range(len(layer_names)), param_counts)
    ax2.set_yticks(range(len(layer_names)))
    ax2.set_yticklabels(layer_names, fontsize=8)
    ax2.set_xlabel('Number of Parameters')
    ax2.set_title('Parameters per Layer')
    ax2.set_xscale('log')

    plt.tight_layout()
    plt.savefig('layer_analysis.png', dpi=150, bbox_inches='tight')
    plt.show()


try:
    plot_layer_summary(model)
except Exception as e:
    print(f"Could not create layer summary plot: {e}")

print("\n=== INSPECTION COMPLETE ===")
print("Model inspection completed successfully!")
print("Generated files:")
print("- model_architecture.png (if successful)")
print("- model_detailed.png (if graphviz available)")
print("- layer_analysis.png (if successful)")