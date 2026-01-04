import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.preprocessing import StandardScaler

# Config
MODEL_PATH = './ckpts/best_model.keras'
TRAIN_DATA_PATH = './dataset/mitbih_train.csv'
OUTPUT_DIR = './exported_models/tflite'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 1. Define Custom LayerNorm
class CustomLayerNorm(layers.Layer):
    def __init__(self, epsilon=1e-3, **kwargs):
        super(CustomLayerNorm, self).__init__(**kwargs)
        self.epsilon = epsilon

    def build(self, input_shape):
        self.gamma = self.add_weight(name='gamma', shape=input_shape[-1:],
                                     initializer='ones', trainable=True)
        self.beta = self.add_weight(name='beta', shape=input_shape[-1:],
                                    initializer='zeros', trainable=True)
        super(CustomLayerNorm, self).build(input_shape)

    def call(self, x):
        mean = tf.reduce_mean(x, axis=-1, keepdims=True)
        variance = tf.reduce_mean(tf.square(x - mean), axis=-1, keepdims=True)
        std = tf.sqrt(variance + self.epsilon)
        outputs = (x - mean) / std
        return outputs * self.gamma + self.beta

    def get_config(self):
        config = super(CustomLayerNorm, self).get_config()
        config.update({'epsilon': self.epsilon})
        return config

# 2. Redefine Model Architecture
def transformer_encoder(inputs, head_size, num_heads, ff_dim, dropout=0):
    # Normalization and Attention
    # x = layers.LayerNormalization(epsilon=1e-6)(inputs)
    x = CustomLayerNorm(epsilon=1e-3)(inputs)
    x = layers.MultiHeadAttention(
        key_dim=head_size, num_heads=num_heads, dropout=dropout
    )(x, x)
    x = layers.Dropout(dropout)(x)
    res = x + inputs

    # Feed Forward Part
    # x = layers.LayerNormalization(epsilon=1e-6)(res)
    x = CustomLayerNorm(epsilon=1e-3)(res)
    x = layers.Conv1D(filters=ff_dim, kernel_size=1, activation="relu")(x)
    x = layers.Dropout(dropout)(x)
    x = layers.Conv1D(filters=inputs.shape[-1], kernel_size=1)(x)
    return x + res

def build_model(
    input_shape,
    head_size,
    num_heads,
    ff_dim,
    num_transformer_blocks,
    mlp_units,
    dropout=0,
    mlp_dropout=0,
    n_classes=5
):
    inputs = keras.Input(shape=input_shape)
    x = inputs
    for _ in range(num_transformer_blocks):
        x = transformer_encoder(x, head_size, num_heads, ff_dim, dropout)

    x = layers.GlobalAveragePooling1D(data_format="channels_first")(x)
    for dim in mlp_units:
        x = layers.Dense(dim, activation="relu")(x)
        x = layers.Dropout(mlp_dropout)(x)
    outputs = layers.Dense(n_classes, activation="softmax")(x)
    return keras.Model(inputs, outputs)

# 3. Load Data (for representative dataset)
def readucr(filename):
    data = pd.read_csv(filename, header=None)
    y = data.iloc[:, -1].astype(int).to_numpy()
    x = data.iloc[:, :-1]
    scaler = StandardScaler()
    standardized_data = scaler.fit_transform(x)
    x = pd.DataFrame(standardized_data, columns=x.columns).to_numpy()
    return x, y.astype(int)

print("Loading data...")
x_train, y_train = readucr(TRAIN_DATA_PATH)
x_train = x_train.reshape((x_train.shape[0], x_train.shape[1], 1))

def representative_dataset_gen():
    indices = np.random.choice(len(x_train), 100, replace=False)
    for i in indices:
        yield [x_train[i:i+1].astype(np.float32)]

# 4. Build New Model and Transfer Weights
print("Building new model with CustomLayerNorm...")
input_shape = x_train.shape[1:]
new_model = build_model(
    input_shape,
    head_size=16,
    num_heads=8,
    ff_dim=4,
    num_transformer_blocks=4,
    mlp_units=[128,64],
    mlp_dropout=0.4,
    dropout=0.25,
    n_classes=5
)

print("Loading original model...")
original_model = tf.keras.models.load_model(MODEL_PATH)

print("Transferring weights...")
# We need to be careful. The structure is identical except for LayerNorm class.
# We can iterate through layers and set weights.
# Since we built them identically, the order of layers should be roughly the same, 
# but Keras functional API graph might order them slightly differently if not careful.
# Safer to match by name if possible, but names are auto-generated.
# Let's try to match by index/structure.

# Flatten layers list
orig_layers = original_model.layers
new_layers = new_model.layers

print(f"Original layers: {len(orig_layers)}")
print(f"New layers: {len(new_layers)}")

if len(orig_layers) != len(new_layers):
    print("Warning: Layer count mismatch!")

# Map weights
for i, (orig_layer, new_layer) in enumerate(zip(orig_layers, new_layers)):
    # print(f"Layer {i}: {orig_layer.name} ({orig_layer.__class__.__name__}) -> {new_layer.name} ({new_layer.__class__.__name__})")
    
    if isinstance(orig_layer, layers.LayerNormalization) and isinstance(new_layer, CustomLayerNorm):
        # Transfer LayerNorm weights
        # Standard LN weights: [gamma, beta]
        # Custom LN weights: [gamma, beta] (defined in build)
        weights = orig_layer.get_weights()
        if weights:
            new_layer.set_weights(weights)
            # print(f"  Transferred LayerNorm weights for layer {i}")
    elif isinstance(orig_layer, new_layer.__class__):
        # Other layers
        weights = orig_layer.get_weights()
        if weights:
            new_layer.set_weights(weights)
            # print(f"  Transferred weights for layer {i}")
    else:
        # Mismatch?
        # MultiHeadAttention might be tricky if it has internal layers.
        # But get_weights() returns a flat list of weights for the whole layer.
        # If the class matches, it should be fine.
        if isinstance(orig_layer, layers.MultiHeadAttention) and isinstance(new_layer, layers.MultiHeadAttention):
             new_layer.set_weights(orig_layer.get_weights())
        elif isinstance(orig_layer, layers.InputLayer):
            pass
        else:
            print(f"  Mismatch at layer {i}: {orig_layer.__class__.__name__} vs {new_layer.__class__.__name__}")
            # Try to set anyway if shapes match
            try:
                new_layer.set_weights(orig_layer.get_weights())
            except Exception as e:
                print(f"  Could not set weights: {e}")

# 4b. Verify Float Inference
print("\nVerifying float model inference...")
try:
    test_input = x_train[0:1]
    pred = new_model.predict(test_input)
    print(f"Prediction shape: {pred.shape}")
    print(f"Prediction: {pred}")
    if np.isnan(pred).any():
        print("WARNING: Prediction contains NaNs!")
except Exception as e:
    print(f"Float inference failed: {e}")

# 5. Quantize
print("\nQuantizing to INT8...")
converter = tf.lite.TFLiteConverter.from_keras_model(new_model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_dataset_gen
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type = tf.int8
converter.inference_output_type = tf.int8

tflite_model = converter.convert()
output_path = os.path.join(OUTPUT_DIR, 'ecgformer_custom_ln_int8.tflite')
with open(output_path, 'wb') as f:
    f.write(tflite_model)

print(f"Saved to {output_path}")

# 6. Verify Load
print("Verifying load...")
try:
    interpreter = tf.lite.Interpreter(model_path=output_path)
    interpreter.allocate_tensors()
    print("SUCCESS: Model loaded successfully!")
except Exception as e:
    print(f"FAILURE: {e}")

# 7. Evaluate Accuracy (Quick Check)
print("\nEvaluating INT8 model accuracy on 1000 samples...")
try:
    interpreter = tf.lite.Interpreter(model_path=output_path)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    input_scale, input_zero_point = input_details[0]['quantization']
    output_scale, output_zero_point = output_details[0]['quantization']
    
    correct = 0
    total = 1000
    
    # Load test data
    x_test, y_test = readucr('./dataset/mitbih_test.csv')
    x_test = x_test.reshape((x_test.shape[0], x_test.shape[1], 1))
    
    for i in range(total):
        input_data = x_test[i:i+1].astype(np.float32)
        # Quantize input
        input_data_int8 = (input_data / input_scale + input_zero_point).astype(np.int8)
        
        interpreter.set_tensor(input_details[0]['index'], input_data_int8)
        interpreter.invoke()
        output_data_int8 = interpreter.get_tensor(output_details[0]['index'])
        
        # Dequantize output
        output_data = (output_data_int8.astype(np.float32) - output_zero_point) * output_scale
        pred = np.argmax(output_data)
        
        if pred == y_test[i]:
            correct += 1
            
    print(f"Accuracy: {correct/total:.4f}")

except Exception as e:
    print(f"Evaluation failed: {e}")
