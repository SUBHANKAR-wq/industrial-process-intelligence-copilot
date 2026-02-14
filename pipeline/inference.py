import os
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras import layers
import numpy as np
import pandas as pd

# =============================
# CLOUD SAFE PATH
# =============================

BASE_DIR = os.path.dirname(os.path.dirname(__file__))

MODEL_PATH = os.path.join(
    BASE_DIR,
    "models",
    "autoencoder_new.h5"
)

# =============================
# CONFIG
# =============================

SENSOR_COLS = [
    "temperature",
    "pressure",
    "flow_rate",
    "vibration"
]

# =============================
# GLOBAL MODEL (LOAD ONCE)
# =============================

autoencoder = None


def get_model():
    """
    Lazy loading with version compatibility:
    Model loads only once when first request comes.
    Handles Keras version mismatches.
    """
    global autoencoder

    if autoencoder is None:
        print("Loading autoencoder model...")
        
        # Try loading with different strategies
        autoencoder = load_model_with_compatibility(MODEL_PATH)
        
        print("Model loaded successfully!")

    return autoencoder


def load_model_with_compatibility(model_path):
    """
    Load model with multiple fallback strategies for version compatibility
    """
    
    # Strategy 1: Try loading with compile=False (basic)
    try:
        print("Attempting to load model normally...")
        return load_model(model_path, compile=False)
    
    except TypeError as e:
        if 'quantization_config' in str(e):
            print("Version mismatch detected (quantization_config issue)")
            
            # Strategy 2: Try with custom objects that ignore unknown args
            try:
                print("Attempting to load with custom objects...")
                
                # Create a custom Dense class that handles unknown arguments
                class CompatibleDense(layers.Dense):
                    @classmethod
                    def from_config(cls, config):
                        # Remove problematic keys for older versions
                        problematic_keys = ['quantization_config', 'quantization_mode']
                        for key in problematic_keys:
                            if key in config:
                                print(f"Removing problematic key: {key}")
                                del config[key]
                        return super().from_config(config)
                
                custom_objects = {
                    'Dense': CompatibleDense,
                    'dense': CompatibleDense,
                }
                
                return load_model(
                    model_path,
                    custom_objects=custom_objects,
                    compile=False
                )
            except Exception as e2:
                print(f"Custom objects loading failed: {e2}")
                
                # Strategy 3: Try with safe mode using tf.keras.models.load_model
                try:
                    print("Attempting to load with safe mode...")
                    
                    # Register safe mode for loading
                    from tensorflow.keras.saving import register_keras_serializable
                    
                    # Define a safe loading function
                    import h5py
                    
                    # Load weights and rebuild model architecture from config
                    with h5py.File(model_path, 'r') as f:
                        # Try to get model config
                        if 'model_config' in f.attrs:
                            import json
                            config = json.loads(f.attrs['model_config'])
                            
                            # Clean config of problematic keys
                            def clean_config(config_item):
                                if isinstance(config_item, dict):
                                    # Remove problematic keys
                                    config_item.pop('quantization_config', None)
                                    config_item.pop('quantization_mode', None)
                                    
                                    # Recursively clean nested configs
                                    for key, value in config_item.items():
                                        if isinstance(value, (dict, list)):
                                            clean_config(value)
                                elif isinstance(config_item, list):
                                    for item in config_item:
                                        clean_config(item)
                                return config_item
                            
                            cleaned_config = clean_config(config)
                            
                            # Rebuild model from cleaned config
                            from tensorflow.keras.models import model_from_json
                            model = model_from_json(
                                json.dumps(cleaned_config),
                                custom_objects={'Dense': CompatibleDense}
                            )
                            
                            # Load weights
                            model.load_weights(model_path)
                            
                            return model
                            
                except Exception as e3:
                    print(f"Safe mode loading failed: {e3}")
                    
                    # Strategy 4: Last resort - try with legacy behavior
                    try:
                        print("Attempting to load with legacy Keras...")
                        
                        # Set legacy Keras environment variable
                        os.environ['TF_USE_LEGACY_KERAS'] = '1'
                        
                        # Reload tensorflow.keras to apply changes
                        import importlib
                        importlib.reload(tf.keras)
                        
                        from tensorflow.keras.models import load_model as legacy_load
                        return legacy_load(model_path, compile=False)
                        
                    except Exception as e4:
                        print(f"All loading strategies failed: {e4}")
                        raise RuntimeError(f"Could not load model: {e}")
        
        else:
            # Different type of error, re-raise
            raise e


# =============================
# INFERENCE
# =============================

def run_inference(df, X):
    """
    Run inference on input data
    """
    try:
        # Load model safely
        model = get_model()

        # Reconstruction
        X_recon = model.predict(X, verbose=0)

        # Reconstruction error
        sensor_errors = (X - X_recon) ** 2

        # Keep original dataframe
        output_df = df.copy()

        # Per sensor error
        for i, sensor in enumerate(SENSOR_COLS):
            if i < sensor_errors.shape[1]:  # Safety check
                output_df[f"{sensor}_error"] = sensor_errors[:, i]

        # Total error
        total_error = sensor_errors.sum(axis=1, keepdims=True)

        # Contribution percentage (avoid division by zero)
        total_error_safe = total_error[:, 0] + 1e-12
        
        for i, sensor in enumerate(SENSOR_COLS):
            if i < sensor_errors.shape[1]:
                output_df[f"{sensor}_contribution"] = (
                    sensor_errors[:, i] / total_error_safe
                )

        # Final reconstruction error
        output_df["reconstruction_error"] = (
            total_error[:, 0] / X.shape[1]
        )

        return output_df
        
    except Exception as e:
        print(f"Error during inference: {e}")
        # Return original dataframe with error columns as NaN
        output_df = df.copy()
        for sensor in SENSOR_COLS:
            output_df[f"{sensor}_error"] = np.nan
            output_df[f"{sensor}_contribution"] = np.nan
        output_df["reconstruction_error"] = np.nan
        return output_df


# Optional: Add a test function
def test_model_loading():
    """Test if model loads successfully"""
    try:
        model = get_model()
        print(f"Model loaded successfully. Type: {type(model)}")
        
        # Test with random data
        test_input = np.random.randn(1, len(SENSOR_COLS))
        output = model.predict(test_input, verbose=0)
        print(f"Test inference successful. Output shape: {output.shape}")
        
        return True
    except Exception as e:
        print(f"Model test failed: {e}")
        return False