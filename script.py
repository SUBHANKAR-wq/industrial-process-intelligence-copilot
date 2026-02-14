import tensorflow as tf

# Load original model (works on your local machine)
model = tf.keras.models.load_model(
    "models/autoencoder.keras",
    compile=False,
    safe_mode=False
)

# Rebuild model WITHOUT problematic config
clean_model = tf.keras.Model(
    inputs=model.inputs,
    outputs=model.outputs
)

# Copy trained weights
clean_model.set_weights(model.get_weights())

# Save clean model
clean_model.save("models/autoencoder_new.h5")