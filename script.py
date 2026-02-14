import tensorflow as tf

# load original model (works locally)
model = tf.keras.models.load_model(
    "models/autoencoder.keras",
    compile=False,
    safe_mode=False
)

# recreate model WITHOUT training config
new_model = tf.keras.Model(
    inputs=model.inputs,
    outputs=model.outputs
)

# copy weights
new_model.set_weights(model.get_weights())

# save clean version
new_model.save("models/autoencoder_clean.h5")