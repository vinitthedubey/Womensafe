import tensorflow as tf

# Load the existing SavedModel
saved_model = tf.saved_model.load('gender_detection.model')

# Convert and save as H5 format
tf.keras.saving.save_model(saved_model, 'gender_detection.h5', save_format='h5')
