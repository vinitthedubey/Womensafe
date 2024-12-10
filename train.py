from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import BatchNormalization, Conv2D, MaxPooling2D, Activation, Flatten, Dropout, Dense
from tensorflow.keras import backend as K
from sklearn.model_selection import train_test_split
import joblib  # Updated import
import matplotlib.pyplot as plt
import numpy as np
import random
import cv2
import os
import glob


# initial parameters
epochs = 50
learning_rate = 1e-3
batch_size = 64
img_dims = (96, 96, 3)

data = []
labels = []

# load image files from the dataset
image_files = [f for f in glob.glob(r'/content/Gender-Detection/gender_dataset_face' + "/**/*", recursive=True) if not os.path.isdir(f)]
random.shuffle(image_files)

# converting images to arrays and labelling the categories
for img in image_files:
    image = cv2.imread(img)
    image = cv2.resize(image, (img_dims[0], img_dims[1]))
    image = img_to_array(image)
    data.append(image)

    label = img.split(os.path.sep)[-2]  # e.g., 'woman'
    if label == "woman":
        label = 1
    else:
        label = 0

    labels.append([label])  # [[1], [0], [0], ...]

# pre-processing
data = np.array(data, dtype="float") / 255.0
labels = np.array(labels)

# split dataset for training and validation
(trainX, testX, trainY, testY) = train_test_split(data, labels, test_size=0.2, random_state=42)

trainY = to_categorical(trainY, num_classes=2)  # [[1, 0], [0, 1], [0, 1], ...]
testY = to_categorical(testY, num_classes=2)

# augmenting dataset
aug = ImageDataGenerator(rotation_range=25, width_shift_range=0.1,
                         height_shift_range=0.1, shear_range=0.2, zoom_range=0.2,
                         horizontal_flip=True, fill_mode="nearest")

# function to define model variations
def build_model(conv_layers, dense_layers, width, height, depth, classes):
    model = Sequential()
    inputShape = (height, width, depth)
    chanDim = -1

    if K.image_data_format() == "channels_first":
        inputShape = (depth, height, width)
        chanDim = 1

    # Add convolutional layers
    for i in range(conv_layers):
        model.add(Conv2D(32 * (2 ** i), (3, 3), padding="same", input_shape=inputShape if i == 0 else None))
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis=chanDim))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))

    model.add(Flatten())

    # Add dense layers
    for i in range(dense_layers):
        model.add(Dense(512))
        model.add(Activation("relu"))
        model.add(BatchNormalization())
        model.add(Dropout(0.5))

    model.add(Dense(classes))
    model.add(Activation("sigmoid"))

    return model

# Model variations for training
models = [
    {"conv_layers": 2, "dense_layers": 1},  # Model 1
    {"conv_layers": 3, "dense_layers": 1},  # Model 2
    {"conv_layers": 3, "dense_layers": 2},  # Model 3
    {"conv_layers": 4, "dense_layers": 2},  # Model 4
    {"conv_layers": 4, "dense_layers": 3},  # Model 5
]

best_accuracy = 0
best_model = None
best_model_name = ""

# Train each model and track accuracy
for i, model_params in enumerate(models):
    print(f"Training Model {i+1} with {model_params['conv_layers']} conv layers and {model_params['dense_layers']} dense layers")

    # Build model
    model = build_model(conv_layers=model_params["conv_layers"],
                        dense_layers=model_params["dense_layers"],
                        width=img_dims[0], height=img_dims[1], depth=img_dims[2], classes=2)

    # Compile the model
    opt = Adam(learning_rate=learning_rate)
    model.compile(loss="binary_crossentropy", optimizer=opt, metrics=["accuracy"])

    # Train the model
    H = model.fit(aug.flow(trainX, trainY, batch_size=batch_size),
                  validation_data=(testX, testY),
                  steps_per_epoch=len(trainX) // batch_size,
                  epochs=epochs, verbose=1)

    # Get accuracy for this model
    accuracy = max(H.history['val_accuracy'])
    print(f"Model {i+1} validation accuracy: {accuracy:.4f}")

    # Save the most accurate model
    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_model = model
        best_model_name = f"gender_detection_model_{i+1}.pkl"

# Save the best model
joblib.dump(best_model, best_model_name)
print(f"Best model saved as {best_model_name} with accuracy: {best_accuracy:.4f}")
