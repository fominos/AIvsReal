from matplotlib import pyplot as plt
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing.image import load_img, img_to_array, array_to_img
import os
import zipfile
from random import shuffle
from glob import glob
import joblist
IMG_SIZE = (224, 224)  # размер входного изображения сети
train_files = []
with zipfile.ZipFile('AIvsReal.zip', 'r') as zip_ref:
    zip_ref.extractall('ar_data')
for filename in os.listdir('ar_data/RealArt/RealArt'):
    train_files.append(os.path.join(filename))
# Extracting files from the folder
train_files_real = glob('ar_data/RealArt/RealArt/*')
train_file_ai = glob('ar_data/AiArtData/AiArtData/*')
# Function to add a prefix to files in the directory


def add_prefix_to_files(file_list, prefix):
    for file in file_list:
        # Getting the directory and file name
        directory = os.path.dirname(file)
        filename = os.path.basename(file)

        # Creating a new file name with a prefix
        new_filename = prefix + filename
        new_file_path = os.path.join(directory, new_filename)

        # Renaming the file
        os.rename(file, new_file_path)


# Adding prefixes to files
add_prefix_to_files(train_files_real, 'real_')
add_prefix_to_files(train_file_ai, 'ai_')
train_files_real = glob('ar_data/RealArt/RealArt/*')
train_file_ai = glob('ar_data/AiArtData/AiArtData/*')
train_files = train_files_real+train_file_ai
# Creating an ImageDataGenerator object with augmentation
datagen = ImageDataGenerator(
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)
# Folder for saving augmented images
output_dir = 'add_images'
os.makedirs(output_dir, exist_ok=True)

# Iterating through each file in the list
for file in train_files:
    # Loading the image
    img = load_img(file)
    x = img_to_array(img)
    x = x.reshape((1,) + x.shape)
    base_filename = os.path.splitext(os.path.basename(file))[0]
    # Generating augmented images
    i = 0
    for batch in datagen.flow(x, batch_size=1, save_to_dir=output_dir, save_prefix=base_filename, save_format='jpg'):
        i += 1
        if i >= 1:
            break
for filename in os.listdir(output_dir):
    train_files.append(os.path.join(output_dir, filename))
# Loading the input image and preprocessing


def load_image(path, target_size=IMG_SIZE):
    img = cv2.imread(path)[..., ::-1]
    img = cv2.resize(img, target_size)
    return preprocess_input(img)  # preprocessing for MobileNetV2
# Generator function for loading training data from disk


def fit_generator(files, batch_size=32):
    batch_size = min(batch_size, len(files))
    while True:
        shuffle(files)
        for k in range(len(files) // batch_size):
            i = k * batch_size
            j = i + batch_size
            if j > len(files):
                j = - j % len(files)
            x = np.array([load_image(path) for path in files[i:j]])
            y = np.array([1. if os.path.basename(path).startswith('real') else 0.
                          for path in files[i:j]])
            yield (x, y)
# Generator function for loading test images from disk


def predict_generator(files):
    while True:
        for path in files:
            img = np.array([load_image(path)])
            yield (img,)


# Loading the pre-trained model
# base_model - an object of the keras.models.Model class(Functional Model)
base_model = MobileNetV2(weights='imagenet',
                         include_top=False,
                         input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3))
# Freezing all weights of the pre-trained network
for layer in base_model.layers:
    layer.trainable = False

x = base_model.layers[-12].output
x = layers.GlobalAveragePooling2D()(x)  # Adding a global average pooling layer
x = layers.Dense(1,  # One output (binary classification)
                 activation='sigmoid',  # Activation function
                 kernel_regularizer=tf.keras.regularizers.l1(1e-4))(x)

# Creating the model
model = tf.keras.Model(inputs=base_model.input, outputs=x, name='aivsreal')
# Compiling the model by specifying the loss function, optimizer, and metrics.
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])
# Starting the training process
shuffle(train_files)  # Shuffling the training dataset
train_datafiles, validation_files = train_test_split(
    train_files, test_size=0.2, random_state=42)
validation_data = next(fit_generator(validation_files))
# Reading data using the generator function
train_data = fit_generator(train_datafiles)
# Starting the training process
model.fit(train_data,
          steps_per_epoch=20,
          epochs=100,
          validation_data=validation_data)
model.save('aivsreal.keras')
# Save the model
joblist.dump(model, 'model.pkl')
