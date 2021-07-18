# Import Library & ZIP
import splitfolders
import zipfile
import os
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from tensorflow.keras.layers import Conv2D, MaxPooling2D, Activation, Dense, Flatten
from tensorflow.keras.layers import Dropout
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from keras.preprocessing import image
from google.colab import files

print(tf.__version__)

!wget - -no-check-certificate \
    https: // dicodingacademy.blob.core.windows.net/picodiploma/ml_pemula_academy/rockpaperscissors.zip \
    - O / tmp/rockpaperscissors.zip

# Ekstrak ZIP
zipku = '/tmp/rockpaperscissors.zip'
zip_zip = zipfile.ZipFile(zipku, 'r')
zip_zip.extractall('/tmp')
zip_zip.close()

# Cek Data ZIP
folder_scissors = os.path.join("/tmp/rockpaperscissors/rps-cv-images/scissors")
folder_paper = os.path.join("/tmp/rockpaperscissors/rps-cv-images/paper")
folder_rock = os.path.join("/tmp/rockpaperscissors/rps-cv-images/rock")
folder_tmp = "/tmp/rockpaperscissors/rps-cv-images"

# Hitung panjang data
cek_scissors = len(os.listdir(folder_scissors))
cek_paper = len(os.listdir(folder_paper))
cek_rock = len(os.listdir(folder_rock))
print(cek_scissors, cek_paper, cek_rock)

# Split Folder
pip install split-folders

# Tentukan Ratio
splitfolders.ratio('/tmp/rockpaperscissors/rps-cv-images',
                   output="/tmp/rockpaperscissors/data", seed=1337, ratio=(.8, .2))

# Cek Classes
os.listdir('/tmp/rockpaperscissors/data/train')
os.listdir('/tmp/rockpaperscissors/data/val')
# Output 1 berarti sama

file_main = '/tmp/rockpaperscissors/data'
dir_train = os.path.join(file_main, 'train')
dir_validation = os.path.join(file_main, 'val')

# Set Data
val_size = 0.4

# Validasi Generator
train_datagen = ImageDataGenerator(
    rotation_range=30,
    brightness_range=[0.2, 1.0],
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode="nearest",
    rescale=1./255,
    validation_split=val_size
)

test_datagen = ImageDataGenerator(
    rotation_range=30,
    brightness_range=[0.2, 1.0],
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode="nearest",
    rescale=1./255,
    validation_split=val_size
)

train_generator = train_datagen.flow_from_directory(
    folder_tmp,
    target_size=(150, 150),
    color_mode="rgb",
    class_mode="categorical",
    batch_size=16,
    shuffle=True,
    subset="training"
)

validation_generator = test_datagen.flow_from_directory(
    folder_tmp,
    target_size=(150, 150),
    color_mode="rgb",
    class_mode="categorical",
    batch_size=16,
    shuffle=False,
    subset="validation"
)

# Model
# RELU
Model = Sequential([
    Conv2D(32, (3, 3), strides=(1, 1),
           activation='relu', input_shape=(150, 150, 3)),
    MaxPooling2D(pool_size=(2, 2), padding='valid'),
    Conv2D(64, (3, 3), strides=(1, 1), activation='relu'),
    MaxPooling2D(pool_size=(2, 2), padding='valid'),
    Conv2D(128, (3, 3), strides=(1, 1), activation='relu'),
    MaxPooling2D(pool_size=(2, 2), padding='valid'),
    Flatten(),
    Dropout(0.2),
    Dense(128, activation='relu'),
    Dense(3, activation='softmax')
])

Adam(learning_rate=0.00146, name='Adam')
Model.compile(optimizer='Adam', loss='categorical_crossentropy',
              metrics=['accuracy'])

# Training
Model.fit(
    train_generator,
    steps_per_epoch=30,
    epochs=20,
    validation_data=validation_generator,
    validation_steps=5,
    verbose=2)

# Prediksi
uploaded = files.upload()

for up_file in uploaded.keys():
    path = up_file
    img = image.load_img(path, target_size=(150, 150))
    imgplot = plt.imshow(img)
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)

    images = np.vstack([x])
    classes = Model.predict(images, batch_size=16)
    hasil = np.argmax(classes)

    # 0 == scissors
    # 1 == rock
    # 2 == paper

    print(up_file)
    if hasil == 0:
        print('scissors')
    elif hasil == 1:
        print('rock')
    elif hasil == 2:
        print('paper')
