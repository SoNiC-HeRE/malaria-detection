import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Set sys.stdout encoding to 'utf-8' if needed
# import sys
# sys.stdout.reconfigure(encoding='utf-8')

train_data_dir = 'C:/Users/e0075ax 246776/Desktop/malaria_detection_project 18-04-2024/malaria_detection_project/malaria_detection/path_to_train_directory'
validation_data_dir = 'C:/Users/e0075ax 246776/Desktop/malaria_detection_project 18-04-2024/malaria_detection_project/malaria_detection/path_to_validation_directory'

img_height, img_width = 224, 224
batch_size = 32
epochs = 10  # Adjust as needed

# Training and validation data generators
train_datagen = ImageDataGenerator(
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True)

train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical')  

validation_datagen = ImageDataGenerator(rescale=1./255)

validation_generator = validation_datagen.flow_from_directory(
    validation_data_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical')

# Load pre-trained MobileNetV2 model 
base_model = MobileNetV2(weights='imagenet', include_top=False)

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(128, activation='relu')(x)
predictions = Dense(2, activation='sigmoid')(x)

model = Model(inputs=base_model.input, outputs=predictions)

# Freeze all layers of the base model
for layer in base_model.layers:
    layer.trainable = False

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // batch_size,
    validation_data=validation_generator,
    validation_steps=validation_generator.samples // batch_size,
    epochs=epochs)

# Save the model
model.save('malaria_parasite_detection_model.h5')
