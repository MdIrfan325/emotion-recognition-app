# src/train_mobilenet.py

import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os

# Set paths
train_dir = "../fer2013_data/train"
val_dir = "../fer2013_data/test"

# Data generators (RGB for MobileNetV2)
train_datagen = ImageDataGenerator(rescale=1./255, horizontal_flip=True)
val_datagen = ImageDataGenerator(rescale=1./255)

train_gen = train_datagen.flow_from_directory(
    train_dir,
    target_size=(48, 48),
    color_mode='rgb',
    class_mode='categorical',
    batch_size=64,
    shuffle=True
)

val_gen = val_datagen.flow_from_directory(
    val_dir,
    target_size=(48, 48),
    color_mode='rgb',
    class_mode='categorical',
    batch_size=64,
    shuffle=False
)

# Build MobileNetV2 model
base_model = MobileNetV2(input_shape=(48, 48, 3), include_top=False, weights='imagenet')
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dropout(0.3)(x)
predictions = Dense(7, activation='softmax')(x)

mobilenet_model = Model(inputs=base_model.input, outputs=predictions)

# Freeze base layers
for layer in base_model.layers:
    layer.trainable = False

mobilenet_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train
mobilenet_model.fit(train_gen, validation_data=val_gen, epochs=15)

# Save TFLite model
converter = tf.lite.TFLiteConverter.from_keras_model(mobilenet_model)
tflite_model = converter.convert()
with open("../models/mobilenetv2_model.tflite", "wb") as f:
    f.write(tflite_model)
