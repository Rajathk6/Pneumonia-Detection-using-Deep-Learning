# Import necessary libraries
import warnings
warnings.filterwarnings('ignore')

# TensorFlow and Keras modules
from tensorflow.keras.applications import VGG16
from tensorflow.keras.layers import Input, Lambda, Dense, Flatten
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from tensorflow.keras.optimizers import Adam
import numpy as np
from glob import glob
import matplotlib.pyplot as plt

# Set dataset paths and image size
IMAGE_SIZE = [224, 224]
TRAIN_PATH = 'C:\\Users\\HP\\Desktop\\DELETE\\chest_xray\\chest_xray\\train'
VALID_PATH = 'C:\\Users\\HP\\Desktop\\DELETE\\chest_xray\\chest_xray\\test'

# Load the VGG16 model without the top layers and with pre-trained ImageNet weights
vgg = VGG16(input_shape=IMAGE_SIZE + [3], weights='imagenet', include_top=False)

# Freeze all the layers of VGG16
for layer in vgg.layers:
    layer.trainable = False

# Adding custom layers on top of VGG16
folders = glob(f'{TRAIN_PATH}/*')  # Get class folders
x = Flatten()(vgg.output)
prediction = Dense(len(folders), activation='softmax')(x)  # Output layer

# Build the model
model = Model(inputs=vgg.input, outputs=prediction)
model.summary()

# Compile the model
model.compile(
    loss='categorical_crossentropy',
    optimizer=Adam(learning_rate=0.0001),
    metrics=['accuracy']
)

# Image data augmentation and preprocessing
train_datagen = ImageDataGenerator(
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)

valid_datagen = ImageDataGenerator(rescale=1./255)

# Load training and validation datasets
train_set = train_datagen.flow_from_directory(
    TRAIN_PATH,
    target_size=IMAGE_SIZE,
    batch_size=32,
    class_mode='categorical'
)

valid_set = valid_datagen.flow_from_directory(
    VALID_PATH,
    target_size=IMAGE_SIZE,
    batch_size=32,
    class_mode='categorical'
)

# Train the model
history = model.fit(
    train_set,
    validation_data=valid_set,
    epochs=5,  # Use more epochs as needed
    steps_per_epoch=len(train_set),
    validation_steps=len(valid_set)
)

# Save the model
model.save('chest_xray_vgg16.h5')

# Plot training and validation accuracy
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.legend()
plt.show()

# Plot training and validation loss
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.legend()
plt.show()

# Load and preprocess a single image for prediction
def predict_image(model_path, image_path):
    model = load_model(model_path)
    img = load_img(image_path, target_size=IMAGE_SIZE)
    x = img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = x / 255.0  # Rescale the image
    
    prediction = model.predict(x)
    class_idx = np.argmax(prediction, axis=1)[0]
    class_label = list(train_set.class_indices.keys())[class_idx]
    return class_label, prediction
# Evaluate the model on the validation set
val_loss, val_accuracy = model.evaluate(valid_set, verbose=1)
print(f"Validation Loss: {val_loss:.4f}")
print(f"Validation Accuracy: {val_accuracy*100:.2f}%")
                              
# Example usage of the prediction function
image_path = 'C:\\Users\\HP\\Desktop\\DELETE\\chest_xray\\val\\NORMAL\\NORMAL2-IM-1427-0001.jpeg'
label, prediction = predict_image('chest_xray_vgg16.h5', image_path)
print(f'Predicted Label: {label}, Prediction Probabilities: {prediction}')