import tensorflow as tf
import os
import pandas as pd
from tensorflow.keras.preprocessing import image

# Example of how to load a multi-label dataset
train_dir = 'path_to_train_data'
valid_dir = 'path_to_valid_data'
labels_csv = 'path_to_labels_csv'  # CSV with image filenames and multi-hot encoded labels

# Load label data from CSV file
labels_df = pd.read_csv(labels_csv)

# Helper function to load and preprocess images
def load_and_preprocess_image(img_path, label):
    img = image.load_img(img_path, target_size=(224, 224))
    img = image.img_to_array(img) / 255.0  # Normalize image to [0, 1]
    return img, label

# Function to create a tf.data.Dataset from CSV
def create_dataset(directory, labels_df, batch_size=32):
    img_paths = labels_df['image'].apply(lambda x: os.path.join(directory, x)).values
    labels = labels_df.drop(columns=['image']).values  # Multi-hot labels
    
    # Create a tf.data.Dataset
    dataset = tf.data.Dataset.from_tensor_slices((img_paths, labels))
    
    # Load and preprocess images
    dataset = dataset.map(lambda x, y: tf.py_function(load_and_preprocess_image, [x, y], [tf.float32, tf.float32]))
    
    # Cache, shuffle, batch, and prefetch
    dataset = dataset.cache().shuffle(buffer_size=1000).batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE)
    return dataset

# Create training and validation datasets
train_dataset = create_dataset(train_dir, labels_df, batch_size=32)
valid_dataset = create_dataset(valid_dir, labels_df, batch_size=32)

# Build the model as before
base_model = ResNet50(weights='imagenet', include_top=False)
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation='relu')(x)
output = Dense(14, activation='sigmoid')(x)  # 14 output classes for multi-label classification

# Define the model
model = Model(inputs=base_model.input, outputs=output)

# Freeze the base model layers for transfer learning
for layer in base_model.layers:
    layer.trainable = False

# Compile the model
model.compile(optimizer=Adam(), loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(train_dataset, epochs=10, validation_data=valid_dataset)

# Evaluate the model
model.evaluate(valid_dataset)
