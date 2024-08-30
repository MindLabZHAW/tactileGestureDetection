#CNN+K-Fold Cross-Validation:Average accuracy using K-Fold Cross-Validation: 0.8571
import os
import pandas as pd
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.model_selection import KFold

# Read data and images
data_path = "DATA/STFT_images/"
file = os.path.join(data_path, 'image_records.csv')
image_df = pd.read_csv(file)

images = []
labels = []

# Load and preprocess images
for index, row in image_df.iterrows():
    img_path = os.path.join(data_path, row['image_filename'])
    img = Image.open(img_path).convert('RGB')
    img = img.resize((128, 128))
    img = np.array(img)
    images.append(img)
    labels.append(row['touch_type'])

images = np.array(images)
labels = np.array(labels)

# Encode labels
label_classes = np.unique(labels)
label_map = {label: idx for idx, label in enumerate(label_classes)}
labels = np.array([label_map[label] for label in labels])
labels = to_categorical(labels, num_classes=len(label_classes))

# Normalize images
images = images / 255.0

# Initialize K-Fold Cross-Validation
kf = KFold(n_splits=10, shuffle=True, random_state=42)
accuracies = []
histories = []  # Store the history of each fold

# Iterate through each fold
for train_index, test_index in kf.split(images):
    X_train, X_test = images[train_index], images[test_index]
    y_train, y_test = labels[train_index], labels[test_index]
    
    # Simple CNN model
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 3)),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(64, activation='relu'),
        Dense(len(label_classes), activation='softmax')
    ])

    # Compile model
    model.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])

    # Early stopping callback
    early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

    # Train model with early stopping
    history = model.fit(X_train, y_train, epochs=20, batch_size=16, verbose=0, validation_data=(X_test, y_test), callbacks=[early_stopping])
    
    # Evaluate model
    loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
    accuracies.append(accuracy)
    histories.append(history)

# Print accuracies
print(f"Accuracies are {accuracies}")

# Calculate mean accuracy
mean_accuracy = np.mean(accuracies)
print(f'Average accuracy using K-Fold Cross-Validation: {mean_accuracy:.4f}')

# Plot accuracy
plt.figure(figsize=(12, 4))
plt.plot(histories[0].history['accuracy'], label='Training accuracy')
plt.plot(histories[0].history['val_accuracy'], label='Validation accuracy')
plt.title('Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()
