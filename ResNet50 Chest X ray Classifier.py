import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import cv2
import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

# Define paths to the main folders
main_folder_path = "./chest_xray"
train_folder_path = main_folder_path + "/train"
test_folder_path = main_folder_path + "/test"
val_folder_path = main_folder_path + "/val"

# Check folder existence
print("Train folder contents:", os.listdir(train_folder_path))
data = []

# Walk through the directory structure
for dataset_type in ["train", "test", "val"]:
    dataset_folder = os.path.join(main_folder_path, dataset_type)

    for label in ["NORMAL", "PNEUMONIA"]:
        label_folder = os.path.join(dataset_folder, label)

        # Check if the folder exists
        if not os.path.exists(label_folder):
            print(f"Folder not found: {label_folder}")
            continue

        for filename in os.listdir(label_folder):
            file_path = os.path.join(label_folder, filename)

            # Ensure it's a file (and you can add more checks like for image extensions)
            if os.path.isfile(file_path):
                data.append(
                    {
                        "file_path": file_path,
                        "label": label,
                        "dataset_type": dataset_type,
                    }
                )

# Define our dataframe
data_df = pd.DataFrame(data)
data_df.shape

# Display the first few rows of the DataFrame and some info
print("DataFrame created successfully!")
print("\nFirst 5 rows:")
print(data_df.head())
print("\nDataFrame Info:")
data_df.info()
print("\nValue Counts for each dataset type:")
print(data_df["dataset_type"].value_counts())
print("\nValue Counts for labels:")
print(data_df["label"].value_counts())

# Define constants
IMG_WIDTH, IMG_HEIGHT = 224, 224  # Default input size for ResNet50

# 1. Load the Pre-trained ResNet50 Model
# We use `include_top=False` to discard the final classification layer from ImageNet
# `weights='imagenet'` specifies that we want to load the weights pre-trained on ImageNet
base_model = ResNet50(
    weights="imagenet", include_top=False, input_shape=(IMG_WIDTH, IMG_HEIGHT, 3)
)

# 2. Freeze the Base Layers
# We don't want to retrain the feature-extraction layers, so we freeze them.
for layer in base_model.layers:
    layer.trainable = False

# 3. Add a New Classification Head
# Get the output of the base model
x = base_model.output

# Add a Global Average Pooling layer to flatten the features
x = GlobalAveragePooling2D()(x)

# Add a fully-connected layer with 1024 neurons and ReLU activation
x = Dense(1024, activation="relu")(x)

# Add a Dropout layer for regularization to prevent overfitting
x = Dropout(0.5)(x)

# Add the final output layer.
# It has one neuron because this is a binary classification (Normal vs. Pneumonia).
# We use the 'sigmoid' activation function to output a probability between 0 and 1.
predictions = Dense(1, activation="sigmoid")(x)

# Create the final model by combining the base model and the new head
model = Model(inputs=base_model.input, outputs=predictions)

# 4. Compile the Model
# We use the Adam optimizer, which is a good default choice.
# For binary classification, 'binary_crossentropy' is the standard loss function.
# We'll monitor 'accuracy' during training.
model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

# Display the model's architecture
print("Model architecture created successfully!")
model.summary()

# Define batch size
BATCH_SIZE = 32
# Assuming your DataFrame is named 'df' from the first step
# If not, please change the name accordingly.

# 1. Split the main DataFrame into train, validation, and test sets
train_df = data_df[data_df["dataset_type"] == "train"]
val_df = data_df[data_df["dataset_type"] == "val"]
test_df = data_df[data_df["dataset_type"] == "test"]

# Optional: You can check the counts to make sure the split is correct
print(f"Training images: {len(train_df)}")
print(f"Validation images: {len(val_df)}")
print(f"Test images: {len(test_df)}")

# 2. Create a data generator for the training set WITH data augmentation
train_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    rotation_range=20,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True,
    fill_mode="nearest",
)

# 3. Create a data generator for the validation and test sets WITHOUT data augmentation
validation_test_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

# 4. Create the generators from our DataFrames
train_generator = train_datagen.flow_from_dataframe(
    dataframe=train_df,
    x_col="file_path",  # Column with image paths
    y_col="label",  # Column with the labels
    target_size=(IMG_WIDTH, IMG_HEIGHT),
    batch_size=BATCH_SIZE,
    class_mode="binary",  # For binary classification
    shuffle=True,
)

validation_generator = validation_test_datagen.flow_from_dataframe(
    dataframe=val_df,
    x_col="file_path",
    y_col="label",
    target_size=(IMG_WIDTH, IMG_HEIGHT),
    batch_size=BATCH_SIZE,
    class_mode="binary",
    shuffle=False,  # No need to shuffle validation data
)

test_generator = validation_test_datagen.flow_from_dataframe(
    dataframe=test_df,
    x_col="file_path",
    y_col="label",
    target_size=(IMG_WIDTH, IMG_HEIGHT),
    batch_size=BATCH_SIZE,
    class_mode="binary",
    shuffle=False,  # No need to shuffle test data
)

# Check if the generators were created successfully
print("\nData generators created successfully from DataFrame!")
print("Class indices used by generators:", train_generator.class_indices)

# Define the number of epochs to train for
# An epoch is one complete pass through the entire training dataset.
# 10 is a good starting point for transfer learning.
EPOCHS = 10


# Define callbacks
# 1. ModelCheckpoint to save the best model during training
# The model will be saved to a file named 'best_model.keras'
# It will only save when the 'val_accuracy' metric improves.
checkpoint_cb = ModelCheckpoint(
    "best_model.keras", save_best_only=True, monitor="val_accuracy", mode="max"
)


# 2. EarlyStopping to halt training when the model stops improving
# It will monitor 'val_loss'. If it doesn't improve for 3 consecutive epochs ('patience=3'),
# training will stop. 'restore_best_weights=True' ensures the model's weights are
# reset to those of the best epoch.
early_stopping_cb = EarlyStopping(
    patience=3, monitor="val_loss", restore_best_weights=True
)

# Start training the model!
# The `fit` function returns a 'history' object, which contains a record of the
# training and validation loss and accuracy for each epoch.
history = model.fit(
    train_generator,
    steps_per_epoch=len(train_generator),  # Number of batches per epoch
    epochs=EPOCHS,
    validation_data=validation_generator,
    validation_steps=len(validation_generator),  # Number of batches for validation
    callbacks=[checkpoint_cb, early_stopping_cb],
)

# display model summary after training
model.summary()


# Use the .evaluate() method on the test generator
# This will return the loss and accuracy on the test set
print("Evaluating the model on the test dataset...")
test_loss, test_accuracy = model.evaluate(test_generator, steps=len(test_generator))

print(f"\nTest Loss: {test_loss:.4f}")
print(f"Test Accuracy: {test_accuracy:.4f}")

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix

# 1. Get predictions from the model
# The `predict` method returns the raw probability scores
predictions_prob = model.predict(test_generator, steps=len(test_generator))

# 2. Convert probabilities to class labels
# If the probability is > 0.5, we classify it as PNEUMONIA (class 1), otherwise NORMAL (class 0)
predicted_classes = (predictions_prob > 0.5).astype(int).flatten()

# 3. Get the true labels
true_classes = test_generator.classes
class_labels = list(test_generator.class_indices.keys())

# 4. Print the Classification Report
print("\nClassification Report:\n")
report = classification_report(
    true_classes, predicted_classes, target_names=class_labels
)
print(report)

# 5. Generate and Display the Confusion Matrix
print("\nConfusion Matrix:\n")
cm = confusion_matrix(true_classes, predicted_classes)

# Use seaborn for a nice heatmap visualization of the confusion matrix
plt.figure(figsize=(6, 5))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=class_labels,
    yticklabels=class_labels,
)
plt.title("Confusion Matrix")
plt.ylabel("True Label")
plt.xlabel("Predicted Label")
plt.show()
