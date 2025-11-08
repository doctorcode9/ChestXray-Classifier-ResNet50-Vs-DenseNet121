import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import cv2
import tensorflow as tf

# building DenseNet121 model
from tensorflow.keras.applications import DenseNet121
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


# Load the Pre-trained DenseNet121 Model (include_top=False)
base_model_densenet = DenseNet121(
    weights="imagenet", include_top=False, input_shape=(IMG_WIDTH, IMG_HEIGHT, 3)
)

# Freeze the base layers
for layer in base_model_densenet.layers:
    layer.trainable = False

# Add the same custom classification head as before
x = base_model_densenet.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation="relu")(x)
x = Dropout(0.5)(x)
predictions = Dense(1, activation="sigmoid")(x)

# Create the final model
densenet_model = Model(inputs=base_model_densenet.input, outputs=predictions)

# Compile the model
densenet_model.compile(
    optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"]
)

print("--- DenseNet121 Model Architecture ---")
densenet_model.summary()

# --- 3. TRAIN THE DENSENET121 MODEL ---

print("\n--- Starting Training for DenseNet121 Model ---")

# Let's save the new model to a different file to avoid overwriting the ResNet model
checkpoint_cb_densenet = ModelCheckpoint(
    "best_model_densenet.keras", save_best_only=True, monitor="val_accuracy", mode="max"
)

early_stopping_cb = EarlyStopping(
    patience=3, monitor="val_loss", restore_best_weights=True
)

EPOCHS = 10

# Train the new model and store its history in a new variable
history_densenet = densenet_model.fit(
    train_generator_densenet,
    steps_per_epoch=len(train_generator_densenet),
    epochs=EPOCHS,
    validation_data=validation_generator_densenet,
    validation_steps=len(validation_generator_densenet),
    callbacks=[checkpoint_cb_densenet, early_stopping_cb],
)

print("\nDenseNet121 training complete!")
print("The best DenseNet121 model has been saved as 'best_model_densenet.keras'")

from tensorflow.keras.models import load_model

# Load the best DenseNet121 model that was saved during training
best_densenet_model = load_model("best_model_densenet.keras")

# Evaluate the model on the test generator for DenseNet
# This generator uses the correct preprocessing for this model
print("Evaluating the DenseNet121 model on the test dataset...")
test_loss_densenet, test_accuracy_densenet = best_densenet_model.evaluate(
    test_generator_densenet, steps=len(test_generator_densenet)
)

print(f"\nDenseNet121 Test Loss: {test_loss_densenet:.4f}")
print(f"DenseNet121 Test Accuracy: {test_accuracy_densenet:.4f}")

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix

# 1. Get predictions from the DenseNet121 model
predictions_prob_densenet = best_densenet_model.predict(
    test_generator_densenet, steps=len(test_generator_densenet)
)

# 2. Convert probabilities to class labels (0 or 1)
predicted_classes_densenet = (predictions_prob_densenet > 0.5).astype(int).flatten()

# 3. Get the true labels and class names
true_classes_densenet = test_generator_densenet.classes
class_labels_densenet = list(test_generator_densenet.class_indices.keys())

# 4. Print the Classification Report
print("\nDenseNet121 Classification Report:\n")
report_densenet = classification_report(
    true_classes_densenet,
    predicted_classes_densenet,
    target_names=class_labels_densenet,
)
print(report_densenet)

# 5. Generate and Display the Confusion Matrix
print("\nDenseNet121 Confusion Matrix:\n")
cm_densenet = confusion_matrix(true_classes_densenet, predicted_classes_densenet)

plt.figure(figsize=(6, 5))
sns.heatmap(
    cm_densenet,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=class_labels_densenet,
    yticklabels=class_labels_densenet,
)
plt.title("DenseNet121 - Confusion Matrix")
plt.ylabel("True Label")
plt.xlabel("Predicted Label")
plt.show()

# display model summary after training
densenet_model.summary()

import matplotlib.pyplot as plt
import numpy as np

# 1. Get a batch of images and labels from the validation generator
images_val, true_labels_val = next(validation_generator_densenet)

# Get the class names from the generator
class_labels = list(validation_generator_densenet.class_indices.keys())

# 2. Make predictions on this batch using the best DenseNet model
# The 'best_densenet_model' should already be loaded.
predictions_prob_val = best_densenet_model.predict(images_val)
predicted_labels_int_val = (predictions_prob_val > 0.5).astype(int).flatten()


# 3. Denormalize the images for visualization
# This function scales pixel values to a 0-1 range, making them easy to display.
def denormalize_image(image):
    image = image.copy()
    image -= image.min()  # Shift the minimum value to 0
    image /= image.max()  # Scale the maximum value to 1
    return image


# 4. Create a grid of images with their predictions and true labels
plt.figure(figsize=(15, 10))

# Display up to the first 16 images from the batch
num_images_to_show = min(len(images_val), 16)

for i in range(num_images_to_show):
    plt.subplot(4, 4, i + 1)

    # Get the predicted and true label names
    predicted_label_name = class_labels[predicted_labels_int_val[i]]
    true_label_name = class_labels[int(true_labels_val[i])]

    # Determine the color for the title based on correctness
    title_color = "green" if predicted_label_name == true_label_name else "red"

    # Denormalize and display the image
    plt.imshow(denormalize_image(images_val[i]))

    # Set the title with the prediction and true label
    plt.title(
        f"Pred: {predicted_label_name}\nTrue: {true_label_name}", color=title_color
    )

    # Hide the axes for a cleaner look
    plt.axis("off")

plt.tight_layout()
plt.show()
