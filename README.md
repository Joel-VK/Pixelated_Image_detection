# Pixelated_Image_detection

Here's a sample README file for your project:

---

# **Pixelation Detection Model**

## **Table of Contents**

1. [Project Overview](#project-overview)
2. [Dataset](#dataset)
3. [Model Architecture](#model-architecture)
4. [Training Process](#training-process)
5. [Evaluation](#evaluation)
6. [Usage](#usage)
7. [Results](#results)
8. [Future Work](#future-work)
9. [Contributing](#contributing)
10. [License](#license)

## **Project Overview**

This project aims to develop a lightweight, efficient Convolutional Neural Network (CNN) model for detecting pixelated images. The goal is to  ensure the model can run at > 60 FPS for real-time applications, particularly on 1080p resolution images. The model's effectiveness is evaluated based on metrics focusing on minimizing false positives.

## **Dataset**

The dataset consists of original and pixelated images, with an equal number of images in each class:
- **Train Set:** 1199 original and 1199 pixelated images.
- **Test Set:** 453 original and 453 pixelated images.

The dataset was further augmented to enhance model performance, generating additional training samples with transformations such as rotation, zoom, and flips.

## **Model Architecture**

The model is a complex CNN with multiple convolutional blocks, each followed by batch normalization, LeakyReLU activation, and max-pooling layers. The architecture includes:
- **5 Convolutional Blocks** with varying filters (32 to 512).
- **Global Average Pooling** to reduce overfitting.
- **Fully Connected Layers** with dropout for regularization.
- **Output Layer** with sigmoid activation for binary classification.

### **Optimizer and Loss Function**
- **Optimizer:** Nadam (Nesterov-accelerated Adaptive Moment Estimation) with an initial learning rate of 0.001.
- **Loss Function:** Binary Cross-Entropy with class weights to handle class imbalance.

## **Training Process**

The model was trained using the following strategy:
- **Data Augmentation:** Applied to the training set to increase data diversity.
- **Class Weights:** Computed to balance the dataset and improve model accuracy.
- **Learning Rate Scheduler:** ReduceLROnPlateau callback to reduce learning rate when validation accuracy plateaus.
- **Checkpointing:** The best model (based on validation accuracy) was saved during training.

Training was conducted over 20 epochs with batch size set to 16, leveraging the Adam optimizer and class weights to address any class imbalance.

## **Evaluation**

After training, the model was evaluated on the test set:
- **Accuracy:** The test accuracy was observed to be over 76%.
- **Precision-Recall Curve:** Used to determine the optimal decision threshold to minimize false positives.
- **Confusion Matrix:** Provided insights into the number of false positives and false negatives.

## **Usage**

### **Prerequisites**

- Python 3.x
- TensorFlow 2.x
- Required libraries: `numpy`, `scikit-learn`, `matplotlib`, `tensorflow`

### **Training the Model**

1. Clone this repository.
2. Ensure the dataset is in the specified directory structure.
3. Run the training script:
    ```bash
    python train_model.py
    ```
4. The trained model will be saved as `best_pixelated_image_detector.keras`.

### **Making Predictions**

You can use the trained model to predict whether an image is pixelated:
```python
import tensorflow as tf

# Load the model
model = tf.keras.models.load_model('best_pixelated_image_detector.keras')

# Predict on a new image
prediction = model.predict(image)
```

### **Adjusting the Decision Threshold**

To reduce false positives, adjust the decision threshold using the precision-recall curve:
```python
# Adjust threshold
y_pred = (y_pred_prob >= best_threshold).astype(int)
```

## **Results**

The model successfully achieved:
- **Test Accuracy:** Over 76% ,with the scope of potential improvement upto 90%.
- **Frame Rate:** Approximately 100 FPS.
- **Optimal Threshold:** Tuned to maximize the F1-score and minimize false positives.

## **Future Work**

- **Optimization:** Further improve model complexity to achieve atleast 60 FPS and try to get more accuracy.

- **Extended Dataset:** Incorporate more diverse images to improve generalization.


