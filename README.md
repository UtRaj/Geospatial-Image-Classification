# Geospatial Image Classification

This project demonstrates a deep learning model built using Keras and TensorFlow to classify satellite images into different categories, such as Cloudy, Desert, Green Area, and Water.

## What It Does

This project builds a Convolutional Neural Network (CNN) model to classify satellite images into four categories. The project leverages data augmentation techniques and image pre-processing to improve model performance.

## How It Works

1. **Data Preparation:**

The data can be obtained from [Kaggle](https://www.kaggle.com/datasets/mahmoudreda55/satellite-image-classification/data), where it's organized into folders representing different categories (e.g., Cloudy, Desert, Green Area, Water).


2. **Data Preprocessing:**

Images are resized and preprocessed using techniques like rescaling, shear, zoom, flip, and rotation to enhance the model's ability to generalize.


3. **Model Architecture:**

    3.1 A deep learning model is built using Convolutional Neural Networks (CNNs) for feature extraction.
 
   3.2 The model architecture includes multiple Conv2D layers with activation functions like ReLU, MaxPooling layers for spatial downsampling, a Flatten layer to convert the 2D feature maps into a 1D vector, and Dense layers for classification.
  
   3.3 Dropout is used to prevent overfitting by randomly disabling neurons during training.

4. **Training and Evaluation:**

   4.1 The model is trained using the prepared dataset with a specified number of epochs.

   4.2  Training and validation metrics such as loss and accuracy are monitored and visualized using matplotlib.


5. **Model Deployment:**

   5.1 After training, the model is saved as a `.h5` file for future use.


6. **Prediction:**

    6.1 The trained model is used to make predictions on new satellite images.

    6.2 Users can change the image path in `run_model.py` to any satellite image of their choice.



## Tools and Methods Used

1. **Python Libraries:** pandas, os, numpy, sklearn, keras, PIL, matplotlib
 
2. **Data Augmentation:** ImageDataGenerator for generating augmented images during training.

3. **Model Building:** Sequential model with Conv2D, MaxPooling2D, Flatten, Dense and Dropout layers.

4. **Model Evaluation:** Metrics such as accuracy and loss are used.

5. **Visualization:** Matplotlib is used to visualize training and validation metrics.


## Usage

1. Clone the repository to your local machine.
   
2. Install the required libraries.
   
3. Obtain the dataset from [Kaggle](https://www.kaggle.com/datasets/mahmoudreda55/satellite-image-classification/data) and organize it into folders as per the provided structure.
   
4. Run `create_model.py` to train the model and save it as `Model.h5`.
   
5. Use `run_model.py` to load the trained model and make predictions on new satellite images by changing the image path to your desired satellite image.


