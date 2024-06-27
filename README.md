# Bird Sound Classification with YAMNet and CNN

## Overview

This project focuses on classifying bird sounds using a combination of YAMNet for feature extraction and a Convolutional Neural Network (CNN) for classification. The model is trained on a custom dataset of bird sounds, and aims to accurately identify different bird species based on their audio recordings.

## Getting Started

### Prerequisites

Ensure you have the following installed:

- Python 3.x
- TensorFlow
- TensorFlow Hub
- TensorFlow IO
- Keras
- librosa
- numpy
- pandas
- matplotlib
- scikit-learn

### Project Structure

- `Model_builder.ipynb`: Jupyter Notebook containing the complete code for data preprocessing, model training, and evaluation.
- `birds_audio_model.h5`: Trained model file.

### Dataset

The dataset includes audio recordings of bird calls, along with metadata providing details such as species name, file name, and data split (train/test). The audio files are organized in a directory structure based on their respective bird species.

## Running the Project

### Step 1: Data Preparation

- Load the dataset and filter it to include only the desired bird species.
- Preprocess the audio files to ensure consistent sampling rates and formats.
- Split the data into training, validation, and test sets.

### Step 2: Feature Extraction with YAMNet

- Use the YAMNet model to extract audio embeddings from the preprocessed audio files.
- Unbatch and cache the datasets for efficient processing.

### Step 3: Model Training

- Define a Sequential model with multiple Dense layers, Batch Normalization, and Dropout for regularization.
- Compile the model with a sparse categorical crossentropy loss function and the Adam optimizer.
- Train the model using the training dataset and validate it using the validation dataset.
- Use callbacks such as ReduceLROnPlateau and EarlyStopping to optimize the training process.

### Step 4: Model Evaluation

- Evaluate the trained model on the test dataset to assess its performance.
- Save the trained model to an H5 file for future use.

### Running the Code

To run the project, execute the following command in your terminal or Jupyter Notebook:

```bash
jupyter notebook Model_builder.ipynb
```

Follow the instructions within the notebook to execute each cell and complete the data preparation, feature extraction, model training, and evaluation steps.

## Results

The trained model achieves a specific accuracy on the test dataset, indicating its ability to correctly classify bird species based on their audio recordings. Detailed performance metrics and visualizations are provided within the Jupyter Notebook and the accompanying report.

## Conclusion

This project demonstrates a comprehensive approach to bird sound classification using deep learning techniques. By leveraging the power of YAMNet for feature extraction and a custom CNN for classification, the model can accurately identify different bird species, contributing to the field of bioacoustics and wildlife monitoring.
