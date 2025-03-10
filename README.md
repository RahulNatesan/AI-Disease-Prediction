# Disease Prediction Model

A prediction model that uses genetic data for disease classification.

## Objective

The goal of this project is to build a machine learning model that can classify diseases based on genetic data.

## Dataset

The data is extracted from a DNA microarray, which measures the expression levels of a large number of genes simultaneously. Samples in the dataset represent patients. For each patient, 7070 gene expressions (values) are measured in order to classify the patient’s disease into one of the following cases: EPD, JPA, MED, MGL, RHB.

## Data Preprocessing:

The dataset is split into training and testing sets, stored locally as `train_data.csv` and `test_data.csv`. The data is loaded using the Pandas library for preprocessing. Key preprocessing steps include:

- Label encoding for classification.
- Limiting the fold difference between 2 and 16,000.
- Selecting a subset of top gene samples based on absolute T-value. The top gene samples are extracted in sets of 2, 4, 6, 8, 10, 12, 15, 20, 25, and 30.

## Implementation in Python:

The prediction model is developed using multiple machine learning algorithms, including:

- Gaussian Naïve Bayes Classifier
- K-Nearest Neighbors (KNN)
- Extra Tree Classifier
- Neural Network - Multi-Layer Perceptron (MLP)
- Decision Tree Classifier

The aim is to select the best-performing model by tweaking hyperparameters and applying various regularization techniques, thereby improving the model's learning ability.

## Accuracy:

The Extra Tree Classifier demonstrated the highest accuracy in classifying gene samples, achieving approximately **95% accuracy** with an optimal gene subset of **25**. The accuracy is validated using a predefined validation dataset during the train/test phase.

## Visualization of Model Performance and Validation:

![Error Rate Gene Subset](results/error_rate_gene_subsets.JPG?raw=true)

![Error Rate HeatMap](results/error_rate_heatmap.JPG?raw=true)

## Results & Inference:

This project explores and compares various machine learning classifiers for predicting diseases based on gene microarray data. The classifiers are trained on labeled gene samples and tested on an unlabeled test sample. The **Extra Tree Classifier** was identified as the most efficient model. Using this classification model, diseases can be predicted from new genetic samples, enabling efficient patient diagnosis.

## Contributions:

1. <a href= "https://github.com/RahulNatesan">Rahul Natesan</a>
