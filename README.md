# Loan Eligibility Prediction

## Overview

This project aims to predict loan eligibility based on various features using machine learning techniques. The dataset used for this project is sourced from Kaggle, which contains information about applicants and their loan status. The goal is to build a predictive model that can help financial institutions assess the eligibility of loan applicants efficiently.

## Dataset

The dataset used in this project can be found on Kaggle at the following link: [Loan Prediction Dataset](https://www.kaggle.com/datasets/ninzaami/loan-predication).

## Libraries Used

The following libraries were utilized in this project:

- **NumPy**: For numerical operations and handling arrays.
- **Pandas**: For data manipulation and analysis.
- **Seaborn**: For data visualization.
- **Scikit-learn**: For building and evaluating the machine learning model.
- **Warnings**: To manage warning messages during execution.

## Steps Involved

1. **Data Loading**: The dataset was loaded using Pandas for analysis.
2. **Data Preprocessing**: 
   - Handled missing values and performed necessary data cleaning.
   - Converted categorical variables into numerical format for model compatibility.
3. **Data Visualization**: 
   - Used Seaborn to visualize the data and understand the relationships between features.
4. **Model Training**: 
   - Split the dataset into training and testing sets using `train_test_split`.
   - Trained a Support Vector Machine (SVM) model to predict loan eligibility.
5. **Model Evaluation**: 
   - Evaluated the model's performance using accuracy score and other relevant metrics.

## Conclusion
The loan eligibility prediction model developed in this project provides a systematic approach to assess loan applications. The use of machine learning techniques can significantly enhance the decision-making process for financial institutions, leading to more efficient and accurate evaluations.

## cAcknowledgments
- Kaggle for providing the dataset.
- The authors of the libraries and all other resources used in this project.
