Spam Detection Using Machine Learning

Overview

This project is a Spam Detection System that classifies messages as Spam (1) or Ham (0) using Machine Learning techniques. The dataset contains labeled SMS messages, and the model is trained to differentiate between spam and non-spam texts.

Steps in the Project

Data Cleaning

Remove unnecessary columns

Handle missing values

Remove duplicate entries

Exploratory Data Analysis (EDA)

Visualize data distribution (spam vs. ham)

Check word frequency

Identify common spam words

Text Preprocessing

Convert text to lowercase

Remove special characters

Tokenization

Remove stopwords & punctuation

Stemming

Handling Imbalanced Data

Applied SMOTE (Synthetic Minority Over-sampling Technique) to balance ham & spam classes.

Feature Engineering

Used TF-IDF Vectorization (Term Frequency-Inverse Document Frequency) to convert text into numerical features.

Model Building & Evaluation

Train-Test Split (80% training, 20% testing)

Used Multinomial Naïve Bayes for classification

Evaluated model performance using:

Accuracy

Confusion Matrix

Precision Score

Saving Model & Deployment

Saved trained TF-IDF Vectorizer & Naïve Bayes Model using pickle

Can be used for real-time spam detection
