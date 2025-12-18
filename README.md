# PLACEMENT-PREDICTOR-LOGISTIC-REGRESSION
Project Overview
This project is a comprehensive machine learning application designed to predict whether a student will be placed in a job based on two primary academic and cognitive features: CGPA (Cumulative Grade Point Average) and IQ. The project follows an end-to-end data science pipeline, ranging from data ingestion and cleaning to model training, evaluation, and serialization for deployment.

Technical Stack
Language: Python

Data Manipulation: Pandas, NumPy

Visualization: Matplotlib, Mlxtend

Machine Learning: Scikit-Learn (Logistic Regression, StandardScaler, metrics)

Model Persistence: Pickle

Detailed Pipeline Explanation
1. Data Cleaning and Preprocessing

The initial dataset contained 100 entries and 4 columns.

Feature Selection: The column Unnamed: 0 was identified as an unnecessary index and removed using iloc to focus purely on the predictive features (cgpa, iq) and the target variable (placement).

Feature Extraction: The independent variables (X) were isolated as CGPA and IQ, while the dependent variable (y) was set as the placement status.

2. Exploratory Data Analysis (EDA)

A scatter plot was utilized to visualize the distribution of the data.

The x-axis represents CGPA and the y-axis represents IQ.

Color coding was applied to differentiate between students who were placed (1) and those who were not (0).

The visualization highlights a clear boundary, suggesting that a linear classifier would be effective for this dataset.

3. Model Training

Train-Test Split: The data was partitioned into a 90% training set and a 10% testing set to ensure the model could be validated on unseen data.

Feature Scaling: Because CGPA (typically 0–10) and IQ (typically 80–140+) exist on different scales, StandardScaler was applied. This normalizes the data, preventing the model from being biased toward the feature with larger numerical values.

Algorithm Selection: Logistic Regression was chosen as the classification algorithm. It is a robust and efficient choice for binary classification tasks where a linear decision boundary is expected.

4. Evaluation and Visualization

Accuracy: The model achieved an accuracy score of 0.9 (90%) on the test set.

Decision Regions: Using the mlxtend library, a decision region plot was generated to show the linear boundary created by the Logistic Regression model, effectively dividing the "Placed" and "Not Placed" categories in the feature space.

5. Model Deployment Preparation

To make the model usable in real-world applications, it was serialized:

Pickle Export: The trained classifier (clf) was saved as model.pkl. This file can be loaded into a web framework (like Flask or Streamlit) to provide real-time predictions without needing to re-run the training notebook.

How to Use
Dependencies: Install the required libraries via pip: pip install pandas numpy scikit-learn matplotlib mlxtend.

Dataset: Ensure placement.csv is in the working directory.

Run: Execute the cells in the Jupyter Notebook to preprocess data, train the model, and generate the model.pkl file.

Inference: Use pickle.load(open('model.pkl', 'rb')) in your deployment script to predict outcomes for new student data.
