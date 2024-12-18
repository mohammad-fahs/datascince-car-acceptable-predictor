# Data science project

# Introduction to the project

Cars are a ubiquitous part of daily life, and making informed decisions about their acceptability is crucial for consumers, manufacturers, and policymakers. The **Car Evaluation Database** provides a structured dataset to analyze and predict car acceptability based on specific attributes like price, safety, and capacity. Derived from a hierarchical decision model, the dataset has been widely used for machine learning and decision analysis research, making it an excellent resource for exploring predictive modeling, clustering, and data visualization techniques.

The objective of this project is to utilize this dataset to solve practical and analytical problems in car evaluation. Specifically, we aim to predict car acceptability, and provide meaningful insights using visualizations. These tasks will help improve our understanding of how various car attributes influence consumer decision-making and acceptance.

## Understanding the problem

Car acceptability depends on several factors, including price, maintenance costs, capacity, safety, and comfort. However, manually analyzing these attributes for decision-making is challenging due to the complexity and variability of the data. This project aims to address the following key problems:

1. **Predictive Modeling**: How can we accurately predict a car's acceptability based on its features? Developing a car acceptability predictor using machine learning models will provide insights into how attributes like price or safety contribute to the overall decision.
2. **Data Visualization**: How do specific attributes like luggage space or maintenance costs affect a car's acceptability? Creating dashboards or visualizations will provide an intuitive way to analyze and interpret the impact of individual attributes on car evaluation.

## Problem description

This project focuses on leveraging data science techniques to solve real-world problems in car evaluation:

1. Car Acceptable Predictor: Use different machine learning models  to predict the car acceptability category (`unacc`, `acc`, `good`, or `v-good`). compare the models accuracy and Explore feature importance to understand which attributes most influence acceptability.
2. Create visualizations (e.g., heatmaps, bar charts, scatter plots) to analyze attribute distributions and their relationship with acceptability.  

# Car Acceptable Predictor:

The **Car Acceptability Predictor** is a machine learning task focused on determining the acceptability of cars based on six key attributes: `buying`, `maint`, `doors`, `persons`, `lug_boot`, and `safety`. The goal is to classify cars into one of four categories (`unacc`, `acc`, `good`, `v-good`) using predictive models. This project leverages supervised learning techniques to analyze and interpret patterns within the data, offering insights into what makes a car acceptable to users.

This project is critical for applications in automotive industry decision-making processes, such as evaluating consumer preferences, optimizing product offerings, and enhancing user satisfaction. By implementing and comparing various machine learning models, we aim to identify the most effective algorithm for accurately predicting car acceptability and gaining a deeper understanding of the factors influencing it.

### Objectives

- **Predictive Modeling**: Train and test machine learning models to classify car acceptability.
- **Feature Analysis**: Evaluate the importance of features in determining car acceptability.
- **Model Evaluation**: Assess the performance of different algorithms using metrics like accuracy, confusion matrix, precision, recall, and F1-score.

### Methodology

We will employ four machine learning algorithms for this task:

1. **Logistic Regression**: A baseline model suitable for multi-class classification.
2. **Decision Tree Classifier**: Captures non-linear relationships and feature interactions.
3. **Random Forest Classifier**: An ensemble method to enhance accuracy and reduce overfitting.
4. **XGBoost Classifier**: A gradient boosting approach known for its robust performance on classification tasks.

### Evaluation Metrics

To ensure robust model evaluation, the following metrics will be used:

- **Accuracy**: Measures overall correctness of predictions.
- **Confusion Matrix**: Visualizes prediction performance across classes.
- **Precision, Recall, and F1-Score**: Assess performance on imbalanced datasets, highlighting model reliability for each class.

## Data set Observations:

- **Structure**: The dataset has 7 columns: `buying`, `maint`, `doors`, `persons`, `lug_boot`, `safety`, and the target column `class`.
- **Data Types**: All columns are categorical (object type).
- **No Missing Values**: All 1728 rows are complete.
- **Class Distribution**:
    - `unacc`: Majority class (70% of the dataset).
    - Other classes (`acc`, `good`, `v-good`) form the minority.

# Dealing with Dataset:

### Dealing with missing information

No missing values were found in the dataset (`info()` revealed all columns had complete data for all 1728 rows). Therefore, no imputation was necessary. If missing values existed, techniques like imputation (using the mean, median, or mode for numerical data and most frequent for categorical data) or removal of rows/columns with too many missing values would be applied.

![image](https://github.com/user-attachments/assets/012763f1-e05d-446a-88ed-8c2903d70b8f)


### Adding new columns

No new columns were added to the dataset because the six attributes were sufficient for predicting car acceptability based on the given model requirements.  Derived features could include **price category indices** (e.g., combining `buying` and `maint`) or **comfort scores** (e.g., combining `doors`, `persons`, and `lug_boot`). But a feature engineering step could explore interactions between features for improved model performance.

### Feature selection

**All Features Included**: The dataset has only six features (`buying`, `maint`, `doors`, `persons`, `lug_boot`, `safety`), and all were used. These features directly influence car acceptability as per the original decision model, and no redundant or irrelevant features were identified.

### Transformations done

- **Encoding Categorical Data**:
    - All categorical features were converted to numerical values using `LabelEncoder`. This ensures the data can be processed by logistic regression, which requires numerical input.

### Remarkable data

- **Imbalanced Class Distribution**:
    - The target variable (`class`) had a heavily imbalanced distribution, with 70% of samples belonging to the `unacc` class.
- **Classes with Few Samples**:
    - Classes like `good` and `v-good` had very few samples in the original dataset. This would lead to poor performance for these categories without oversampling.

## 1- Logistic regression:

Logistic regression is a supervised machine learning algorithm primarily used for binary classification problems, but it can be extended to handle multiclass classification tasks using techniques such as one-vs-rest (OvR) or multinomial logistic regression. Unlike linear regression, which predicts continuous values, logistic regression predicts probabilities that map to discrete classes. This is achieved by applying the logistic (sigmoid) function to the linear combination of input features, which ensures the output is bounded between 0 and 1.

### Why Use Logistic Regression for the Car Acceptability Dataset?

1. **Multiclass Classification**: The car acceptability dataset has four classes (`unacc`, `acc`, `good`, `v-good`). Logistic regression is versatile and can effectively handle multiclass classification using appropriate settings.
2. **Interpretable Model**: Logistic regression provides insights into feature importance, making it easier to understand how each attribute contributes to the car acceptability prediction.
3. **Efficient for Small Datasets**: With only 1728 instances and six attributes, logistic regression is computationally efficient and well-suited for this task.

### Code Snippets

Imports 

```python
# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
from imblearn.over_sampling import SMOTE
```

Data set Loading 

```python
# Step 1: Load the Dataset
file_path = 'car.csv'
car_data = pd.read_csv(file_path)
```

Initialize and encode of object-type columns 

```python
# Initialize LabelEncoder
label_encoders = {}

# Encode all object-type columns
for column in car_data.select_dtypes(include='object').columns:
    le = LabelEncoder()
    car_data[column] = le.fit_transform(car_data[column])
    label_encoders[column] = le  # Store the encoder for future decoding

print(car_data.head())
```

Splitting the dataset 

```python
# Features and target
X = car_data.drop(columns=['class'])
y = car_data['class']

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"Training set size: {X_train.shape}")
print(f"Test set size: {X_test.shape}")
```

Initialize and fit the model

```python
# Initialize the model
log_reg = LogisticRegression(max_iter=1000, random_state=42)

# Fit the model
log_reg.fit(X_train, y_train)

print("Model training complete.")

# Predictions
y_pred = log_reg.predict(X_test)

```

Classification report and Accuracy 

```python
# Classification report
print("Classification Report:")
print(classification_report(y_test, y_pred))

# Accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")
```

### Model Accuracy

![image 1](https://github.com/user-attachments/assets/a25c06c4-3bda-4a97-8746-50014f549e2b)


This model achieved an overall accuracy of **66%**, but its performance was inconsistent across classes. While it excelled in predicting class `2 (good)` with a high recall of **92%** and an F1-score of **81%**, it struggled significantly with the other classes, especially `1 (acc)` and `3 (v-good)`, where the recall was **0%**, indicating it failed to identify any instances of these classes. The imbalance in class performance suggests that logistic regression may not be well-suited for this multi-class problem without additional preprocessing or hyperparameter tuning.

![image 2](https://github.com/user-attachments/assets/eb1dff71-b8fb-402c-a46b-85bed80b901e)

- **Class 0 (`unacc`)**: The logistic regression model misclassified most of the instances in this class, predicting **69** as class `2 (good)` and **12** as class `3 (v-good)`. This explains the poor precision (0.26) and recall (0.13) for this class, indicating a significant misclassification. It could be due to a lack of strong feature separation between `unacc` and other categories.
- **Class 1 (`acc`)**: The model incorrectly classified **9** instances of class `1` as class `2 (good)`. This shows the model's struggle with identifying this class, as reflected by the **0% recall** for class `1` in the classification report.
- **Class 2 (`good`)**: Class `2` was the best predicted, with **217** correct predictions and only **17** misclassifications to class `0` (`unacc`). The model performed well here, aligning with its high recall of **92%** and F1-score of **81%** for this class.
- **Class 3 (`v-good`)**: Similar to class `1`, class `3` is poorly handled, with **5** misclassifications to class `2 (good)` and **12** instances misclassified to class `0`. This confirms the poor recall and precision of **0%** for class `3` in the classification report.

### Feature Importance

```bash
# Get feature coefficients
coefficients = log_reg.coef_[0]

# Create a DataFrame of features and their coefficients
feature_importance = pd.DataFrame({
    'Feature': X.columns,
    'Coefficient': coefficients
})

# Sort by absolute coefficient values
top_features = feature_importance.reindex(feature_importance['Coefficient'].abs().sort_values(ascending=False).index)

# Display top features
print("Top Features based on Logistic Regression Coefficients:")
print(top_features.head(10))

```

![image 3](https://github.com/user-attachments/assets/07b6b83e-ca36-4945-aea2-fbbdbc01baf2)

- **Safety** is the most influential feature, with a significantly positive coefficient, suggesting that higher safety values strongly impact positive classification outcomes.
- **Persons** and **Lug_boot** also positively affect predictions, indicating that the number of persons the car can hold and luggage boot size are important considerations.
- **Maint (Maintenance cost)** and **Buying** show smaller coefficients, suggesting they are less influential compared to the other features.
- **Doors** is the least impactful feature, with a near-zero positive coefficient.

## 2- Decision Tree Classifier

Decision Trees are supervised machine learning models used for both classification and regression tasks. They split the data into subsets based on feature values, forming a tree-like structure with decision nodes and leaf nodes. The simplicity and interpretability of decision trees make them a popular choice for structured data problems. 

### Why Use a Decision Tree for the Car Acceptability Dataset?

1. **Handles Multiclass Classification**: Decision trees can natively support multiclass classification, making them an appropriate choice for the four car acceptability classes (`unacc`, `acc`, `good`, `v-good`).
2. **Interpretable and Visualizable**: The model provides a clear decision-making process, making it easy to visualize and understand the factors leading to specific predictions.
3. **Robust to Nonlinear Relationships**: Decision trees do not assume linearity in the data, enabling them to capture complex interactions among the attributes in the car acceptability dataset.

### Code Snippets

Imports 

```python
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import numpy as np
```

Data set Loading 

```python
# Step 1: Load the Dataset
file_path = 'car.csv'
car_data = pd.read_csv(file_path)
```

Initialize and encode of object-type columns 

```python
# Initialize LabelEncoder
label_encoders = {}

# Encode all object-type columns
for column in car_data.select_dtypes(include='object').columns:
    le = LabelEncoder()
    car_data[column] = le.fit_transform(car_data[column])
    label_encoders[column] = le  # Store the encoder for future decoding

print(car_data.head())
```

Splitting the dataset 

```python
# Split data
X = df.drop(columns=['class'])
y = df['class']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```

Initialize and fit the model

```python
# Train Decision Tree Classifier model
dt_classifier = DecisionTreeClassifier(random_state=42)
dt_classifier.fit(X_train, y_train)
print("Model training complete.")
```

Classification report and Accuracy 

```python
# Evaluate the model
y_pred = dt_classifier.predict(X_test)
print("Classification Report:")
print(classification_report(y_test, y_pred))
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")
```

### Model Accuracy

![image 4](https://github.com/user-attachments/assets/27176bb7-60a3-4541-89bd-20d105811104)

The decision tree model performed exceptionally well, achieving an overall accuracy of **97%**. It delivered consistently high precision, recall, and F1-scores across all classes, particularly excelling in class `2 (good)` with perfect scores and performing well on the less frequent classes (`1` and `3`) with recall values of **91%** and **94%**, respectively. Its ability to capture patterns in all classes suggests it effectively handled the hierarchical or feature-based splits in the data.

![image 5](https://github.com/user-attachments/assets/1d4843c1-810f-4273-8170-e0e3c984a6bb)

- **Class 0 (`unacc`)**: The decision tree model correctly classified **76** instances of class `0` as `unacc`, with only **6** misclassified to class `1` (`acc`). This high precision and recall for class `0` align with the **0.97 precision** and **0.92 recall** in the classification report.
- **Class 1 (`acc`)**: The decision tree showed impressive performance here as well, with **10** correct predictions for class `1` and only **1** misclassification to class `0`. The model’s **91% recall** for class `1` reflects its ability to identify `acc` cases accurately.
- **Class 2 (`good`)**: This was the strongest class for the decision tree model, with **235** correct predictions and no misclassifications, aligning perfectly with the **100% recall** and **1.00 F1-score** for this class.
- **Class 3 (`v-good`)**: Class `3` also performed well with **16** correct predictions, though there were **1** misclassification to class `0`. This explains the **94% recall** for class `3`.

### Feature Importance

```python
# Get feature importance
feature_importance = pd.DataFrame({
    'Feature': X.columns,
    'Importance': dt_classifier.feature_importances_
})

# Sort by importance values
top_features = feature_importance.reindex(feature_importance['Importance'].sort_values(ascending=False).index)

# Display top features
print("Top Features based on Decision Tree Feature Importance:")
print(top_features.head(10))
```

![image 6](https://github.com/user-attachments/assets/21d964a5-5737-4404-b626-29c985aabc73)

- **Safety** and **Maint** are almost equally important, highlighting the decision tree's ability to split on these features frequently during classification.
- **Persons** contributes significantly, likely due to its direct relevance to car acceptability.
- **Buying** and **Lug_boot** play secondary roles but are still notable.
- **Doors** has the least influence, consistent with logistic regression.

## **3- Random Forest Classifier**

### Why Use **Random Forest Classifier** for the Car Acceptability Dataset?

Random Forest is an ensemble learning technique that combines multiple decision trees to improve predictive accuracy and control overfitting. Each tree in the forest is trained on a random subset of the data and features, and predictions are made by aggregating the outputs of individual trees.

### Why Use Random Forest for the Car Acceptability Dataset?

1. **Improved Accuracy**: By aggregating multiple decision trees, Random Forest reduces overfitting and provides more robust predictions for multiclass classification tasks.
2. **Handles Feature Importance**: Random Forest provides insights into feature importance across the dataset, offering a deeper understanding of the significant attributes influencing car acceptability.
3. **Scalability**: While the dataset is small, Random Forest is scalable to larger datasets, making it a versatile choice for similar problems with larger data.

### Code Snippets

Imports 

```python
# Import necessary libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score
```

Data set Loading 

```python
# Step 1: Load the Dataset
file_path = 'car.csv'
car_data = pd.read_csv(file_path)
```

Initialize and encode of object-type columns 

```python
# Initialize LabelEncoder
label_encoders = {}

# Encode all object-type columns
for column in car_data.select_dtypes(include='object').columns:
    le = LabelEncoder()
    car_data[column] = le.fit_transform(car_data[column])
    label_encoders[column] = le  # Store the encoder for future decoding

print(car_data.head())
```

Splitting the dataset 

```python
# Split data
X = df.drop(columns=['class'])
y = df['class']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

```

Initialize and fit the model

```python
# Train Random Forest Classifier model
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)  # Updated model
rf_classifier.fit(X_train, y_train)
print("Model training complete.")
```

Classification report and Accuracy 

```python
# Evaluate the model
y_pred = rf_classifier.predict(X_test)
print("Classification Report:")
print(classification_report(y_test, y_pred))
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")
```

### Model Accuracy

![image 7](https://github.com/user-attachments/assets/dc689664-0416-4df1-aa47-fe73314da3a3)

Similar to the decision tree, the random forest model also achieved **97% accuracy** and demonstrated strong performance across all metrics. It showed slight improvement in handling class `1 (acc)` with a perfect recall of **100%** and a balanced F1-score of **79%**. The model's ensemble nature likely contributed to its robustness and ability to generalize better across all classes, including the less frequent ones.

![image 8](https://github.com/user-attachments/assets/af6b2d74-5c1e-48dc-b784-fd0807c442b0)


- **Class 0 (`unacc`)**: The random forest model predicted **74** instances of class `0` correctly but misclassified **6** instances to class `1` and **3** to class `2`. These errors explain the slight decrease in recall for this class when compared to the decision tree. Still, the model maintained strong precision and recall in this class.
- **Class 1 (`acc`)**: The model performed excellently here, with perfect recall (**100%**) and only **0** misclassifications. This aligns with its **1.00 recall** for class `1` in the classification report.
- **Class 2 (`good`)**: Like the decision tree, the random forest model performed perfectly for class `2`, with **235** correct predictions and no misclassifications. This strong performance corresponds with the **100% recall** and **99% precision** reported earlier.
- **Class 3 (`v-good`)**: The random forest model performed similarly to the decision tree on this class, with **16** correct predictions and a single misclassification to class `0`, reflecting the **94% recall** in the classification report.

### Feature Importance

```python
# Get feature importance
feature_importance = pd.DataFrame({
    'Feature': X.columns,
    'Importance': rf_classifier.feature_importances_  # Updated to RandomForest feature importances
})

# Sort by importance values
top_features = feature_importance.reindex(feature_importance['Importance'].sort_values(ascending=False).index)

# Display top features
print("Top Features based on Random Forest Feature Importance:")
print(top_features.head(10))
```

![image 9](https://github.com/user-attachments/assets/a4e139c9-1474-48ec-9edc-4a46c5bf2d89)

- **Safety** remains the dominant feature, showing stability across algorithms.
- **Persons** is the second most important feature, reinforcing its role in improving predictions.
- Unlike the decision tree, **Buying** gains higher importance here, suggesting that averaging across multiple trees amplifies its relevance.
- **Maint** is less important than in the decision tree, while **Lug_boot** and **Doors** maintain their lower significance.

## **4- XGBoost Classifier**

### Why Use **XGBoost Classifier** for the Car Acceptability Dataset?

Extreme Gradient Boosting (XGBoost) is a high-performance, gradient-boosting algorithm designed for both classification and regression tasks. It uses an ensemble of decision trees trained in sequence, with each tree correcting the errors of the previous one. XGBoost is known for its speed and predictive power.

### Why Use XGBoost for the Car Acceptability Dataset?

1. **Superior Performance**: XGBoost excels in handling structured data and often outperforms other models in terms of accuracy and generalization on small to medium-sized datasets like the car acceptability dataset.
2. **Regularization**: Built-in regularization techniques prevent overfitting, making it robust for predicting multiclass outcomes.
3. **Feature Interaction**: XGBoost captures complex feature interactions more effectively than standalone decision trees, which is beneficial for understanding the interplay of attributes in determining car acceptability.

### Code Snippets

Imports 

```python
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier
```

Data set Loading 

```python
# Step 1: Load the Dataset
file_path = 'car.csv'
car_data = pd.read_csv(file_path)
```

Initialize and encode of object-type columns 

```python
# Initialize LabelEncoder
label_encoders = {}

# Encode all object-type columns
for column in car_data.select_dtypes(include='object').columns:
    le = LabelEncoder()
    car_data[column] = le.fit_transform(car_data[column])
    label_encoders[column] = le  # Store the encoder for future decoding

print(car_data.head())
```

Splitting the dataset 

```python
# Split data
X = df.drop(columns=['class'])
y = df['class']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```

Initialize and fit the model

```python
# Train XGBoost Classifier model
xgb_classifier = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
xgb_classifier.fit(X_train, y_train)
print("Model training complete.")
```

Classification report and Accuracy 

```python
# Evaluate the model
y_pred = xgb_classifier.predict(X_test)
print("Classification Report:")
print(classification_report(y_test, y_pred))
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")
```

### Model Accuracy

![image 10](https://github.com/user-attachments/assets/48efdf02-f863-4c5b-bb7d-5f206c8fd6b5)

The XGBoost classifier achieved the best overall accuracy of **98%**, with near-perfect performance in most classes. It maintained precision and recall values close to **100%** for the dominant class `2 (good)` and strong performance for classes `0 (unacc)` and `3 (v-good)`. However, while it perfectly identified class `1 (acc)` instances with a recall of **100%**, its F1-score of **81%** indicates room for slight improvement in balancing precision and recall for this minority class.

![image 11](https://github.com/user-attachments/assets/00b199fb-ff8b-4be1-8e6a-0c13e4394c6c)

- **Class 0 (`unacc`)**: The XGBoost model showed very good performance for class `0`, correctly classifying **79** instances and misclassifying only **4** to class `1`. This accounts for the high precision and recall (0.99 and 0.95, respectively) for class `0` in the classification report.
- **Class 1 (`acc`)**: This class was handled well, with **11** correct predictions and no misclassifications to other classes. The model’s **100% recall** for class `1` indicates it identified all instances, but the **81% F1-score** suggests slight imbalances in precision.
- **Class 2 (`good`)**: Class `2` was perfectly predicted by the XGBoost model, with **235** correct predictions and no errors, reflecting its **100% recall** and **1.00 precision**.
- **Class 3 (`v-good`)**: Similar to the decision tree and random forest, XGBoost predicted **15** instances of class `3` correctly but misclassified **1** to class `0` and **1** to class `1`. This results in a slightly lower recall of **88%**, but still strong overall performance for this minority class.

### Feature Importance

```python
# Get feature importance
feature_importance = pd.DataFrame({
    'Feature': X.columns,
    'Importance': xgb_classifier.feature_importances_
})

# Sort by importance values
top_features = feature_importance.sort_values(by='Importance', ascending=False)

# Display top features
print("Top Features based on XGBoost Feature Importance:")
print(top_features.head(10))
```

![image 12](https://github.com/user-attachments/assets/908dd72e-3c9d-4dca-a8cb-74736db88dd2)

- **Safety** is even more dominant here than in other algorithms, underlining its critical role in accurate classification.
- **Persons** remains the second most important, indicating consistency across models.
- **Maint** overtakes **Buying**, showing that XGBoost identifies it as more relevant than random forests do.
- **Lug_boot** and **Doors** are again the least important, consistent across all models.

# Algorithm Evaluation

### According to classification report

- Logistic regression struggled due to the multi-class nature and class imbalance of the dataset.
- Decision tree and random forest models performed comparably well, with random forest slightly better in handling minority class predictions.
- XGBoost provided the best overall results, combining high accuracy with strong generalization across all classes. This makes it the most effective model for this task, particularly for datasets with complex relationships and imbalanced distributions.

### According to Confusion metrics

- **Logistic Regression**: Struggled with imbalanced classes, particularly class `1 (acc)` and class `3 (v-good)`, as indicated by its poor precision and recall values, especially for these classes. The confusion matrix highlights these difficulties with misclassifications to other classes.
- **Decision Tree**: Demonstrated excellent overall performance, with high precision and recall across all classes. The confusion matrix confirms its effectiveness, especially for class `2 (good)`, where it had perfect predictions.
- **Random Forest**: Performed similarly to the decision tree but with slightly better handling of the minority class `1 (acc)`. Its confusion matrix shows minimal misclassifications, especially for class `2 (good)`.
- **XGBoost**: Achieved the best overall performance, with strong precision and recall, especially for class `2 (good)`. The confusion matrix shows that it handled minority classes well, despite minor misclassifications for class `1 (acc)` and `3 (v-good)`.

# Visualizations

## Pie Chart for Target Class Proportions

```python
t_col = 'class'
class_counts = df[target_col].value_counts()
plt.figure(figsize=(8, 8))
plt.pie(class_counts, labels=class_counts.index, autopct='%1.1f%%', startangle=90, colors=sns.color_palette("viridis", len(class_counts)))
plt.title("Class Proportions", fontsize=16)
plt.show()
```

![image 13](https://github.com/user-attachments/assets/828dc288-4a6b-4870-aebb-7d6dd54d62c8)

The pie chart visualizes the distribution of classes in the dataset:

- The largest slice (70%) represents the `unacc` class, which indicates a high frequency of examples that are classified as `unacc` (unacceptable).
- The next largest slice (22.2%) represents the `acc` class, indicating that a significant portion of the dataset falls into the `acceptable` category.
- Smaller portions of the pie (4.0% and 3.8%) represent the `good` and `vgood` classes, respectively, showing that fewer examples are classified as `good` or `very good` compared to `unacc` and `acc`.

This imbalance in class distribution may indicate that the dataset is heavily skewed towards the `unacc` class, which could affect the performance of machine learning models if not addressed (e.g., through techniques like class weighting, resampling, or synthetic data generation).

## Grid to compare feature distributions by class

### buying

The plot displays a set of bar charts, each representing the distribution of the `buying` feature within one of the four target classes (`unacc`, `acc`, `vgood`, `good`):

1. **`unacc` Class**:
    - The distribution of the `buying` feature in the `unacc` class shows a clear preference for `vhigh` and `high`, which indicates that cars in this category tend to have higher prices.
2. **`acc` Class**:
    - The `acc` class has a more balanced distribution across all levels of `buying`, but `high` and `vhigh` are still somewhat more common than the others, suggesting that acceptable cars also tend to have higher prices but with more variety.
3. **`vgood` Class**:
    - For the `vgood` class, the distribution is heavily skewed towards the `high` and `vhigh` categories. This suggests that cars in the `vgood` class are typically those that are more expensive, which aligns with the expectation that better-rated cars are higher priced.
4. **`good` Class**:
    - The `good` class shows a similar but less extreme trend, with a noticeable preference for `high` and `vhigh`, indicating that good cars also tend to be in the higher price range, but this class has a more balanced distribution compared to `vgood`.

![image 14](https://github.com/user-attachments/assets/5b083768-7daf-414a-babb-83516615567d)

### Maintenance

The plot displays a set of bar charts, each representing the distribution of the `Maintenance` feature within one of the four target classes (`unacc`, `acc`, `vgood`, `good`):

- **`unacc` Class**:
    - The `unacc` class has the largest counts across all `maint` categories.
    - Higher maintenance categories (`vhigh`, `high`) dominate, suggesting that a car classified as `unacc` often has higher maintenance requirements.
    - This indicates that higher maintenance may negatively affect the car's acceptability.
- **`acc` Class**:
    - The distribution is more balanced compared to the `unacc` class, with `med` and `high` maintenance categories being more prevalent.
    - Lower maintenance (`low`) is also relatively common, indicating that cars classified as `acc` span a wider range of maintenance requirements.
- **`vgood` Class**:
    - The `vgood` class has a very small count overall, and the majority of cars in this category have lower maintenance levels (`low`).
    - This implies that low-maintenance cars are more likely to be classified as `vgood`.
- **`good` Class**:
    - Similar to the `vgood` class, the `good` class has smaller counts overall, with a skew toward lower maintenance categories (`low` and `med`).
    - The trend suggests that cars classified as `good` typically require lower maintenance.

![image 15](https://github.com/user-attachments/assets/475c3474-09fa-47d2-a68f-cf2406e16be9)

### Safety

The plot displays a set of bar charts, each representing the distribution of the `safety` feature within one of the four target classes (`unacc`, `acc`, `vgood`, `good`):

- **`unacc` Class**:
    - The `unacc` class has the largest count overall, particularly for the `low` safety category.
    - There is a significant drop in count as safety improves (`med`, `high`), suggesting that cars with lower safety features are more likely to be classified as `unacc`.
- **`acc` Class**:
    - The `acc` class shows a more balanced distribution across the `med` and `high` safety categories, with very few cars in the `low` safety category.
    - This indicates that medium and high safety levels are more associated with cars classified as `acc`.
- **`vgood` Class**:
    - The `vgood` class has the smallest count but exclusively appears in the `high` safety category.
    - This suggests that only cars with the highest safety features are classified as `vgood`.
- **`good` Class**:
    - Similar to the `vgood` class, the `good` class predominantly appears in the `high` safety category, though with fewer examples.
    - There are also a few cars in the `med` safety category, but none in the `low` safety category.

![image 16](https://github.com/user-attachments/assets/3ee59bb9-5f2b-44d5-8d47-0c5669007034)


## Heatmap Visualizations

![image 17](https://github.com/user-attachments/assets/1705d6b5-e1ca-4eb0-b301-2e53d4bcef79)


1. **`buying` vs Class**:
    - High buying costs (`vhigh` and `high`) are predominantly associated with the `unacc` class.
    - Low buying costs (`low`) have a more balanced distribution, with a higher association with `acc` and `good` classes, indicating affordability contributes to better acceptability.
2. **`maint` vs Class**:
    - Similar to `buying`, high maintenance costs (`vhigh` and `high`) are strongly associated with the `unacc` class.
    - Lower maintenance costs (`low`) are linked with better class acceptability, especially `vgood`, showing that affordable maintenance improves a car's classification.
3. **`doors` vs Class**:
    - The number of doors does not show a strong correlation with higher classes (`good` or `vgood`), but cars with `2` doors are primarily classified as `unacc`.
    - Cars with `4` or `5more` doors show a relatively more balanced distribution, favoring `acc` and `good` classes, suggesting practicality matters.
4. **`persons` vs Class**:
    - Cars with a capacity of `2` persons are almost exclusively classified as `unacc`, indicating low capacity reduces acceptability.
    - Cars with capacities of `4` and `more` are strongly associated with `vgood` and `good` classes, demonstrating the importance of higher capacity for higher acceptability.
5. **`lug_boot` vs Class**:
    - Cars with a `small` luggage boot are more associated with the `unacc` class.
    - Larger luggage capacities (`big` and `med`) have more balanced distributions across `acc`, `good`, and `vgood`, highlighting that larger storage is important for higher classifications.
6. **`safety` vs Class**:
    - Safety is one of the strongest predictors: `low` safety is heavily associated with the `unacc` class.
    - `high` safety is required for a car to achieve a `vgood` classification and is strongly linked to `good` and `acc` as well
    

This visualization highlights that `safety` and `persons` are the most influential factors for determining car acceptability, with higher values leading to better classifications. Economic factors (`buying` and `maint`) also strongly influence acceptability, with lower costs associated with better classes. Features like `doors` and `lug_boot` have a moderate impact, with higher practicality (more doors and larger luggage capacity) favoring better classes. These insights can guide model building and feature selection, emphasizing safety, capacity, and affordability for better prediction accuracy.

which is indicated also in the XGBoost Algorithm feature importance analysis 

![image 18](https://github.com/user-attachments/assets/f388f98b-c068-47a8-bfe2-603a03738abe)


# Findings

### **Key Findings**

1. **Model Accuracy and Performance:**
    - **Logistic Regression** has the lowest accuracy (66%) due to its inability to model complex, non-linear relationships. It struggles with minority classes (`acc` and `v-good`), evident from its confusion matrix where most instances are misclassified.
    - **Decision Tree, Random Forest, and XGBoost** all achieve high accuracy (97-98%), showcasing their ability to capture intricate patterns in the data.
    - **XGBoost** slightly outperforms the other models with 98% accuracy, leveraging its advanced gradient boosting mechanism to optimize performance, especially for minority classes.
2. **Feature Importance Insights:**
    - **Safety** is the most influential feature across all models, aligning with real-world expectations that safety is a critical factor in car acceptability.
    - **Persons (seating capacity)** is consistently the second most important feature, emphasizing its significance in predicting acceptability.
    - **Maint (maintenance cost)** and **Buying (purchase cost)** are more critical for tree-based models, which can better capture non-linear relationships between these features and car acceptability.
    - **Lug_boot (luggage capacity)** and **Doors (number of doors)** have consistently low importance, indicating they play a minor role in determining acceptability.
3. **Confusion Matrices and Feature Utilization:**
    - Logistic regression misclassifies many instances of minority classes (`acc` and `v-good`) as majority classes (`good`), showing its linear nature limits its ability to leverage feature interactions.
    - Tree-based models (especially Random Forest and XGBoost) better distribute focus across classes, correctly identifying minority class instances.
    - XGBoost’s slight edge over Random Forest is due to its ability to fine-tune splits and focus on difficult-to-predict instances, evident from the marginally better precision and recall for minority classes.
    - 
### **Conclusions and Recommendations**

1. **Best Model**:
    - **XGBoost** is the best-performing model for this dataset. Its ability to optimize for both majority and minority classes, combined with its robust handling of feature importance, makes it ideal for predicting car acceptability.
2. **Key Features to Focus On**:
    - Any decision-making process based on this dataset should prioritize **Safety**, **Persons**, and **Maint**, as they are the most significant predictors of car acceptability across all models.
3. **Model Usage**:
    - For real-time applications or scenarios with strict performance requirements, XGBoost is recommended due to its superior balance and accuracy.
    - If interpretability is a priority, **Decision Tree** is a simpler alternative while still achieving high accuracy.
4. **Improvements**:
    - Addressing the dataset imbalance (e.g., oversampling minority classes or applying SMOTE) could further enhance model performance, particularly for logistic regression.
    - Feature engineering, such as combining **Safety** and **Persons** into a composite feature, might help refine predictions.
5. **Broader Implications**:
    - The findings reinforce the practical importance of **Safety** and **Persons**, which should be prioritized in car design, marketing, and recommendation systems.
    - Lower importance of **Lug_boot** and **Doors** suggests these features have limited influence on consumer acceptability and can be deprioritized in predictive modeling.

# Identify limitations

most of the limitations are caused by the dataset 

1. **Imbalanced Class Distribution**: The dataset has a significant imbalance in the target classes. The "unacc" class represents over 70% of the dataset, while "good" and "v-good" together make up less than 8%. This imbalance can lead to biased models that perform well on the majority class but poorly on the minority classes.
2. **Lack of Numerical Attributes**: All attributes are categorical. While this is suitable for decision-tree-based models, it limits the use of algorithms that rely on numerical data or require meaningful distance metrics.
3. **Simplified Feature Representation**: The dataset omits intermediate structural concepts such as **PRICE**, **TECH**, and **COMFORT**, directly linking the target to the six input features. This simplification may hinder the exploration of relationships between higher-level features and the target.
4. **Limited Number of Attributes**: With only six attributes, the dataset is relatively simplistic. It may not fully represent the complexity of real-world car evaluation scenarios.

# Future Work

This project demonstrates the effectiveness of machine learning in predicting car acceptability and highlights how different car attributes influence consumer decisions. However, there are several avenues for further research and development to enhance the utility and impact of this analysis. Future work can focus on the following directions:

- **Incorporating Additional Features**: Extend the dataset to include real-world factors such as fuel efficiency, emission standards, or brand reputation for a more comprehensive evaluation.
- **Advanced Modeling Techniques**: Explore deep learning models or hybrid approaches combining neural networks with traditional machine learning for improved predictions.
- **Real-Time Predictions**: Develop an interactive web application or API that allows users to input car attributes and receive instant acceptability predictions.
- **Explainability and Interpretability**: Use techniques like SHAP (SHapley Additive exPlanations) to provide insights into the contribution of each feature, enhancing model transparency and trust.
- **Clustering Analysis**: Perform advanced clustering to group cars with similar characteristics, which could help manufacturers target specific market segments.
