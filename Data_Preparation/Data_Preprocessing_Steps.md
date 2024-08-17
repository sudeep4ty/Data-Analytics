
# Data Preprocessing Steps

This document explains the data preprocessing steps performed on the dataset before modeling. Each step is detailed below, including the rationale behind the need for a scaled version of the data.

## Step 1: Uploading the Dataset
The first step in the preprocessing pipeline involves uploading the dataset into the environment. In Google Colab, we use the `files.upload()` function to browse and select the dataset file from our local machine. Once uploaded, the dataset is loaded into a DataFrame using `pandas`, allowing us to manipulate and analyze the data.

```python
from google.colab import files
import pandas as pd

# Upload the file
uploaded = files.upload()

# Load the dataset
data = pd.read_csv('Dataset.csv')

# Display the first few rows of the dataset
data.head()
```
## Step 2: Checking for Missing Values
After loading the dataset, it’s crucial to check for any missing values. Missing data can lead to biased or incorrect models, so identifying and handling these values is essential. We use the `isnull().sum()` function to count the number of missing values in each column of the dataset.

```python
# Check for missing values in the dataset
missing_values = data.isnull().sum()

# Display columns with missing values
missing_values[missing_values > 0]
```
## Step 3: Removing the duplicates
We have removed the duplicates values almost 103 rows have the duplicate data rows which were removed at the first and then we proceed to the next step

```python
# Checking for duplicates in the dataset
duplicate_count = data.duplicated().sum()
print(f"Number of duplicate rows: {duplicate_count}")

# Remove duplicates if any
data = data.drop_duplicates()
```
## Step 4: Handling Missing Values
For columns with missing numerical data, we fill the missing values with the mean of that column. This approach ensures that the missing data does not skew the analysis, retaining the overall distribution of the data.

```python
# Fill missing numerical values with the mean
data.fillna({
    'MonthlyCharges': data['MonthlyCharges'].mean(),
    'tenure': data['tenure'].mean()
}, inplace=True)

# Display the first few rows to confirm changes
data.head()
```
## Step 5: Descriptive Statistics
We calculate the mean, median, mode, and standard deviation for numerical columns to understand the distribution and spread of the data. Additionally, we check for skewness to assess the symmetry of the data distribution.
Results:
    -The dataset has a significant number of non-senior citizens, with a small proportion of senior citizens contributing to the positive skew in SeniorCitizen.
    -Customer tenures are fairly spread out, with a central tendency around 29-32 months, but a long tail of customers who have been with the service for much longer.
    -Monthly charges are varied, with a mean slightly lower than the median, suggesting a few customers paying much less, pulling down the average.



## Step 6: Encoding Categorical Variables
Machine learning models typically require numerical input, so we convert categorical variables into numerical format using one-hot encoding. This method creates binary columns for each category, making the data suitable for modeling.

```python
# Convert categorical variables to numeric using one-hot encoding
encoded_data = pd.get_dummies(data, drop_first=True)

# Display the first few rows of the encoded data
encoded_data.head()
```

## Step 7: Splitting the Dataset into Training and Testing Sets
To evaluate the model's performance, we split the dataset into two parts: a training set and a testing set. The training set (80% of the data) is used to train the model, while the testing set (20%) is used to validate its performance on unseen data. This split is achieved using the `train_test_split` function.

```python
from sklearn.model_selection import train_test_split

# Define features (X) and target (y)
X = encoded_data.drop('Churn_Yes', axis=1)
y = encoded_data['Churn_Yes']

# Split the dataset into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Display the shapes of the training and testing sets
X_train.shape, X_test.shape, y_train.shape, y_test.shape
```

## Step 8: Scaling the Data
**Why Scaling is Necessary:** Scaling is crucial because different features in the dataset might have different units and scales. If one feature has a much larger scale than another, it can dominate distance-based calculations and skew the results of the model. Scaling ensures that each feature contributes equally to the model's predictions.

We use standard scaling, where each feature is centered around zero and has a standard deviation of one. This normalization helps in improving the convergence speed of some algorithms and ensures that all features contribute equally.

```python
from sklearn.preprocessing import StandardScaler

# Initialize the scaler
scaler = StandardScaler()

# Fit the scaler on the training data and transform both the training and testing data
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Convert the scaled data back to a DataFrame for easier handling
X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns)
X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test.columns)

# Display the first few rows of the scaled training data
X_train_scaled.head()
```

## Step 9: Saving the Preprocessed Data and Downloading the Files
Finally, we save the preprocessed training and testing datasets as CSV files. These files can then be downloaded for further analysis or modeling. By saving the data, we ensure that the same cleaned and scaled data is used consistently throughout the project.

```python
# Save the training and testing datasets
X_train_scaled.to_csv('X_train_scaled.csv', index=False)
X_test_scaled.to_csv('X_test_scaled.csv', index=False)
y_train.to_csv('y_train.csv', index=False)
y_test.to_csv('y_test.csv', index=False)

# Download the datasets
files.download('X_train_scaled.csv')
files.download('X_test_scaled.csv')
files.download('y_train.csv')
files.download('y_test.csv')
```

---

This concludes the data preprocessing steps. By following these steps, we ensure that the dataset is properly prepared for modeling, leading to more accurate and reliable results.
