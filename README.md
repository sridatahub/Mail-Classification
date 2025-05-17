# Mail Classification using Logistic Regression

This project focuses on building a **Mail Classification System** using the **Logistic Regression** algorithm. The primary objective is to automatically classify emails as spam or not spam (ham), which is a common and crucial application in real-world email filtering systems.

### 📌 Why is this important?

With the growing volume of email communication, filtering out spam messages is essential for maintaining productivity and security. This project showcases a practical implementation of a supervised machine learning technique for binary classification, helping users understand the complete workflow of developing a basic spam filter.

---

## 🚀 Project Workflow

The project follows a structured machine learning pipeline:

1. **Importing Libraries and Dataset**  
   Load essential Python libraries and the dataset required for classification.

2. **Data Preprocessing**  
   Clean and prepare the email data by handling null values, converting text to lowercase, removing punctuation, and tokenizing.

3. **Splitting into Training and Test Data**  
   Split the cleaned dataset into training and testing sets to evaluate model performance.

4. **Feature Extraction**  
   Use text vectorization techniques (like TF-IDF or CountVectorizer) to convert text into numerical features.

5. **Model Training and Evaluation**  
   Train a Logistic Regression model and evaluate it using accuracy, confusion matrix, and other metrics.
6. **Building Predictive system**
   Check our model with unseen data 
   

---

## 🔧 Technologies Used

- Python
- scikit-learn
- Pandas
- NumPy
- Matplotlib / Seaborn (for optional visualizations)

---
### 1. Importing Libraries and Dataset

```python
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer  # To convert our text to numerical vectors
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

#Loading the csv data file into pandas dataframe.
raw_mail_data = pd.read_csv("/content/mail_dataset.csv")
raw_mail_data.head()
```

### 2. Data Preprocessing
```python
# Converting null values to empty strings
mail_data = raw_mail_data.where((pd.notnull(raw_mail_data)), '')

# Shape: returns number of rows and columns
mail_data.shape

# Label Encoding: spam = 1, ham = 0
mail_data.loc[mail_data['Category'] == 'spam', 'Category'] = 1
mail_data.loc[mail_data['Category'] == 'ham', 'Category'] = 0

# Preview the data
mail_data.head()

# Separating input (messages) and output (labels)
X = mail_data['Message']
Y = mail_data['Category']
```
### 3. Splitting into Training and Test Data
```python
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.2,random_state=3)
print(X_train.shape)
print(X_test.shape)
print(X.shape)
```
### 4. Feature Extraction

```python
# Transforming the text data into feature vectors that can be used as input for our model (Logistic Regression)
Feature_Extraction = TfidfVectorizer(min_df=1, stop_words='english', lowercase=True)

# Applying TF-IDF transformation to training and testing message data
X_train_features = Feature_Extraction.fit_transform(X_train)
X_test_features = Feature_Extraction.transform(X_test)

# Checking the data type of the labels
Y_train.dtype

# Converting label data from object to integer type to avoid issues during model training
Y_train = Y_train.astype('int')
Y_test = Y_test.astype('int')

# Confirming label data type
Y_train.dtype

# View raw text data before transformation
print(X_train)

# View transformed numerical feature vectors
print(X_train_features)
```
### 5. Model Training and Evaluation

```python
# Creating and training the Logistic Regression model
model = LogisticRegression()
model.fit(X_train_features, Y_train)

# Prediction on training data
training_data_prediction = model.predict(X_train_features)
training_data_accuracy = accuracy_score(Y_train, training_data_prediction)
print("Prediction on training data:", training_data_accuracy)

# Prediction on test data
test_data_prediction = model.predict(X_test_features)
test_data_accuracy = accuracy_score(Y_test, test_data_prediction)
print("Prediction on test data:", test_data_accuracy)

### 6. Predictive System Building

```python
# Sample input mail
mail_input = ["Free entry in 2 a wkly comp to win FA Cup final tkts 21st May 2005. Text FA to 87121 to receive entry question(std txt rate)T&C's apply 08452810075over18's"]

# Transforming the input using the same TF-IDF vectorizer
mail_input_features = Feature_Extraction.transform(mail_input)

# Making prediction
prediction = model.predict(mail_input_features)

# Displaying result
if prediction[0] == 1:
    print("Spam Mail")
else:
    print("Ham Mail")
```
```python
mail_input2 = ["I'm gonna be home soon and i don't want to talk about this stuff anymore tonight, k? I've cried enough today."]
mail_input_features = Feature_Extraction.transform(mail_input2)
prediction = model.predict(mail_input_features)
if prediction[0] == 1:
  print("Spam Mail")
else:
  print("Ham Mail")
```
