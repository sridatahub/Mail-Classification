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
