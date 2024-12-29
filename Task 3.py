# Import Libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn import tree
import matplotlib.pyplot as plt

# Load the dataset
# The original URL may have been incorrect or outdated
# This URL points to a file within a .zip archive - we need to handle this differently
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00222/bank-additional.zip"  

# Download the zip file and extract it using the 'requests' and 'zipfile' libraries.
import requests
import zipfile
import io

response = requests.get(url)  # Download zip file
with zipfile.ZipFile(io.BytesIO(response.content)) as z:  # Extract bank-additional-full.csv
    with z.open("bank-additional/bank-additional-full.csv") as f:
        data = pd.read_csv(f, sep=';')


# Display dataset info
print("Dataset Head:")
print(data.head())
print("\nDataset Info:")
print(data.info())

# One-hot encode categorical variables
data_encoded = pd.get_dummies(data, drop_first=True)

# Separate features and target
X = data_encoded.drop("y_yes", axis=1)  # 'y_yes' represents if the customer subscribed
y = data_encoded["y_yes"]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Create the Decision Tree Classifier
dt_classifier = DecisionTreeClassifier(criterion='entropy', max_depth=5, random_state=42)

# Train the model
dt_classifier.fit(X_train, y_train)

# Predict the test set
y_pred = dt_classifier.predict(X_test)

# Evaluate the model
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Visualize the Decision Tree
plt.figure(figsize=(15, 10))
tree.plot_tree(dt_classifier, feature_names=X.columns, class_names=["No", "Yes"], filled=True)
plt.show()
 
