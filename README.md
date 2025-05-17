# SGD-Classifier
```
Program Developed by : A.Ranen Joseph Solomon
Register Number : 212224040269
```

## AIM:
To write a program to predict the type of species of the Iris flower using the SGD Classifier.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
**1.Import Libraries**

Import pandas, numpy, and required modules from sklearn.

**2.Load and Prepare Data**

Load the Iris dataset, convert it to a DataFrame, and add the target column.

**3.Split Data**

Define features (x) and labels (y), then split into training and test sets.

**4.Train the Model**

Initialize SGDClassifier, then train it using the training data.

**5.Test and Evaluate**

Make predictions on the test set and evaluate using accuracy, confusion matrix, and classification report.

## Program and Output:

```
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report
```
```
iris  = load_iris()
```
```
df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
df['target'] = iris.target
print(df.head())
```
![image](https://github.com/user-attachments/assets/8cc28bf3-3e70-47b4-896b-6e482b7f0dbc)

```
x = df.drop('target', axis=1)
y = df['target']
```
```
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
```
```
sgd_clf = SGDClassifier(max_iter=1000, tol=1e-3)
sgd_clf.fit(x_train, y_train)
```
![image](https://github.com/user-attachments/assets/a4a67caf-1cf9-47a3-8d4c-f2e33828d263)
```
y_pred = sgd_clf.predict(x_test)
```
```
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.3f}")
```
![image](https://github.com/user-attachments/assets/4512cf53-6608-411a-bb3e-fb4c627caf7d)
```
cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(cm)
```
![image](https://github.com/user-attachments/assets/2a68d350-9bd2-4f25-b94d-a674ec2e7716)

```
classification_report1 = classification_report(y_test, y_pred)
print(classification_report1)
```
![image](https://github.com/user-attachments/assets/a36d15e9-6a52-479c-abb9-cf41233455a5)

## Result:
Thus, the program to predict the Iris species using the SGD Classifier was successfully written and tested using Python. The model gave good results with high accuracy and correctly classified most of the flower samples.
