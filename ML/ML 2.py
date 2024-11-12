"Classify the email using the binary classification method. Email Spam detection has two states: a)
Normal State - Not Spam, b) Abnormal State - Spam. Use K-Nearest Neighbors and Support Vector
Machine for classification. Analyze their performance"


import pandas as pd
from sklearn.model_selection import train test_split from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC from sklearn.metrics import accuracy_score, classification_report
data= pd.read_csv("emails.csv")
#take new line in jupyter
data.drop(['Email No.'], axis-1, inplace=True)
#take new line in jupyter
X= data.drop("Prediction", axis=1)
y= data["Prediction"] Target variable print("Features: ",X) print("Target: ",y)
print("Features:" ,X)
print("Target:" ,y)
#take new line in jupyter
X_train, X_test, y_train, y_test train_test_split(X, y, test size=0.3, random_state=42)
#take new line in jupyter
knn_model = KNeighborsClassifier(n_neighbors=5) 
knn_model.fit(X_train, y_train)
svm_model = SVC()
svm_model.fit(X_train, y_train)
#take new line in jupyter
knn_predictions = knn_model.predict(X_test) 
knn_accuracy accuracy_score(y_test, knn_predictions) 
knn_report classification_report(y_test, knn_predictions)
#take new line in jupyter
print(knn_predictions)
#take new line in jupyter
print("K-Nearest Neighbors Accuracy:")
print(knn_accuracy)
print("K-Nearest Neighbors Classification Report:")
print(knn_report)
#take new line in jupyter
svm_predictions = svm_model.predict(X_test)
svm_accuracy = accuracy_score(y_test, svm_predictions)
svm_report classification_report(y_test, svm_predictions)
#take new line in jupyter
 print(svm_predictions)
#take new line in jupyter
print("Support Vector Machine Accuracy:")
print(svm_accuracy)
print("Support Vector Machine Classification Report:")
print(svm_report)
