"Implement K-Means clustering/ hierarchical clustering on sales _data _sample.csv dataset. Determine
the number of clusters using the elbow method."

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score,recall_score, f1_score
#take new line in jupyter
 data=pd.read_csv("diabetes.csv")
data
#take new line in jupyter
 X = data.drop("Outcome", axis=1)
y = data["Outcome"] 
#take new line in jupyter
X
#take new line in jupyter
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,random_state=42)
#take new line in jupyter
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
#take new line in jupyter
X_train
#take new line in jupyter
k = 3 
knn = KNeighborsClassifier(n_neighbors=k)
knn.fit(X_train, y_train)
#take new line in jupyter
y_pred = knn.predict(X_test)
y_pred
#take new line in jupyter
conf_matrix = confusion_matrix(y_test, y_pred)
accuracy = accuracy_score(y_test, y_pred)
error_rate = 1 - accuracy
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
print("Confusion Matrix:")
print(conf_matrix)
print("Accuracy:", accuracy)
print("Error Rate:", error_rate)
print("Precision:", precision)
print("Recall:", recall)
#End
