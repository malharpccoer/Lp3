"Classify the email using the binary classification method. Email Spam detection has two states: a)
Normal State - Not Spam, b) Abnormal State - Spam. Use K-Nearest Neighbors and Support Vector
Machine for classification. Analyze their performance"

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

df = pd.read_csv('spam_email_data.csv')
df['label'] = df['label'].map({'spam': 1, 'not spam': 0})

vectorizer = CountVectorizer(stop_words='english')
X = vectorizer.fit_transform(df['email'])
y = df['label']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

knn_model = KNeighborsClassifier(n_neighbors=5)
knn_model.fit(X_train, y_train)
y_pred_knn = knn_model.predict(X_test)

svm_model = SVC(kernel='linear', C=1.0)
svm_model.fit(X_train, y_train)
y_pred_svm = svm_model.predict(X_test)

accuracy_knn = accuracy_score(y_test, y_pred_knn)
accuracy_svm = accuracy_score(y_test, y_pred_svm)

print("KNN Model Accuracy:", accuracy_knn)
print("SVM Model Accuracy:", accuracy_svm)

print("\nKNN Classification Report:")
print(classification_report(y_test, y_pred_knn))

print("\nSVM Classification Report:")
print(classification_report(y_test, y_pred_svm))

print("\nKNN Confusion Matrix:")
print(confusion_matrix(y_test, y_pred_knn))

print("\nSVM Confusion Matrix:")
print(confusion_matrix(y_test, y_pred_svm))
