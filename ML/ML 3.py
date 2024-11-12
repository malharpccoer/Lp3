"Given a bank customer, build a neural network-based classifier that can determine whether they will
leave or not in the next 6 months.
Dataset Description: The case study is from an open-source dataset from Kaggle. The dataset contains
10,000 sample points with 14 distinct features such as CustomerId, CreditScore, Geography, Gender,
Age, Tenure, Balance, etc. Perform following steps:
1. Read the dataset.
2. Distinguish the feature and target set and divide the data set into training and test sets.
3. Normalize the train and test data.
4. Initialize and build the model. Identify the points of improvement and implement the same.
5. Print the accuracy score and confusion matrix (5 points)."


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
plt.style.use('dark_background' )
#take new line in jupyter
df = pd.read_csv(' /kaggle/input/bank-customer-churn-modeling/Churn_Modelling.csv')
df.head()
#take new line in jupyter
df.shape
#take new line in jupyter
df.info()
#take new line in jupyter
df = df.drop(['RowNumber', 'Surname','CustomerId', 'EstimatedSalary'], axis = 1)
df.head()
#take new line in jupyter
tenure_exited_no = df[df.Exited==@] .Tenure
tenure_exited_yes = df[df.Exited==1].Tenure
#take new line in jupyter
plt.figure(figsize=(18,5))
sns.histplot(tenure_exited_yes, color='pink', label='Exited: Yes’)
sns.histplot(tenure_exited_no, color='purple', label='Exited: No', alpha = 0.4)
plt.xlabel( ‘Tenure’ )
plt.ylabel('Number of Customers’ )
plt.title('Distribution of Tenure’)
plt.legend()
plt.show()
 #take new line in jupyter
numOfProducts_exited_yes = df[df.Exited==1] .NumOfProducts
numOfProducts_exited_no = df[df.Exited==0] .NumOfProducts 
 #take new line in jupyter
plt.figure(figsize=(10,5))
sns.histplot(num0fProducts_exited_yes, color='red', label='Exited: Yes')
sns.histplot(numOfProducts_exited_no, color='purple', label='Exited: No', alpha = 0.4)
plt.xlabel('Number of Products’)
plt.ylabel('Number of Customers’)
plt.title('Distribution of Products’ )
plt.legend()
plt.show()  
#take new line in jupyter
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
df.Geography = le.fit_transform(df.Geography)
df.Gender = le.fit_transform(df.Gender)
#take new line in jupyter
 df.head()        
#take new line in jupyter   
from sklearn.preprocessing import MinMaxScaler
sclr = MinMaxScaler()
df.CreditScore = sclr.fit_transform(df[['CreditScore' ]])
df.Age = sclr.fit_transform(df[['Age']])
df .Balance = sclr.fit_transform(df[[' Balance’ ]])   
#take new line in jupyter
 df.head()
 #take new line in jupyter
x = df.drop(['Exited'], axis=1)
y = df.Exited   
  #take new line in jupyter  
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
  #take new line in jupyter    
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Dense, Flatten
 #take new line in jupyter 
 model.evaluate(x_test,y_test)
  #take new line in jupyt
 predictions = model.predict(x_test)
predictions
  #take new line in jupyter   
y-pred = []
for i in predictions:
if i> 0.5:
y_pred.append(1)
else:
y_pred.append(0)
  #take new line in jupyter
y_pred[0:5]
  #take new line in jupyter
from sklearn.metrics import confusion_matrix, classification_report
cr = classification_report(y_test, y_pred)
print(cr)
  #take new line in jupyter
 cm = confusion_matrix(y_test, y_pred)
cm
 #take new line in jupyter
plt.figure(figsize=(5,4))
sns.heatmap(cm, annot=True, fmt='d')
plt.xlabel( ‘Predicted’ )
plt.ylabel( ‘Truth’ )                                 
  #take new line in jupyter                                 
                                 
