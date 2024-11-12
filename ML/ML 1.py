"Predict the price of the Uber ride from a given pickup point to the agreed drop-off location. Perform
following tasks: 1. Pre-process the dataset. 2. Identify outliers. 3. Check the correlation. 4. Implement
linear regression and random forest regression models. 5. Evaluate the models and compare their
respective scores like R2, RMSE, etc."


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_madel import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error
data = pd.read_csv("Uber.csv")
#take new line in jupyer
data["pickup_datetime"] = pd.to_datetime(data["pickup_datetime"])
missing_values = data.isnull().sum()
print("Missing values in the dataset:")
print(missing_values)
data. dropna(inplace-True)
missing_values = data.isnull().sum()
print( "Missing values after handlin
print(missing_values)
sns.boxplot (x=data["fare_amount”])
plt.show() 
#take new line in jupyer 
Ql = data["fare_amount"].quantile(@.25)
Q3 = data["fare_amount"].quantile(@.75)
TQR = Q3 - Ql
threshold = 1.5
lower_bound = Ql - threshold * IQR
upper_bound = Q3 + threshold * IQR
#take new line in jupyer            
data_no_outliers = data{(data["fare_amount"] >= lower_bound) & (data["fare_amount"] <= upper_bound)]
sns.boxplot (x=data_no_outliers["fare_amount" ])
plt.show()
 #take new line in jupyter
data.plot(kind="box",subplots=True, layout=(7, 2), figsize=(15, 20))
    #take new line in jupyter
correlation_matrix = data.corr()

sns.heatmap(correlation_matrix, annot=True)

pit. show()
  #take new line in jupyter
 
X = data[['pickup_longitude’, ‘pickup_latitude', “dropoff_longitude’, ‘dropoff_latitude', "passenger_count']]
y = data['fare_amount'] 
y
 #take new line in jupyter
 X_train, X_test, y_train, y_test = train_test_split(x, y, test_size-@.2, random_state=42)
  #take new line in jupyter
 lr_model = LinearRegression()
lr_model.fit(X_train, y_train)
 #take new line in jupyter
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(x_train, y_train)  
 #take new line in jupyter
 y_pred_lr = lr_model.predict(x_test)
y_pred_ir

print("Linear Model:”,y_pred_lr)
y_pred_rf = rf_model.predict(x_test)

print("Random Forest Model:", y_pred_rf)
   #take new line in jupyter
r2_lr = r2_score(y_test, y_pred_ir)
rmse_lr = np.sqrt(mean_squared_error(y test, y_pred_lr))
    #take new line in jupyter
print("Linear Regression - R2:", r2_lr)
print("Linear Regression - RMSE:", rmse_lr)
 #take new line in jupyter
 r2_rf = r2_score(y_test, y_pred_rf)
rmse_rf = np.sqrt(mean_squared_error(y test, y_pred_rf))
print("Random Forest Regression R2:", r2_rf)
print("Random Forest Regression RMSE:",rmse_rf)
       #end
       
      
 
                 
                
