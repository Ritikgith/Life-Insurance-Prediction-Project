import pandas as pd
import numpy as np
import numpy as nparray
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression 
from sklearn import metrics

insurance_dataset = pd.read_csv(r'C:/Users/DELL/Desktop/life insurance project/Health_insurance.csv')

x=insurance_dataset.drop(columns='charges',axis=1)
y=insurance_dataset['charges']

print(x)
print(y)


insurance_dataset.replace({'sex':{'male':0,'female':1}},inplace=True)

insurance_dataset.replace({'smoker':{'yes':0,'no':1}},inplace=True)

insurance_dataset.replace({'region':{'southeast':0,'southwest':1,'northeast':2,'northwest':3}},inplace=True)

print(insurance_dataset)

X=insurance_dataset.drop(columns='charges',axis=1)
Y=insurance_dataset['charges']



print(x)
print(y)

X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size=0.2, random_state=2)
print(X.shape,X_train.shape,X_test.shape)


regressor=LinearRegression()

regressor.fit(X_train, Y_train)


training_data_prediction=regressor.predict(X_train)

r2_train=metrics.r2_score(Y_train, training_data_prediction)
print('R square value: ', r2_train)

test_data_prediction=regressor.predict(X_test)
r2_test=metrics.r2_score(Y_test, test_data_prediction)
print('R square value: ', r2_test)

input_data=(18,1,27,0,0,3)

input_data_as_numpy_array=np.asarray(input_data)

input_data_reshape=input_data_as_numpy_array.reshape(1,-1)

prediction=regressor.predict(input_data_reshape)
print(prediction)

print("The insurance cost is USD: ",prediction[0])

input_data=(18,0,34.1,0,0,0)

input_data_as_numpy_array=np.asarray(input_data)

input_data_reshape=input_data_as_numpy_array.reshape(1,-1)

prediction=regressor.predict(input_data_reshape)
print(prediction)

print("The insurance cost is USD: ",prediction[0])