import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression 
from sklearn import metrics

insurance_dataset = pd.read_csv(r'C:/Users/DELL/Desktop/life insurance project/Health_insurance.csv')

# print(insurance_dataset)
print(insurance_dataset.head(5))
print(insurance_dataset.tail(5))

print(insurance_dataset.shape)  # find the no. of rows and columns
