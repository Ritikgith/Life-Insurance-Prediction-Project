import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression 
from sklearn import metrics

insurance_dataset = pd.read_csv(r'C:/Users/DELL/Desktop/life insurance project/Health_insurance.csv')

insurance_dataset.replace({'sex':{'male':0,'female':1}},inplace=True)

insurance_dataset.replace({'smoker':{'yes':0,'no':1}},inplace=True)

insurance_dataset.replace({'region':{'southeast':0,'southwest':1,'northeast':2,'northwest':3}},inplace=True)

print(insurance_dataset)
