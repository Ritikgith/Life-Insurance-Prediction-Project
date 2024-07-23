import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression 
from sklearn import metrics

insurance_dataset = pd.read_csv(r'C:/Users/DELL/Desktop/DATA SCIENCE PROJECTS/life insurance project/Health_insurance.csv')

print(insurance_dataset.describe())

# Age distribution
sns.set()
plt.figure(figsize=(6,6))
sns.displot(insurance_dataset['age'])
plt.title('Age distribution')
plt.show()
print(insurance_dataset['age'].value_counts())



# # # Smoke distribution
# # sns.set()
# plt.figure(figsize=(4,4))
# sns.countplot(x=('smoker'),data=insurance_dataset)
# plt.title("Smoke Distribution")
# plt.show()
# print(insurance_dataset['smoker'].value_counts())

# # sns.set()
# plt.figure(figsize=(4,4))
# sns.countplot(x='sex',data=insurance_dataset)
# plt.title('Sex distribution')
# plt.show()

# print(insurance_dataset['sex'].value_counts())

# plt.figure(figsize=(4,4))
# sns.displot(insurance_dataset['bmi'])
# plt.title("BMI distribution")
# plt.show()

# # Normal BMI range = 18.5 to 24.9

# plt.figure(figsize=(6,6))
# plt.title("Children Distribution")
# sns.countplot(x='children',data=insurance_dataset)
# plt.show()

# print(insurance_dataset['children'].value_counts())

# plt.figure(figsize=(4,4))
# sns.displot(x='region',data=insurance_dataset)
# plt.title("Region")
# plt.show()

# print(insurance_dataset['region'].value_counts())

# plt.figure(figsize=(6,6))
# sns.displot(x='charges',data=insurance_dataset)
# plt.title("Charges")
# plt.show()