# Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee

## AIM:
To write a program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the required libraries .
2. Read the data frame using pandas.
3. Get the information regarding the null values present in the dataframe.
4. Apply label encoder to the non-numerical column inoreder to convert into numerical values.
5. Determine training and test data set.
6. Apply decision tree regression on to the dataframe.
7. Get the values of Mean square error, r2 and data prediction.

## Program:
```
Program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.
Developed by: Jeevitha E
RegisterNumber:  212222230054
```
```
import pandas as pd
df=pd.read_csv("Salary.csv")
df.head(10)
df.info()
df.isnull().sum()
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
df['Position']=le.fit_transform(df['Position'])
df.head(10)
x=df[['Position','Level']]
y=df['Salary']
from sklearn.model_selection import train_test_split as tts
Xtrain,Xtest,Ytrain,Ytest=tts(x,y,test_size=0.2,random_state=2)
from sklearn.tree import DecisionTreeRegressor
dt=DecisionTreeRegressor()
dt.fit(Xtrain,Ytrain)
Ypred=dt.predict(Xtest)
from sklearn import metrics
mse=metrics.mean_squared_error(Ytest,Ypred)
mse
r2=metrics.r2_score(Ytest,Ypred)
r2
dt.predict([[5,6]])
```
## Output:
![image](https://github.com/Jeevithaelumalai/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/118708245/886ae37b-d2b2-4439-8c85-97b5b1de23a5)

![image](https://github.com/Jeevithaelumalai/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/118708245/51ba4849-d534-4057-8cc9-7c32f7f1348e)

![image](https://github.com/Jeevithaelumalai/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/118708245/173b2569-0abf-4c46-9250-06b4d15ad3f0)

![image](https://github.com/Jeevithaelumalai/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/118708245/af25d7c2-b8d7-4395-bc5c-053de9aac8e6)

![image](https://github.com/Jeevithaelumalai/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/118708245/09a951d4-0992-4c12-9bce-35f90986df96)

![image](https://github.com/Jeevithaelumalai/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/118708245/94adc397-d261-4a69-b71c-f1850ba70ead)

![image](https://github.com/Jeevithaelumalai/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/118708245/08457437-2daa-4a62-84ce-8cd9fe642140)

![image](https://github.com/Jeevithaelumalai/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/118708245/fb16b2b3-bcae-4875-96f7-cf54c5759fc3)

![image](https://github.com/Jeevithaelumalai/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/118708245/082d9bcd-5971-4aa5-8b8c-97c2c7ab31ee)

## Result:
Thus the program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee is written and verified using python programming.
