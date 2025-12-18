import numpy as np
import pandas as pd

df = pd.read_csv('placement.csv')
# Steps

# 0. Preprocess + EDA + Feature Selection
# 1. Extract input and output cols
# 2. Scale the values
# 3. Train test split
# 4. Train the model
# 5. Evaluate the model/model selection
# 6. Deploy the model
     
print(df.head())
print(df.info())
print(df.shape)

#Remove the unecessary column

df=df.iloc[:,1:]
print(df.head())

#Using matplotlib for visual representation of the data 

import matplotlib.pyplot as plt

plt.scatter(df['cgpa'],df['iq'],c=df['placement'])
plt.show()

x=df.iloc[:,0:2]
y=df.iloc[:,-1]

print(x)
print(y)

from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.1)

print(x_train)
print(y_train)

print(x_test)
print(y_test)

from sklearn.preprocessing import StandardScaler

scaler=StandardScaler()

x_train=scaler.fit_transform(x_train)

print(x_train)

x_test=scaler.transform(x_test)

print(x_test)

from sklearn.linear_model import LogisticRegression

clf=LogisticRegression()

#--------MODEL TRAINING--------

clf.fit(x_train,y_train)

y_pred=clf.predict(x_test)

print(y_test)

#------CHECKING ACCURACY OF THE MODEL---------
from sklearn.metrics import accuracy_score

accuracy=accuracy_score(y_test,y_pred)
print("Accuracy of the model=",accuracy)

#------DEPLOYING THE MACHINE LEARNING MODEL--------
import pickle

pickle.dump(clf,open('model.pkl','wb'))


