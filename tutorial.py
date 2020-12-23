import tensorflow as tf
from keras.models import Sequential
import pandas as pd
from keras.layers import Dense

data = pd.read_csv('/Users/Shourya/Documents/University/Junior/NU Football Analytics/Keras Tutorial/diabetes.csv', delimiter=",")
data.describe()

# Inspecting values
data.info()

# Checking correlation b/w variables
import seaborn as sns
import matplotlib as plt
corr = data.corr()
#creating heatmap to figure out if 2 values are correlated
sns.heatmap(corr,
            xticklabels=corr.columns.values,
            yticklabels=corr.columns.values)

data['BloodPressure'].corr( data['BMI'])

#Preparing the test and training datasets
import numpy as np

labels=data['Outcome']
features = data.iloc[:,0:8]

from sklearn.model_selection import train_test_split

X=features
#converting column in dataset in an array
y=np.ravel(labels)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

#Normalizing the values
from sklearn.reprocessing import StandardScaler

#Put the data on a standard scale
scaler = StandardScaler().fit(X_train)

#Transform the test and training dataset to normalized values
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

#Picking activation functions for each layer
#Using a rectified linear unit (relu) activation function for the first 2 layers
f(x) = 0 if x <= 0
f(x) = 1 if x > 0

from keras.models import Sequential
from keras.layers import Dense

model = Sequential()

model.add(Dense(8, activation='relu', input_shape=(8,)))

model.add(Dense(8, activation='relu'))

model.add(Dense(1, activation='sigmoid'))

