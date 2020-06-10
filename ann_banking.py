# -*- coding: utf-8 -*-
"""
Editor: Eric Albrecht
THis is the test deep learning model of the Udemy Course
"""

import numpy as np
#import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split

#Verification of tensorflow version
# print(tf.__version__)

#read in the dataset
#exclude columns you know won't have an impact ex. name
dataset = pd.read_csv('Churn_Modelling.csv')

#Takes all rows, and takes columns 3 to every column but the last
x = dataset.iloc[:,3:-1].values

#Takes all of the actual values out of the data set
y = dataset.iloc[:,-1].values

#print out the imported data features to verify entire dataset is entered
# print(x)
# print(y)

# encodes values to numeric values, randomly decided to binary
le = LabelEncoder()
x[:,2] = le.fit_transform(x[:,2])

# using one hot encoding to encode all values
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [1])], remainder='passthrough')
x = np.array(ct.fit_transform(x))

# splitting the data set into the training set and the testing set
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state=0)

# feature scaling is fundamental. 
sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)

#Initialize ANN. This is layered rather than spider webbed
ann = tf.keras.models.Sequential()

#Add input layer and first hidden layer. This is done using the dense class
# relu refers to rectifier
ann.add(tf.keras.layers.Dense(units=6, activation='relu'))

#This is the second hidden layer of the neural network
ann.add(tf.keras.layers.Dense(units=6, activation='relu'))

#Adding the output layer. Sigmoid activation function allows for predictions and probabilities
ann.add(tf.keras.layers.Dense(units=1,activation='sigmoid'))

# Compiling the ANN
ann.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])  

#Training the ANN on the training set
ann.fit(x_train, y_train, batch_size = 32, epochs = 100)          



