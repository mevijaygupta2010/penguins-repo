import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#Read the data
penguin_data=pd.read_csv("Penguins_Cleaned.csv")
#penguin_data.head()
species_mapper={'Adelie':0,'Chinstrap':1,'Gentoo':2}
def ct_func_encode(val):
    return species_mapper[val]
penguin_data['species']=penguin_data['species'].apply(ct_func_encode)
penguin_data.head()

#Ordinal Feature Encoding
columns=['sex','island']
for col in columns:
    dummy = pd.get_dummies(penguin_data[col], prefix=col)
    penguin_data = pd.concat([penguin_data,dummy], axis=1)
    del penguin_data[col]
penguin_data.head()

# Separating X and y
X = penguin_data.drop('species', axis=1)
Y = penguin_data['species']

#splitting the data

#from sklearn.model_selection import train_test_split
#X_train, X_test, Y_train, y_test = train_test_split(X,Y,test_size=0.2, random_state=13)

# Build random forest model
from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier()
clf.fit(X, Y)
# Saving the model
import pickle
pickle.dump(clf, open('penguins_clf.pkl', 'wb'))
