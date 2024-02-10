import pandas as pd
import numpy as np
from sklearn.model_selection import  train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix

loans = pd.read_csv('loan_data.csv') # read in the data 

cat_feats = ['purpose'] 

final_data = pd.get_dummies(loans, columns = cat_feats, drop_first = True)

X = final_data.drop('not.fully.paid', axis = 1) # Sets all the features to be trained on and drops the Target Column

y = final_data['not.fully.paid'] # Sets the target value

# train test split 
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.3) 

# instantiate the DecisionTreeClassifier object
dtree = DecisionTreeClassifier()

dtree.fit(X_train, y_train) # fit the data onto the model

predictions = dtree.predict(X_test) # predict results

print(confusion_matrix(y_test, predictions))
print('\n')
print(classification_report(y_test, predictions))

'''
confusion_matrix:
[[2017  403]
 [ 356   98]]

Classfication Report
              precision    recall  f1-score   support

           0       0.85      0.83      0.84      2420
           1       0.20      0.22      0.21       454

    accuracy                           0.74      2874
   macro avg       0.52      0.52      0.52      2874
weighted avg       0.75      0.74      0.74      2874
'''
