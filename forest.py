import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomForestClassifier

sns.set_style('whitegrid')

loans = pd.read_csv('loan_data.csv')

# print the basic info about the data
# info = [loans.info(), loans.head(), loans.describe()]

# for item in info:
#     print(item, end = '\n')

'''
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 9578 entries, 0 to 9577
Data columns (total 14 columns):
 #   Column             Non-Null Count  Dtype  
---  ------             --------------  -----  
 0   credit.policy      9578 non-null   int64  
 1   purpose            9578 non-null   object 
 2   int.rate           9578 non-null   float64
 3   installment        9578 non-null   float64
 4   log.annual.inc     9578 non-null   float64
 5   dti                9578 non-null   float64
 6   fico               9578 non-null   int64  
 7   days.with.cr.line  9578 non-null   float64
 8   revol.bal          9578 non-null   int64  
 9   revol.util         9578 non-null   float64
 10  inq.last.6mths     9578 non-null   int64  
 11  delinq.2yrs        9578 non-null   int64  
 12  pub.rec            9578 non-null   int64  
 13  not.fully.paid     9578 non-null   int64  
dtypes: float64(6), int64(7), object(1)
memory usage: 1.0+ MB
None

   credit.policy             purpose  int.rate  ...  delinq.2yrs  pub.rec  not.fully.paid
0              1  debt_consolidation    0.1189  ...            0        0               0
1              1         credit_card    0.1071  ...            0        0               0
2              1  debt_consolidation    0.1357  ...            0        0               0
3              1  debt_consolidation    0.1008  ...            0        0               0
4              1         credit_card    0.1426  ...            1        0               0

[5 rows x 14 columns]

       credit.policy     int.rate  installment  ...  delinq.2yrs      pub.rec  not.fully.paid
count    9578.000000  9578.000000  9578.000000  ...  9578.000000  9578.000000     9578.000000
mean        0.804970     0.122640   319.089413  ...     0.163708     0.062122        0.160054
std         0.396245     0.026847   207.071301  ...     0.546215     0.262126        0.366676
min         0.000000     0.060000    15.670000  ...     0.000000     0.000000        0.000000
25%         1.000000     0.103900   163.770000  ...     0.000000     0.000000        0.000000
50%         1.000000     0.122100   268.950000  ...     0.000000     0.000000        0.000000
75%         1.000000     0.140700   432.762500  ...     0.000000     0.000000        0.000000
max         1.000000     0.216400   940.140000  ...    13.000000     5.000000        1.000000

[8 rows x 13 columns]
'''

# Exploratory Data Analysis

# Create a histogram of two FICO distributions on top of each other, one for each credit.policy outcome.

# print(
#         loans[['credit.policy', 'fico']].loc[loans['credit.policy'] == 1].head()
#         )

# fig, ax = plt.subplots()

# loans['fico'].loc[loans['credit.policy'] == 1].hist(ax = ax, bins = 30)
# loans['fico'].loc[loans['credit.policy'] == 0].hist(ax = ax, bins = 30)

# loans['fico'].loc[loans['not.fully.paid'] == 0].hist(ax = ax, bins =  30) 
# loans['fico'].loc[loans['not.fully.paid'] == 1].hist(ax = ax, bins =  30) 

# sns.countplot(data = loans, x = 'purpose', hue  = 'not.fully.paid' )

# sns.jointplot(data = loans, x = 'fico', y = 'int.rate')

sns.lmplot(data = loans, 
           x = 'fico', 
           y = 'int.rate',
           col = 'not.fully.paid',
           hue = 'credit.policy' 
           )

cat_feats = ['purpose']

final_data = pd.get_dummies(loans, columns = cat_feats, drop_first = True)

X = final_data.drop('not.fully.paid', axis = 1)
y = final_data['not.fully.paid'] 

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.3)

rfc = RandomForestClassifier(n_estimators = 200)

rfc.fit(X_train, y_train)

rfc_pred = rfc.predict(X_test)

print(confusion_matrix(y_test, rfc_pred))
print('\n')
print(classification_report(y_test, rfc_pred))

'''
[[2423   15]
 [ 426   10]]


              precision    recall  f1-score   support

           0       0.85      0.99      0.92      2438
           1       0.40      0.02      0.04       436

    accuracy                           0.85      2874
   macro avg       0.63      0.51      0.48      2874
weighted avg       0.78      0.85      0.78      2874

Both the DecisionTreeClassifier and RandomForestClassifier performed about the same.
'''


