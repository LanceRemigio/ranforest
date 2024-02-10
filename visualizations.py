import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

loans = pd.read_csv('loan_data.csv')

# Exploratory Data Analysis

# Create a histogram of two FICO distributions on top of each other, one for each credit.policy outcome.

print(
        loans[['credit.policy', 'fico']].loc[loans['credit.policy'] == 1].head()
        )


plt.figure(figsize = (10,6))

loans['fico'].loc[loans['credit.policy'] == 1].hist(
        bins = 30,
        label =  'Credit Policy = 1'
        )

loans['fico'].loc[loans['credit.policy'] == 0].hist(
        bins = 30,
        label = 'Credit Policy = 0'
        )

plt.xlabel('FICO')
# plt.show()

plt.savefig('./figs/credithist.png')

plt.figure(figsize = (10,6))

loans['fico'].loc[loans['not.fully.paid'] == 1].hist(
        bins = 30,
        label =  'Not Fully Paid = 1'
        )

loans['fico'].loc[loans['not.fully.paid'] == 0].hist(
        bins = 30,
        label = 'Not Fully Paid = 0'
        )
plt.xlabel('FICO')

plt.savefig('./figs/notfullypaidhist.png')

sns.countplot(data = loans, x = 'purpose', hue  = 'not.fully.paid' )
plt.savefig('./figs/count.png')


sns.jointplot(data = loans, x = 'fico', y = 'int.rate')
plt.savefig('./figs/intrateJoint.png')

plt.figure(figsize = (11,6))
sns.lmplot(data = loans, 
           x = 'fico', 
           y = 'int.rate',
           hue = 'credit.policy' ,
           col = 'not.fully.paid',
           )
plt.savefig('./figs/lmplot.png')
