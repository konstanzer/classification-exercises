import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import pandas_profiling

import scipy.stats as scs
from sklearn.model_selection import train_test_split

import warnings
warnings.filterwarnings("ignore")

from prepare import prep_iris


iris = prep_iris()
iris.info()


iris.profile_report()


iris.iloc[:,:5].describe()


sns.pairplot(iris.iloc[:,:5], hue='species', diag_kind='hist');


measurements = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
for x in measurements:
    f = plt.subplots(figsize=(12, .5))
    sns.boxplot(x=iris[x], color='gray')
    plt.xlim(0,8)
    plt.show()


#the distribution looks symmetrical here, outliers are probably okay
sns.histplot(iris.sepal_width);


#all tests show significant differences
for x in measurements:
    m1 = iris[x][iris.species == 'versicolor']
    m2 = iris[x][iris.species == 'virginica']
    print(x, round(np.mean(m1),2), round(np.mean(m2),2))
    t, p = scs.mannwhitneyu(m1, m2)
    print(f"   p-value: {p}")


f = plt.subplots(figsize=(12, 5))
im_melting = pd.melt(iris.iloc[:,:5], id_vars='species', var_name='part',value_name="measured")
sns.swarmplot(x=im_melting.measured, y=im_melting.part, hue=im_melting.species);


f = plt.subplots(figsize=(12, 7))
sns.violinplot(x=im_melting.measured, y=im_melting.part, hue=im_melting.species);


X_train, X_test, y_train, y_test = train_test_split(iris, iris.species, test_size=.15, stratify=iris.species)


X_train.shape, y_train.shape, X_test.shape, y_test.shape


#divide again if you want a dev set
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=15/85, stratify=y_train)


X_train.shape, y_train.shape, X_test.shape, y_test.shape, X_val.shape, y_val.shape



