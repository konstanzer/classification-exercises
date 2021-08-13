import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt

import scipy.stats as scs
from sklearn.model_selection import train_test_split

import warnings
warnings.filterwarnings("ignore")

from prepare import prep_iris


iris = prep_iris()
iris.info()


iris.iloc[:,:5].describe()


sns.pairplot(iris.iloc[:,:5], hue='species', diag_kind='hist');


measurements = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width']
for x in measurements:
    f = plt.subplots(figsize=(12, .5))
    sns.boxplot(x=iris[x], color='gray')
    plt.xlim(0,8)
    plt.show()


#the distribution looks symmetrical here
sns.histplot(iris.sepal_width);


for x in measurements:
    m1 = iris[x][iris.species == 'versicolor']
    m2 = iris[x][iris.species == 'virginica']
    print(x, round(np.mean(m1),2), round(np.mean(m2),2))
    t, p = scs.mannwhitneyu(m1, m2)
    print(f"   p-value: {p}")


im_melting = pd.melt(iris.iloc[:,:5], id_vars='species')
#sns.swarmplot(im_melting)
im_melting



