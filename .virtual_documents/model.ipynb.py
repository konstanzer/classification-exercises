import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import plot_tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import plot_confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import plot_roc_curve

from imblearn.over_sampling import RandomOverSampler


df = pd.read_csv("titanic.csv", index_col=0)
df.info()


#missing values
np.sum(df.isna())


#there
df.age = df.age.fillna(df.age.mean())


df=df.drop(columns=['deck', 'embark_town'])
df=df.dropna(axis=0)


df2=pd.get_dummies(df, drop_first=True)
labels=df2.pop('survived')


df2.head()


df2=df2.drop(columns=['pclass'])


X_train, X_test, y_train, y_test = train_test_split(df2, labels, test_size=.17,
                                                    random_state=36, stratify=labels)
X_train, X_dev, y_train, y_dev = train_test_split(X_train, y_train, test_size=17/83,
                                                  random_state=36, stratify=y_train)


over = RandomOverSampler(random_state=36)
X_ros, y_ros = over.fit_resample(X_train, y_train)


X_train.shape, y_train.shape, X_test.shape, y_test.shape, X_dev.shape, y_dev.shape, X_ros.shape, y_ros.shape


df.survived.value_counts()


base = np.zeros(len(X_train)) #baseline: Jacked
round(sum(base==y_train)/len(X_train),2) #accuracy


#cost complexity pruning removes nodes based on impurity; node complexities above alpha get pruned
dt = DecisionTreeClassifier(max_leaf_nodes=7, ccp_alpha=.007, random_state=36)
dt.fit(X_train, y_train)
print(dt.score(X_dev, y_dev)) #accuracy
plot_confusion_matrix(dt, X_dev, y_dev, cmap='cool');      


#does 0versampling help?
dt_sample = DecisionTreeClassifier(ccp_alpha=.007, random_state=36)
dt_sample.fit(X_ros, y_ros)
print(dt_sample.score(X_dev, y_dev)) #accuracy
plot_confusion_matrix(dt_sample, X_dev, y_dev, cmap='cool'); #maybe not


f = plt.subplots(figsize=(12,6))
#good tree
plot_tree(dt, feature_names=df2.columns, class_names=['live','die']);


pred = dt.predict(X_dev)
#calculates precision, recall, F1, support
print(classification_report(y_dev, pred))


dt_overfit = DecisionTreeClassifier(random_state=36)
dt_overfit.fit(X_train, y_train)
#this overgrown tree badly overfit the training data
plot_confusion_matrix(dt_overfit, X_train, y_train, cmap='cool');


#not so hot on non-training data
print(dt_overfit.score(X_dev, y_dev)) #accuracy
plot_confusion_matrix(dt_overfit, X_dev, y_dev, cmap='cool');


f = plt.subplots(figsize=(12,4))
#this is more convoluted than a stalker's pinboard; deosn't generalize to new, unseen data
plot_tree(dt_overfit);


from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
import eli5


rf = RandomForestClassifier(min_samples_leaf=1, max_depth=10,random_state=36)
rf.fit(X_train, y_train)
rf.score(X_dev, y_dev)


pred = rf.predict(X_dev)
print(classification_report(y_dev, pred))


#tpr(sensitivity) and tnr(specificity)
plot_roc_curve(rf, X_dev, y_dev); #AUC = .88


plot_confusion_matrix(rf, X_dev, y_dev, cmap='bone_r');


#the default forest also overfits but maybe not as severely as the lone tree
print(rf.score(X_train, y_train))
plot_confusion_matrix(rf, X_train, y_train, cmap='bone_r');


#random grid search for better hyperparameters
param_dict = {"min_samples_leaf":[1,2,4,8,16],
              "max_depth":[2,4,8,16,32],
              "n_estimators": [50,100,200,400,800],
              "min_samples_split": [1,2,4]}
rf_rand = RandomizedSearchCV(estimator=RandomForestClassifier(),
                               param_distributions=param_dict,
                               n_iter=50, cv = 3, verbose=2, random_state=36, n_jobs=-1)
rf_rand.fit(X_train, y_train)


params = rf_rand.best_params_
params


rf2 = RandomForestClassifier(**params, random_state=36) #unpack dictionary
rf2.fit(X_train, y_train)
rf2.score(X_dev, y_dev) #better


#best model yet after I dropped extra pclass. struggles more with survivors prol because it is the minority class
#we might try balancing the classes
#the tuned model doesn't overfit as badly as the original forest
plot_confusion_matrix(rf2, X_dev, y_dev, cmap='bone_r');


#feature importance ranking
eli5.show_weights(rf2, feature_names=list(X_train.columns))


pred = rf2.predict(X_dev)
print(classification_report(y_dev, pred))


#tpr(sensitivity) and tnr(specificity)
plot_roc_curve(rf2, X_dev, y_dev); #AUC = .88


from sklearn.neighbors import KNeighborsClassifier


#not the most promising
knn = KNeighborsClassifier(n_neighbors=20)
knn.fit(X_train, y_train)
knn.score(X_dev, y_dev)


plot_confusion_matrix(knn, X_dev, y_dev, cmap='copper_r');


pred = knn.predict(X_dev)
print(classification_report(y_dev, pred))


from sklearn.linear_model import LogisticRegressionCV
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline


#the CV version does some automatic tuning
lr = make_pipeline(StandardScaler(), LogisticRegressionCV(random_state=36))
lr.fit(X_train, y_train)
lr.score(X_dev, y_dev)


pred = lr.predict(X_dev)
print(classification_report(y_dev, pred))


#feature importance ranking
#there are good death features but no good predictors for living, meaning living, from a group perspective, is indiscriminate but dying is biased
#It's like the Titanic chopped down the bigger classes and leveled em
eli5.show_weights(lr, feature_names=list(X_train.columns))


print(df.value_counts("pclass"))
#In fact, this is true. The survivors are much more balanced than than the original boarders.
print(df[df.survived==1].value_counts("pclass"))
print(df[df.survived==0].value_counts("pclass"))


print(df.value_counts("sex"))
#A lot more men got on the doomed ship. By the end women outnumbered them 2 to 1.
print(df[df.survived==1].value_counts("sex"))
print(df[df.survived==0].value_counts("sex"))


plot_confusion_matrix(lr, X_dev, y_dev, cmap='copper_r');


#tpr(sensitivity) and tnr(specificity)
plot_roc_curve(lr, X_dev, y_dev); #AUC = .86


X_train.columns


vars=['class_Third','age', 'fare']
small_train, small_dev = X_train[vars], X_dev[vars]


#predicitng with 3 features is better than baseline
lr2 = LogisticRegression(random_state=36)
lr2.fit(small_train, y_train)
lr2.score(small_dev, y_dev)


vars=['class_Third','age', 'fare', 'sex_male']
small_train, small_dev = X_train[vars], X_dev[vars]


#predicitng with 4 features is much better than baseline
lr2 = LogisticRegression(random_state=36)
lr2.fit(small_train, y_train)
lr2.score(small_dev, y_dev)



