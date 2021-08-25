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
from imblearn.over_sampling import SMOTE


df = pd.read_csv("titanic.csv", index_col=0)
df.info()


#missing values
np.sum(df.isna())


#there
df.age = df.age.fillna(np.round(df.age.mean(),1))


df=df.drop(columns=['deck', 'embark_town'])
df=df.dropna(axis=0)


df2=pd.get_dummies(df, drop_first=True)


df2.head()


df2=df2.drop(columns=['pclass', 'passenger_id'])


df2.to_csv("titanic_ml.csv")
df2 = pd.read_csv("titanic_ml.csv", index_col=0)
labels=df2.pop('survived')


X_train, X_test, y_train, y_test = train_test_split(df2, labels, test_size=.17,
                                                    random_state=36, stratify=labels)
X_train, X_dev, y_train, y_dev = train_test_split(X_train, y_train, test_size=17/83,
                                                  random_state=36, stratify=y_train)


over = RandomOverSampler(random_state=36)
X_ros, y_ros = over.fit_resample(X_train, y_train)
smote = SMOTE(random_state=36)
X_smote, y_smote = smote.fit_resample(X_train, y_train)


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
plot_tree(dt, feature_names=df2.columns, class_names=['die','live']);


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


#the default forest also overfits but not as severely as the lone tree
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


rf3 = RandomForestClassifier(n_estimators = 100,
                             min_samples_split = 4,
                             min_samples_leaf = 4,
                             max_depth = 4, random_state=36)
rf3.fit(X_train, y_train)
rf3.score(X_dev, y_dev)


#best model yet after I dropped extra pclass. struggles more with survivors prol because it is the minority class
#we might try balancing the classes
plot_confusion_matrix(rf2, X_dev, y_dev, cmap='bone_r');


#using synthetic data
rf_smote = RandomForestClassifier(**params, random_state=36)
rf_smote.fit(X_smote, y_smote)
print(rf_smote.score(X_dev, y_dev)) #accuracy
plot_confusion_matrix(rf_smote, X_dev, y_dev, cmap='bone_r'); #not better


#feature importance ranking
eli5.show_weights(rf2, feature_names=list(X_train.columns))


pred = rf2.predict(X_dev)
print(classification_report(y_dev, pred))


#tpr(sensitivity) and tnr(specificity)
plot_roc_curve(rf2, X_dev, y_dev); #AUC = .88


from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline


#not the most promising
scores_out, scores_in = [], []
k_values = range(1,41)
for k in k_values:
    #don't forget to scale!
    knn = make_pipeline(StandardScaler(),
                        KNeighborsClassifier(n_neighbors=k, weights='uniform'))
    knn.fit(X_train, y_train)
    scores_out.append(round(knn.score(X_dev, y_dev),3))
    scores_in.append(round(knn.score(X_train, y_train),3))    


f = plt.subplots(figsize=(16,4))
plt.plot(k_values, scores_in)
plt.plot(k_values, scores_out)
plt.grid()
plt.xlim(1,40)
plt.legend(["train","dev"])
plt.title("kNN accuracy on train & dev sets for different values k");


#more errors on positive/survived class
k = k_values[np.argmax(scores_out)]
knn = make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=k))
knn.fit(X_train, y_train)
print(knn.score(X_dev, y_dev), k)
plot_confusion_matrix(knn, X_dev, y_dev, cmap='copper_r');


pred = knn.predict(X_dev)
print(classification_report(y_dev, pred))


from sklearn.linear_model import LogisticRegressionCV
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import Ridge
import eli5


#the CV version does some automatic tuning
#in theory the beta can control for different feature scales, here scaling is necessary for convergence
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


features=['class_Third','age', 'fare']
small_train, small_dev = X_train[features], X_dev[features]
#predicitng with 3 features is better than baseline
lr2 = LogisticRegression(random_state=36)
lr2.fit(small_train, y_train)
lr2.score(small_dev, y_dev)


features=['class_Third','age', 'fare', 'sex_male']
small_train, small_dev = X_train[features], X_dev[features]
#predicting with 4 features is much better than baseline
lr2 = LogisticRegression(random_state=36)
lr2.fit(small_train, y_train)
lr2.score(small_dev, y_dev)


#best
features=['class_Third', 'age', 'sex_male', 'sibsp']
small_train, small_dev = X_train[features], X_dev[features]
#predicting with 4 features is much better than baseline
lr2 = LogisticRegression(random_state=36)
lr2.fit(small_train, y_train)
lr2.score(small_dev, y_dev)


plot_confusion_matrix(lr2, small_dev, y_dev, cmap='copper_r');


lr2.coef_[0], lr2.intercept_


features=['sex_male', 'class_Third', 'sibsp', 'age',
          'class_Second', 'alone', 'parch', 'fare']
small_train, small_dev = X_train[features], X_dev[features]
#predicting with 8 features at different regularization levels
lambd = [.001, .01, .1, 1., 10., 100.]
for C in lambd: #.01 does better than default 1 but after ten, no regularization is happening
    lr2 = make_pipeline(StandardScaler(), LogisticRegression(C=C ,random_state=36))
    lr2.fit(small_train, y_train)
    print(C, lr2.score(small_dev, y_dev))
    #lr2 = LogisticRegression(C=C ,random_state=36)
    #lr2.fit(small_train, y_train)
    #print("without scaling: ") #not converging
    #print(C, lr2.score(small_dev, y_dev))


person=36
print(f"prob. of dying and living: {lr2.predict_proba([small_dev.iloc[person]])[0]}")
small_dev.iloc[person]


#https://stats.idre.ucla.edu/other/mult-pkg/faq/general/faq-how-do-i-interpret-odds-ratios-in-logistic-regression/
age, sibsp = 21, 2
logit = np.round(2.798 - .3177*sibsp - .0286*age, 2)
odds = np.round(np.exp(logit), 2) #5:1 calculated as e^logit
prob = np.round(odds/(odds+1), 2) #5/6
print(f"model output, logit, using features and betas: {logit}")
print(f"odds of person 36's survival: {odds}:1")
print(f"equivalent probability of survival: {prob}")


#the transformation from probabilites to odds to logodds
#why bother? it's difficult to output a variable within a restricted range
#probabilites are between 0 and 1 and odds are between  and infinity
probs = np.linspace(0.00001,.9999,1000)
odds = probs/(1-probs)
logodds = np.log(odds)


plt.plot(probs,odds)
plt.ylim(0,10)
plt.title("probs and odds")
plt.show()
plt.plot(odds,logodds)
plt.xlim(0,10)
plt.title("odds and logodds")
plt.show()


#logodds aren't bounded - and this is what logistic regression outputs
#you have to inverse the sigmoid function to get a probability
for odds in [.00001, .0001, .001, .01, .1, 1, 2, 20, 50, 100, 1000]:
    print(odds, np.log(odds))


def sigmoid(x): return 1 / (1+np.exp(-x))
scaler = StandardScaler()

m = 500 #number of samples
y = np.array(y_train[:m])

alpha = .1 #learning rate
epochs = 150

X = np.array(X_train[['sex_male', 'class_Third', 'age']])
X = scaler.fit_transform(X[:m,:]) #ensures rapid convergence
X = np.c_[X, np.zeros((m,))] #extra column for bias term

X[:2]


theta = np.zeros(4) #initialization

for i in range(epochs):
    
    #keep in mind, predicted labels aren't helpful for learning; probabilities produce interpretable losses
    y_pred = sigmoid(X@theta) #vectorized implmenetation
    
    #logloss function
    J = -np.mean(y*np.log(y_pred) + (1-y)*np.log(1-y_pred))
    
    #theta gets updated by the derivative of the cost function w.r.t. theta
    #where's J? it isn't needed to update theta but to check for convergence , bias update???
    theta -= alpha * X.T@(y_pred - y) / m
    
    if iget_ipython().run_line_magic("10", " == 0: print(f\"cost: {np.round(J,3)}, theta: {np.round(theta,3)}\")")


theta = np.zeros(4) #initialization
lambd = .1

for i in range(epochs):
    
    y_pred = sigmoid(X@theta) #vectorized implmenetation
    
    #penalizing large betas reduces overfitting on small datasets
    #l2 (ridge) is deafult: penalty is the square of the magnitude of the betas
    J = -np.mean(y*np.log(y_pred) + (1-y)*np.log(1-y_pred)) + (1/(m*lambd)) * np.sum(np.square(theta))
    
    #theta gets updated by the derivative of the cost function w.r.t. theta, bias update???
    theta -= alpha * X.T@(y_pred - y) / m
    
    if iget_ipython().run_line_magic("10", " == 0: print(f\"cost: {np.round(J,3)}, theta: {np.round(theta,3)}\")")


from sklearn.metrics import mean_squared_error
from sklearn.metrics import log_loss

#but why not MSE? unlike linear regression, it won't give a convex function and you may converge on a local minima
np.mean((y-y_pred)**2) #also the losses are tiny

mean_squared_error(y, y_pred)


log_loss(y, y_pred)



