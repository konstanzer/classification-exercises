"""
|           | pred dog   | pred cat   |
|:--------  |-----------:|-----------:|
| is dog    |         46 |         7  |
| is cat    |         13 |         34 |

assuming dog is positve class (can't exactly assume that from order)
fp: is dog, pred cat
fn: is cat, pred dog

accuracy = 46+34 / 46+34+7+13 = .80
precision = tp/tp+fp = 46 / 46+7 = .87
recall = tp/tp+fn = 46 / 46+13 = .78
f1 = 2(p*r / p+r) = 2(.68 / 1.65) = .82

this model needs to pick it up on cat identification b/c I like cats
"""
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelBinarizer

"""
cs.csv shows defects and preds from 3 models
which metric/model is best to identify the most defects?
->precision b/c it punishes missing the postive class
"" the least errors in defects predictions?
-> recall b/c it punishes guessing positive class too much
"""
defects = pd.read_csv("c3.csv")
labels = LabelBinarizer().fit_transform(defects.actual)

for m in ['model1','model2','model3']:
	preds = LabelBinarizer().fit_transform(defects[m])
	print( f"{m} accuracy: {accuracy_score(labels, preds)}" )
	print( f"{m} precision: {precision_score(labels, preds)}" )
	print( f"{m} recall: {recall_score(labels, preds)}\n" )

"""
paws.csv shows cat/dog labels and preds from 4 models
make a baseline and compare models to its accuracy
->models 2 and 3 underperform it
if preds are only made on dogs, which model for phase 1 and 2?
->model 2 has the best precision (dog predictions),
		phase 2 maybe model 1
for cats?
->model 4 ha very few false negatives so that one,
	model 1 is second best so maybe that one for phase 2
"""
catsdogs = pd.read_csv("paws.csv")
print(catsdogs['actual'].value_counts()) #doggos
base = np.ones((5000,))
labels = LabelBinarizer().fit_transform(catsdogs.actual)
print( f"baseline accuracy: {accuracy_score(labels, base)}\n" )

for m in ['model1','model2','model3','model4']:
	preds = LabelBinarizer().fit_transform(catsdogs[m])
	print( f"{m} accuracy: {accuracy_score(labels, preds)}" )
	print( f"{m} precision: {precision_score(labels, preds)}" )
	print( f"{m} recall: {recall_score(labels, preds)}\n" )



