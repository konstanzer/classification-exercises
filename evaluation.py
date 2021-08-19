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

"""
cs.csv shows defects and preds from 3 models
which metric/model is best to identify the most defects?
"" the least errors in defects predictions?
"""
defects = pd.read_csv("c3.csv")

"""
paws.csv shows cat/dog labels and preds from 4 models
make a baseline and compare models to its accuracy
if preds are only made on dogs, which model for phase 1 and 2?
for cats?
"""
catsdogs = pd.read_csv("paws.csv")
print(catsdogs['actual'].value_counts()) #3254 doggos
base = np.ones((3254,))
print(base)

