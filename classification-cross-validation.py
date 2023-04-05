import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score

import warnings
warnings.filterwarnings("ignore")

digits = load_digits()

X = digits.data
y = digits.target

# Visuallizing the images
# plt.gray()
# plt.matshow(digits.images[9])
# plt.show()

def find_score(model, X_train, X_test, y_train, y_test):
    model.fit(X_train, y_train)
    return model.score(X_test, y_test)

# K Fold Cross Validation From Scratch

kf = KFold(n_splits=15)

lr_score = []
svm_score = []
rf_score = []

for train_index, test_index in kf.split(digits.data):
    X_train, X_test, y_train, y_test = train_test_split(digits.data,digits.target,test_size=0.1,random_state=0)
    lr_score.append(find_score(LogisticRegression(),X_train, X_test, y_train, y_test))
    svm_score.append(find_score(SVC(),X_train, X_test, y_train, y_test))
    rf_score.append(find_score(RandomForestClassifier(),X_train, X_test, y_train, y_test))

lr_score = np.array(lr_score)
svm_score = np.array(svm_score)
rf_score = np.array(rf_score)

print(f"Score of Logistic Regression is {lr_score.mean()}")
print(f"Score of SVM is {svm_score.mean()}")
print(f"Score of RandomForestClassifier is {rf_score.mean()}")

# OUTPUT
#Score of Logistic Regression is 0.9611111111111108
# Score of SVM is 0.9888888888888889
# Score of RandomForestClassifier is 0.9748148148148146

#### Sklearn cross val score
print(cross_val_score(LogisticRegression(), digits.data, digits.target))
print(cross_val_score(SVC(), digits.data, digits.target))
print(cross_val_score(RandomForestClassifier(n_estimators=20), digits.data, digits.target))

# OUTPUT
# [0.92222222 0.86944444 0.94150418 0.93871866 0.89693593]
# [0.96111111 0.94444444 0.98328691 0.98885794 0.93871866]
# [0.93333333 0.88888889 0.94428969 0.95543175 0.89693593]