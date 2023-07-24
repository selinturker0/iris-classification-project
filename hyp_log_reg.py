import numpy as np
import pandas as pd

dataset = pd.read_csv("Iris.csv")
# print(dataset)
# print(dataset.groupby("Species").size())

X = dataset.iloc[:, 1:-1].values
y = dataset.iloc[:, -1].values
# print(X)
# print(y)

from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
le.fit_transform(y)
# print(y)

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
# print(X_train)
# print(X_test)
# print(y_train)
# print(y_test)

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
c_space = np.logspace(-5, 8, 15)
param_grid = {'C': c_space}

log_reg = LogisticRegression()
log_reg_cv = GridSearchCV(log_reg, param_grid, cv=5)
log_reg_cv.fit(X, y)


print("Tuned Logistic Regression Parameters: {}".format(log_reg_cv.best_params_))
print("Best score is {}".format(log_reg_cv.best_score_))


