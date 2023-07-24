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
from sklearn.tree import DecisionTreeClassifier

clf_en = DecisionTreeClassifier(criterion='entropy', max_depth=3, random_state=0)
clf_en.fit(X_train, y_train)
y_pred_en = clf_en.predict(X_test)

np.set_printoptions()
print(np.concatenate((y_pred_en.reshape(len(y_pred_en), 1), y_test.reshape(len(y_test), 1)), 1))

from sklearn.metrics import accuracy_score
print(accuracy_score(y_test, y_pred_en))

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred_en)
print('\n Confusion matrix\n\n', cm)