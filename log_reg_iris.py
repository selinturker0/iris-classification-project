import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

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

log_reg = LogisticRegression()
log_reg.fit(X_train, y_train)
y_pred = log_reg.predict(X_test)

np.set_printoptions()
print(np.concatenate((y_pred.reshape(len(y_pred), 1), y_test.reshape(len(y_test), 1)), 1))

from sklearn.metrics import accuracy_score

print(accuracy_score(y_test, y_pred))

from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, y_pred)
print('\n Confusion matrix\n\n', cm)

sns.set_style("whitegrid")
sns.pairplot(dataset.loc[:, dataset.columns != 'Id'], hue='Species', height=1.5)
plt.show()
