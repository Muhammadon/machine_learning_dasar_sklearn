# sample data set pada scikit learn

from sklearn.datasets import load_iris

iris = load_iris()
# print(iris)

print(iris.keys())
print()

# 1. deskripsi dari sample dataset / metadata
# print(iris.DESCR)

# 2. explanatory & response variables (features & targrt)

# explanatory variables (features)
X = iris.data
print(X.shape)
# print(X)
print()

# response variable (target)
y = iris.target
print(y.shape)
# print(y)
print()

# 3. features & target names
feature_names = iris.feature_names
print(feature_names)
print()

target_names = iris.target_names
print(target_names)
print()

# 4. visualisasi data

# visualisasi sepal length & width
import matplotlib.pyplot as plt

X = X[:, :2] # kita hanya menyertakan dua kolom pertama saja dari variabel X

x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5

plt.scatter(X[:, 0], X[:, 1], c=y)
plt.xlabel('sepal length')
plt.ylabel('sepal width')

plt.xlim(x_min, x_max)
plt.ylim(y_min, y_max)
plt.grid(True)
# plt.show()

# 5. training & testing dataset
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1)

print(f'X train : {X_train.shape}')
print(f'X test : {X_test.shape}')
print(f'y train : {y_train.shape}')
print(f'y test : {y_test.shape}')
print()

# 6. load iris dataset sebagai pandas dataframe
import pandas as pd
iris = load_iris(as_frame=True)

iris_features_df = iris.data
print(iris_features_df)