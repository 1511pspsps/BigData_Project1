from sklearn.model_selection import GridSearchCV
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
import numpy as np
from sklearn.neural_network import MLPClassifier

# Import the data

# You should check the path before run the code
data = pd.read_table('F:/各个学科/工业大数据/lab/project1/data/new_data.csv', sep=',')
train = np.array(data)
data = pd.read_table('F:/各个学科/工业大数据/lab/project1/data/test_set.csv', sep=',')
test = np.array(data)

a = [5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
X_train = train[:, a]
y_train = train[:, 4]
X_test = test[:, a]
y_test = test[:, 4]

RF_clf__tuned_parameters = {"n_estimators": [10, 11, 12, 13, 14],
                            "max_depth": [6, 7, 8, 9, 10, 11, 12, 13],
                            "min_samples_split": [2, 3, 4, 5, 6, 7]}
RF = RandomForestClassifier()
estimator = GridSearchCV(RF, RF_clf__tuned_parameters, n_jobs=5)
estimator.fit(X_train, y_train)
print("---------------------------------------------------------")
print(estimator.best_params_)
