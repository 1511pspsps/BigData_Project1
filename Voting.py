import pandas as pd
import numpy as np
import time
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.ensemble import RandomForestClassifier

# Import the data
# -----You should check the path before running the code-----
data = pd.read_table('F:/各个学科/工业大数据/lab/project1/data/new_data.csv', sep=',')
train = np.array(data)
data = pd.read_table('F:/各个学科/工业大数据/lab/project1/data/test_set.csv', sep=',')
test = np.array(data)
a = [5, 6, 7, 8, 9, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
X_train = train[:, a]
y_train = train[:, 4]
X_test = test[:, a]
y_test = test[:, 4]

SVM_clf = SVC(gamma=0.1, probability=True)
MLP_clf = MLPClassifier(hidden_layer_sizes=(10, 10))
DT_clf = DecisionTreeClassifier(criterion='entropy', max_depth=7, min_samples_split=15)
RF_clf = RandomForestClassifier(n_estimators=13, max_depth=10, min_samples_split=2, bootstrap=True)

ensemble = VotingClassifier(estimators=[('mlp', MLP_clf), ('svm', SVM_clf),
                                        ('dt', DT_clf), ('rf', RF_clf)], voting='soft')

time_start = time.time()
ensemble = ensemble.fit(X_train, y_train)
time_end = time.time()
print('Training time: ', time_end - time_start)

pre = ensemble.predict(X_test)
print('The score of ensemble:\t', accuracy_score(y_test, pre))
