import time
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score

# Import the data
data = pd.read_table('F:/各个学科/工业大数据/lab/project1/data/new_data.csv', sep=',')
train = np.array(data)
data = pd.read_table('F:/各个学科/工业大数据/lab/project1/data/test_set.csv', sep=',')
test = np.array(data)

# The fifth column is the label
a = [5, 6, 7, 8, 9, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
X_train = train[:, a]
y_train = train[:, 4]
X_test = test[:, a]
y_test = test[:, 4]

# RandomForestClassifier
RF = RandomForestClassifier(n_estimators=13, max_depth=10, min_samples_split=2, bootstrap=True)
time_start = time.time()
RF = RF.fit(X_train, y_train)
time_end = time.time()
print('Training time: ', time_end - time_start)
pre = RF.predict(X_test)
print('The score:\t', accuracy_score(y_test, pre))

