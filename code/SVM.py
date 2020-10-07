import time
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
import numpy as np

# Import the data
data = pd.read_table('F:/各个学科/工业大数据/lab/project1/data/new_data.csv', sep=',')
train = np.array(data)
data = pd.read_table('F:/各个学科/工业大数据/lab/project1/data/test_set.csv', sep=',')
test = np.array(data)

# The fifth column is the label
a = [5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
X_train = train[:, a]
y_train = train[:, 4]
X_test = test[:, a]
y_test = test[:, 4]

# Create a SVC
SVM_clf = SVC(gamma=0.1)

# Calculate the Training time
time_start = time.time()
SVM_clf = SVM_clf.fit(X_train, y_train)
time_end = time.time()
print('Training time: ', time_end - time_start)

# Apply the model
y_pred = SVM_clf.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
