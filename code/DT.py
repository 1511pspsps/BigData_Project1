import time
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier

from sklearn.tree import export_graphviz
from six import StringIO
import pydotplus
import os



# Import the data
col_names = ['gameId', 'creationTime', 'gameDuration', 'seasonId', 'winner',
             'firstBlood', 'firstTower', 'firstInhibitor', 'firstBaron', 'firstDragon', 'firstRiftHerald',
             't1_towerKills', 't1_inhibitorKills', 't1_baronKills', 't1_dragonKills', 't1_riftHeraldKills',
             't2_towerKills', 't2_inhibitorKills', 't2_baronKills', 't2_dragonKills', 't2_riftHeraldKills']


# You should check the path before run the code
data = pd.read_csv("F:/各个学科/工业大数据/lab/project1/data/new_data.csv", header=None, names=col_names)
test = pd.read_csv("F:/各个学科/工业大数据/lab/project1/data/test_set.csv", header=None, names=col_names)


data = data.iloc[1:]
test = test.iloc[1:]
feature_cols = ['firstBlood', 'firstTower', 'firstInhibitor', 'firstBaron', 'firstDragon', 'firstRiftHerald',
                't1_towerKills', 't1_inhibitorKills', 't1_baronKills', 't1_dragonKills', 't1_riftHeraldKills',
                't2_towerKills', 't2_inhibitorKills', 't2_baronKills', 't2_dragonKills', 't2_riftHeraldKills']
X_train = data[feature_cols]
y_train = data.winner
X_test = test[feature_cols]
y_test = test.winner

# Create a tree
DT_clf = DecisionTreeClassifier(criterion='entropy', max_depth=7, min_samples_split=15)


# Calculate the Training time
# time_start = time.time()
DT_clf = DT_clf.fit(X_train, y_train)
# time_end = time.time()
# print('Training time: ', time_end - time_start)

# Apply the tree
y_pred = DT_clf.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))

# Visualize the decision tree
os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/Graphviz2.38/bin/'
dot_data = StringIO()
export_graphviz(DT_clf, out_file=dot_data, filled=True, rounded=True, special_characters=True, feature_names=feature_cols,
                class_names=['0', '1'])
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
graph.write_png('tree.png')
