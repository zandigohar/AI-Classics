import csv

import matplotlib.pyplot as plt

import numpy as np

from sklearn import datasets, metrics
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

names = ["Nearest Neighbors", "Linear SVM", "Gaussian Process",
         "Decision Tree", "Random Forest", "Neural Net", "AdaBoost",
         "Naive Bayes", "QDA"]

classifier = [
    KNeighborsClassifier(3),
    SVC(kernel="linear", C=0.025),
    GaussianProcessClassifier(1.0 * RBF(1.0)),
    DecisionTreeClassifier(max_depth=5),
    RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
    MLPClassifier(alpha=1, max_iter=1000),
    AdaBoostClassifier(),
    GaussianNB(),
    QuadraticDiscriminantAnalysis()]


data_all=[]
with open('data.csv', newline='') as file:
    reader = csv.reader(file)
    for row in reader:
        data_all.append(row)


m=len(data_all)
n=len(data_all[0])

data=[]
data2=[]
res=[]
for i in range(m):
    data=[]
    for j in range(n-1):
        if j==1 and i!=0:
            if data_all[i][j]=='M': # BAD
                res.append(1)
            else:
                res.append(0)
        elif i>0 and j>1:
            data.append(float(data_all[i][j]))
    if data:
        data2.append(data)



X = np.asarray(data2)
y = np.asarray(res)
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.20)


expected = y_test
for i in range(len(classifier)):
    classifier[i].fit(X_train, y_train)
    predicted = classifier[i].predict(X_test)
    accuracy=accuracy_score(expected,predicted)
    print('Accuracy of %s classifier: %f %%' % (names[i],100*accuracy))


print('____________________________________________________________')


forest = ExtraTreesClassifier(n_estimators=250,random_state=0)
forest.fit(X, y)
importances = forest.feature_importances_
std = np.std([tree.feature_importances_ for tree in forest.estimators_],
             axis=0)
indices = np.argsort(importances)[::-1]

features=[]
for i in indices:
    features.append(data_all[0][2+i])

# Print the feature ranking
print("Feature ranking:")

for f in range(X.shape[1]):
    print("%d. %s feature %d (%f)" % (f + 1, features[indices[f]], indices[f], importances[indices[f]]))

# Plot the feature importances of the forest
plt.figure()
plt.title("Feature importances")
plt.bar(range(X.shape[1]), importances[indices],
       color="r", yerr=std[indices], align="center")
plt.xticks(range(X.shape[1]), indices)
plt.xlim([-1, X.shape[1]])
plt.show()
