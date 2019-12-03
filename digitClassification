# Import datasets, classifiers and performance metrics
from sklearn import datasets, metrics
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import Perceptron

# The digits dataset
digits = datasets.load_digits()
n_samples = len(digits.data)
data=digits.data
X = digits.data / digits.data.max()
y = digits.target

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.20, random_state=10)

# Create a classifier: GaussianNB and Perceptron
GaussianNB_classifier = GaussianNB()
Perception_classifier = Perceptron()
Perception_classifier2 = Perceptron(penalty='l1')
Perception_classifier3 = Perceptron(penalty='l2')
Perception_classifier4 = Perceptron(penalty='elasticnet')

# Learn the digits on the train dataset:
GaussianNB_classifier.fit(X_train, y_train)
Perception_classifier.fit(X_train, y_train)
Perception_classifier2.fit(X_train, y_train)
Perception_classifier3.fit(X_train, y_train)
Perception_classifier4.fit(X_train, y_train)

# Predict the value of the digit on the test Dataset:
expected = y_test
GaussianNB_predicted = GaussianNB_classifier.predict(X_test)
Perception_predicted = Perception_classifier.predict(X_test)
Perception_predicted2 = Perception_classifier2.predict(X_test)
Perception_predicted3 = Perception_classifier3.predict(X_test)
Perception_predicted4 = Perception_classifier4.predict(X_test)

errGNB=100*(1-accuracy_score(expected,GaussianNB_predicted))
err=100*(1-accuracy_score(expected,Perception_predicted))
err2=100*(1-accuracy_score(expected,Perception_predicted2))
err3=100*(1-accuracy_score(expected,Perception_predicted3))
err4=100*(1-accuracy_score(expected,Perception_predicted4))

print('Error of GaussianNB classifier: ',errGNB,'%')
print('Error of None penalty perception classifier: ',err,'%')
print('Error of l1 perception classifier: ',err2,'%','--> Best Performing')
print('Error of l2 perception classifier: ',err3,'%')
print('Error of elasticnet perception classifier: ',err4,'%')
print('____________________________________________________________')


Perception_classifier21 = Perceptron(penalty='l1',alpha=0.001)
Perception_classifier22 = Perceptron(penalty='l1',alpha=0.0001)
Perception_classifier23 = Perceptron(penalty='l1',alpha=0.01)
Perception_classifier24 = Perceptron(penalty='l1',alpha=0.00001)

Perception_classifier21.fit(X_train, y_train)
Perception_classifier22.fit(X_train, y_train)
Perception_classifier23.fit(X_train, y_train)
Perception_classifier24.fit(X_train, y_train)

Perception_predicted21 = Perception_classifier21.predict(X_test)
Perception_predicted22 = Perception_classifier22.predict(X_test)
Perception_predicted23 = Perception_classifier23.predict(X_test)
Perception_predicted24 = Perception_classifier24.predict(X_test)


err21=100*(1-accuracy_score(expected,Perception_predicted21))
err22=100*(1-accuracy_score(expected,Perception_predicted22))
err23=100*(1-accuracy_score(expected,Perception_predicted23))
err24=100*(1-accuracy_score(expected,Perception_predicted24))
print('Error of l1 perception classifier with alpha=0.001: ',err21,'%')
print('Error of l1 perception classifier with alpha=0.0001: ',err22,'%')
print('Error of l1 perception classifier with alpha=0.01: ',err23,'%')
print('Error of l1 perception classifier with alpha=0.00001: ',err24,'%','--> Best Performing')
print('____________________________________________________________')


print("Classification report for GaussianNB_classifier %s:\n%s\n"
      % (GaussianNB_classifier, metrics.classification_report(expected, GaussianNB_predicted)))
print("Confusion matrix:\n%s" % metrics.confusion_matrix(expected, GaussianNB_predicted))


print('____________________________________________________________')
print("Classification report for Perception_classifier %s:\n%s\n"
      % (Perception_classifier24, metrics.classification_report(expected, Perception_predicted24)))
print("Confusion matrix:\n%s" % metrics.confusion_matrix(expected, Perception_predicted24))
