#!/usr/bin/python

""" 
    This is the code to accompany the Lesson 2 (SVM) mini-project.

    Use a SVM to identify emails from the Enron corpus by their authors:    
    Sara has label 0
    Chris has label 1
"""
    
import sys
import time
from sklearn.metrics import accuracy_score
sys.path.append("../tools/")
from email_preprocess import preprocess
from sklearn.svm import SVC

### features_train and features_test are the features for the training
### and testing datasets, respectively
### labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess()




#########################################################
### your code goes here ###

# Lowering number of samples for faster prediction.
# features_train = features_train[:len(features_train)/100]
# labels_train = labels_train[:len(labels_train)/100]
#-----------------
clr = SVC(C=10000.0)
start_time = time.time()
clr.fit(features_train, labels_train)
print "Training time:- ",(time.time()-start_time)
pred_time = time.time()
pred = clr.predict(features_test)
print "Prediction time:- ", (time.time() - pred_time)
count = 0
for i in range(0, len(pred)):
    if pred[i] == 1:
        count += 1
print count
score = accuracy_score(pred, labels_test)
print "Accuracy is: {}%".format(str(score * 100))
#########################################################


