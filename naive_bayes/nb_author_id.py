#!/usr/bin/python

""" 
    This is the code to accompany the Lesson 1 (Naive Bayes) mini-project. 

    Use a Naive Bayes Classifier to identify emails by their authors
    
    authors and labels:
    Sara has label 0
    Chris has label 1
"""
    
import sys
import time
sys.path.append("../tools/")
from email_preprocess import preprocess
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
### features_train and features_test are the features for the training
### and testing datasets, respectively
### labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess()




#########################################################
### your code goes here ###
clr = GaussianNB()
start_time = time.time()
clr.fit(features_train, labels_train)
print "Training time:- ",(time.time()-start_time)
pred_time = time.time()
pred = clr.predict(features_test)
print "Prediction time:- ", (time.time() - pred_time)
score = accuracy_score(pred, labels_test)
print "Accuracy is: {}%".format(str(score * 100))

#########################################################


