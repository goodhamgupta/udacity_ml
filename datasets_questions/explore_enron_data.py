#!/usr/bin/python

"""
    Starter code for exploring the Enron dataset (emails + finances);
    loads up the dataset (pickled dict of dicts).

    The dataset has the form:
    enron_data["LASTNAME FIRSTNAME MIDDLEINITIAL"] = { features_dict }

    {features_dict} is a dictionary of features associated with that person.
    You should explore features_dict as part of the mini-project,
    but here's an example to get you started:

    enron_data["SKILLING JEFFREY K"]["bonus"] = 5600000

"""

import cPickle
import sys
sys.path.append("/Users/shubhamgupta/shubham/udacity_ml/final_project")
from poi_email_addresses import poiEmails
import pdb
enron_data = cPickle.load(open("../final_project/final_project_dataset.pkl", "r"))
address_list = poiEmails()
valid_email_count = 0
valid_salary_count = 0
valid_poi = 0
for obj in enron_data.iteritems():
    x, y = obj
    if y.get("email_address") in address_list:
        valid_poi += 1
    if y.get("email_address") != "NaN":
        valid_email_count += 1
    if y.get("salary") != "NaN":
        valid_salary_count += 1

print(len(enron_data))
print(valid_poi)
print(valid_email_count)
print(valid_salary_count)

from sklearn.linear_model import LinearRegression