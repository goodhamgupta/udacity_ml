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

import pickle
import sys
sys.path.append("/home/lp-153/shubham/udacity_ml/ud120-projects/final_project")
from poi_email_addresses import poiEmails

enron_data = pickle.load(open("../final_project/final_project_dataset.pkl", "r"))
address_list = poiEmails()
valid_email_count = 0
valid_salary_count = 0
for obj in enron_data.iteritems():
    x, y = obj
    if y.get("email_address") != "NaN":
        valid_email_count += 1
    if y.get("salary") != "NaN":
        valid_salary_count += 1

print valid_email_count
print valid_salary_count
