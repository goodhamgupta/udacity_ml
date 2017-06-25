#!/usr/bin/python

import pickle
import sys
import matplotlib.pyplot
sys.path.append("../tools/")
from feature_format import featureFormat, targetFeatureSplit


### read in data dictionary, convert to numpy array
data_dict = pickle.load( open("../final_project/final_project_dataset.pkl", "r") )
features = ["salary", "bonus"]
data_dict.pop('TOTAL', 0)
data = featureFormat(data_dict, features)

max_x = 0.0
max_y = 0.0
for point in data:
    salary = point[0]
    bonus = point[1]
    if salary > max_x:
        max_x = salary
    if bonus > max_y:
        may_y = bonus
    matplotlib.pyplot.scatter( salary, bonus )

matplotlib.pyplot.xlabel("salary")
matplotlib.pyplot.ylabel("bonus")
matplotlib.pyplot.show()
print max_x
print "\n"
print max_y
keys = data_dict.keys()
for i in keys:
    sal = data_dict[i]['salary']
    bon = data_dict[i]['bonus']
    if (float(sal) > 1000000.0 and sal != 'NaN') or (float(bon) > 500000.0 and bon != 'NaN'):
        print(i)
        print(data_dict[i])
        print "******\n"
    else:
        continue
