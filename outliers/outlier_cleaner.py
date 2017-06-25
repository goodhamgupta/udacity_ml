#!/usr/bin/python
import numpy as np
import itertools

def outlierCleaner(predictions, ages, net_worths):
    """
        Clean away the 10% of points that have the largest
        residual errors (difference between the prediction
        and the actual net worth).

        Return a list of tuples named cleaned_data where
        each tuple is of the form (age, net_worth, error).
    """
    diff = []
    predicted_sum, actual_sum = 0.0,0.0
    for a,b,c in zip(predictions, ages, net_worths):
        predicted_sum += a
        actual_sum += c
        diff.append((b, c, abs(a - c)))
    cleaned_data = sorted(diff, key=lambda x: x[-1], reverse=True)
    return cleaned_data[10:]
