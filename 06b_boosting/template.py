# Author: Huldar
# Date: 2024-10-21
# Project: Assignment 6b
# Notes: Point is to learn about sklearn and random forests
# Acknowledgements: 
#

# Imports
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import (train_test_split, RandomizedSearchCV)
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (confusion_matrix, accuracy_score, recall_score, precision_score)

from tools import get_titanic, build_kaggle_submission


def get_better_titanic():
    '''
    Loads the cleaned titanic dataset but change
    how we handle the age column.
    '''
    ...


def rfc_train_test(X_train, t_train, X_test, t_test):
    '''
    Train a random forest classifier on (X_train, t_train)
    and evaluate it on (X_test, t_test)
    '''
    ...


def gb_train_test(X_train, t_train, X_test, t_test):
    '''
    Train a Gradient boosting classifier on (X_train, t_train)
    and evaluate it on (X_test, t_test)
    '''
    ...


def param_search(X, y):
    '''
    Perform randomized parameter search on the
    gradient boosting classifier on the dataset (X, y)
    '''
    # Create the parameter grid
    gb_param_grid = {
        'n_estimators': [...],
        'max_depth': [...],
        'learning_rate': [...]}
    # Instantiate the regressor
    gb = GradientBoostingClassifier()
    # Perform random search
    gb_random = RandomizedSearchCV(
        param_distributions=gb_param_grid,
        estimator=gb,
        scoring="accuracy",
        verbose=0,
        n_iter=50,
        cv=4)
    # Fit randomized_mse to the data
    gb_random.fit(X, y)
    # Print the best parameters and lowest RMSE
    return gb_random.best_params_


def gb_optimized_train_test(X_train, t_train, X_test, t_test):
    '''
    Train a gradient boosting classifier on (X_train, t_train)
    and evaluate it on (X_test, t_test) with
    your own optimized parameters
    '''
    ...


def _create_submission():
    '''Create your kaggle submission
    '''
    ...
    prediction = ...
    build_kaggle_submission(prediction)
    
    
    
# Testing area
#----------------------------------------------------------------------------------------------    
if __name__ == "__main__":
    """
    You can test your code inside this scope without having to comment it out
    everytime before submitting. It also makes it easier for you to
    know what is going on in your code.
    """
 
    # Section 2.5
    
    # Section 2.6
    
    # Section 3 - Get best classifier and submit to kaggle, might be optional, but you can see how you would have faired in the competition
    # Section 3.1
    
    
    
    # Confirmation message for a succesful run
    print("\n---------------------------------------------------------------\nRun succesful :)\n")
    print(("Estimated {:.2f} points out of " + str(n_sections) + "\nGrade estimate: {:.2f}\n").format(n_sections_correct, 10*n_sections_correct/n_sections))

'''
    if str() == str() :
        print("Pass")
        n_sections_correct = n_sections_correct + 1
    else:
        print("Fail")
'''