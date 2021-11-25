# southeast-rentprice
This is an analysis and modelling of the rent price data in the SouthEase region United States.

## Installation
 - import numpy as np
 - import pandas as pd
 - from scipy import stats
 - import matplotlib.pyplot as plt
 - import seaborn as sns
 - from sklearn.pipeline import Pipeline
 - from sklearn.linear_model import RidgeCV
 - from sklearn.model_selection import cross_validate
 - from sklearn.compose import ColumnTransformer
 - from sklearn.preprocessing import OneHotEncoder
 - from sklearn.preprocessing import StandardScaler
 - from sklearn.model_selection import train_test_split
 - from sklearn import ensemble
 - from sklearn.metrics import mean_squared_error
 - import joblib

## Project Motivation
My project goal is to build a model for predicting rent price based on listings dataset in SouthEast states in the United States.
I am co-working with my coleagues to create a dashboard that returns the predicted rent price based on user's input. 

## File Description
 - KM_exploring.ipynb: A Jupyter Notebook I initially explored the dataset and come up with data cleaning and feature engineering
 - se_model.ipynb: Store the best prediction model I found and stored joblib files needed to work on .py file.
 - southeast.py: A module (SouthEast) I created for transforming a raw datapioint and predict the price based on pre-trained model
 - preprocessor.joblib: A joblib file that contains fitted preprocessor based on se_model
 - (se-model.joblib): Not listed in this repository due to the large file size. A joblib file that stores the best random forest model.
