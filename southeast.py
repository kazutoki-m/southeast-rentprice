import numpy as np
import joblib
import os

import pandas as pd
from scipy import stats
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn import ensemble

class SouthEast:

  def __init__(self):
    self.centroid = np.array([32.4969853, -84.45705188]) # Marion County, Georgia
    # TODO: self.model = joblib.load(os.path.join('..', 'assets', 'models', 'example.joblib'))
    self.model = joblib.load('se-model.joblib')
    self.preprocessor_ohe = joblib.load('preprocessor.joblib')

  def transform(self, datapoint):
    """
    Applies a transformation to the input datapoint so that the
    model can make a prediction using it.
    
    Parameters:
    -----------
    
      datapoint : dict
        A dictionary with the following keys and types:
          'lat': [float], # example [1.0]
          'long': [float], # example [2.0]
          'beds': [float], # example [1.0]
          'baths': [float], # example [2.0]
          'sqfeet': [float], # example [1000.0]
          'state': [str], # example ['ca']
          'type': [str], # example ['apartment']
          'laundry_options': [str], # example ['nan']
          'parking_options': [str], # example ['nan']
          'cats_allowed': [bool], # example [True] 
          'dogs_allowed': [bool], # example [False]
          'smoking_allowed': [bool], # example [False]
          'wheelchair_access': [bool], # example [True] 
          'electric_vehicle_charge': [bool], # example [False] 
          'comes_furnished': [bool], # example [True]
    
    Returns:
    --------

      point : Any
        A transformed datapoint.

    """
    df = pd.DataFrame(datapoint)

    '''(b) Create a column of pets_allowed'''
    temp = df[['cats_allowed', 'dogs_allowed']].apply(sum, axis=1).item()
    if temp == 0:
      pets_allowed = 0
    else:
      pets_allowed = 1
    df['pets_allowed'] = pets_allowed

    '''(c) Get room_dummies'''
    # room_type = ['apartment', 'house', 'condo', 'duplex', 'manufactured',
    #              'townhouse', 'loft', 'cottage/cabin', 'in-law', 'flat']
    # room_dummy = OneHotEncoder(categories='auto').fit(np.array(room_type).reshape(-1, 1))
    # room_dummies = room_dummy.transform(np.array(df['type']).reshape(-1, 1))

    '''(d) Create a park_dummy'''
    park_2 = ['attached garage']
    park_0 = ['off-street parking']
    if df['parking_options'].item() in park_2:
      park_dummy = 'two'
    elif df['parking_options'].item() in park_0:
      park_dummy = 'zero'
    else:
      park_dummy = 'one'
    df['park_dummy'] = park_dummy

    '''(e) Create a laundry_dummy'''
    in_unit = ['w/d in unit']
    in_building = ['laundry on site', 'w/d hookups', 'laundry in bldg']
    no_laundry = ['no laundry on site']
    if df['laundry_options'].item() in in_unit:
      laundry_dummy = 'in_unit'
    elif df['laundry_options'].item() in no_laundry:
      laundry_dummy = 'no_laundry'
    else:
      laundry_dummy = 'in_building'
    df['laundry_dummy'] = laundry_dummy

    '''(f) Get state dummy'''
    # states = ['fl', 'ga', 'tn', 'sc', 'al', 'la', 'ky', 'ms', 'nc', 'il', 'in']
    # state_dummy = OneHotEncoder(categories='auto').fit(np.array(states).reshape(-1, 1))
    # state_dummies = state_dummy.transform(np.array(df['state']).reshape(-1, 1))

    '''(g) Convert boolean columns into zero-one'''
    bool_cols = ['smoking_allowed', 'wheelchair_access', 'electric_vehicle_charge', 'comes_furnished',
                 'comes_furnished']
    bool_vars = df[bool_cols]
    df[bool_cols] = df[bool_cols].astype(int)

    '''(h) Select only columns for modeling'''
    df_processed = df.copy()[
      ['type', 'pets_allowed', 'smoking_allowed', 'wheelchair_access', 'electric_vehicle_charge',
       'comes_furnished', 'laundry_dummy', 'park_dummy', 'sqfeet', 'beds', 'baths', 'lat', 'long', 'state']]

    cat_cols = ['type', 'laundry_dummy', 'state']
    con_cols = ['sqfeet', 'beds', 'baths']
    point = self.preprocessor_ohe.transform(df_processed)

    return point.toarray()

  def predict(self, datapoint):
    """
    Returns a prediction for the input datapoint.
    
    Parameters:
    -----------
    
      datapoint : dict
        A dictionary with the following keys and types:
          'lat': [float], # example [1.0]
          'long': [float], # example [2.0]
          'beds': [float], # example [1.0]
          'baths': [float], # example [2.0]
          'sqfeet': [float], # example [1000.0]
          'state': [str], # example ['ca']
          'type': [str], # example ['apartment']
          'laundry_options': [str], # example ['nan']
          'parking_options': [str], # example ['nan']
          'cats_allowed': [bool], # example [True] 
          'dogs_allowed': [bool], # example [False]
          'smoking_allowed': [bool], # example [False]
          'wheelchair_access': [bool], # example [True] 
          'electric_vehicle_charge': [bool], # example [False] 
          'comes_furnished': [bool], # example [True]
    
    Returns:
    --------

      price : float
        A price prediction given the input datapoint.

    """
    # TODO: return self.model.predict(self.transform(datapoint))
    price = self.model.predict(self.transform(datapoint))
    return price
