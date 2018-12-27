#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 26 14:02:37 2018

@author: genevievemeloche
"""

import math

from IPython import display
from matplotlib import cm
from matplotlib import gridspec
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from sklearn import metrics
import tensorflow as tf
from tensorflow.python.data import Dataset

tf.logging.set_verbosity(tf.logging.ERROR)
pd.options.display.max_rows = 10
pd.options.display.float_format = '{:.1f}'.format

pd.set_option('display.max_columns', None)

black_friday_dataframe = pd.read_csv('BlackFriday.csv')

print(black_friday_dataframe.columns.tolist())
print(black_friday_dataframe.describe())

black_friday_dataframe = black_friday_dataframe.reindex(
    np.random.permutation(black_friday_dataframe.index))


def preprocess_features(black_friday_dataframe):

  selected_features = black_friday_dataframe[['Gender', 'Age', 
                                              'Occupation', 'City_Category', 
                                              'Stay_In_Current_City_Years', 
                                              'Marital_Status', 
                                              'Product_Category_1',
                                              'Product_Category_2',
                                              'Product_Category_3',
                                              'Purchase']]
  processed_features = selected_features.copy()
  # Create a synthetic feature.
  processed_features["num_gender"] =  processed_features["Gender"].apply((lambda x: 1 if x == 'M' else '0'))
  processed_features["num_gender"] = processed_features["num_gender"].astype('int32')
  print(processed_features["num_gender"])

  return processed_features

def preprocess_targets(black_friday_dataframe):

  output_targets = pd.DataFrame()
  # Scale the target to be in units of thousands of dollars.
  output_targets["Purchase"] = (
    black_friday_dataframe["Purchase"] / 100.0)
  return output_targets



# Choose the first 12000 (out of 17000) examples for training.
training_examples = preprocess_features(black_friday_dataframe.head(10000))
training_targets = preprocess_targets(black_friday_dataframe.head(10000))

# Choose the last 5000 (out of 17000) examples for validation.
validation_examples = preprocess_features(black_friday_dataframe.tail(10000))
validation_targets = preprocess_targets(black_friday_dataframe.tail(10000))


# Double-check that we've done the right thing.
print("Training examples summary:")
display.display(training_examples.describe())
print("Validation examples summary:")
display.display(validation_examples.describe())


print("Training targets summary:")
display.display(training_targets.describe())
print("Validation targets summary:")
display.display(validation_targets.describe())

#Check correlation
correlation_dataframe = training_examples.copy()
correlation_dataframe["target"] = training_targets["Purchase"]

correlation_dataframe.corr()



def construct_feature_columns(input_features):
  """Construct the TensorFlow Feature Columns.

  Args:
    input_features: The names of the numerical input features to use.
  Returns:
    A set of feature columns
  """
  return set([tf.feature_column.numeric_column(my_feature)
              for my_feature in input_features])
      
def my_input_fn(features, targets, batch_size=1, shuffle=True, num_epochs=None):
    """Trains a linear regression model.
  
    Args:
      features: pandas DataFrame of features
      targets: pandas DataFrame of targets
      batch_size: Size of batches to be passed to the model
      shuffle: True or False. Whether to shuffle the data.
      num_epochs: Number of epochs for which data should be repeated. None = repeat indefinitely
    Returns:
      Tuple of (features, labels) for next data batch
    """
    
    # Convert pandas data into a dict of np arrays.
    features = {key:np.array(value) for key,value in dict(features).items()}                                           
 
    # Construct a dataset, and configure batching/repeating.
    ds = Dataset.from_tensor_slices((features,targets)) # warning: 2GB limit
    ds = ds.batch(batch_size).repeat(num_epochs)
    
    # Shuffle the data, if specified.
    if shuffle:
      ds = ds.shuffle(10000)
    
    # Return the next batch of data.
    features, labels = ds.make_one_shot_iterator().get_next()
    return features, labels

def train_model(
    learning_rate,
    steps,
    batch_size,
    feature_columns,
    training_examples,
    training_targets,
    validation_examples,
    validation_targets):
  """Trains a linear regression model.
  
  In addition to training, this function also prints training progress information,
  as well as a plot of the training and validation loss over time.
  
  Args:
    learning_rate: A `float`, the learning rate.
    steps: A non-zero `int`, the total number of training steps. A training step
      consists of a forward and backward pass using a single batch.
    feature_columns: A `set` specifying the input feature columns to use.
    training_examples: A `DataFrame` containing one or more columns from
      `california_housing_dataframe` to use as input features for training.
    training_targets: A `DataFrame` containing exactly one column from
      `california_housing_dataframe` to use as target for training.
    validation_examples: A `DataFrame` containing one or more columns from
      `california_housing_dataframe` to use as input features for validation.
    validation_targets: A `DataFrame` containing exactly one column from
      `california_housing_dataframe` to use as target for validation.
      
  Returns:
    A `LinearRegressor` object trained on the training data.
  """

  periods = 3
  steps_per_period = steps / periods

  # Create a linear regressor object.
  my_optimizer = tf.train.FtrlOptimizer(learning_rate=learning_rate)
  my_optimizer = tf.contrib.estimator.clip_gradients_by_norm(my_optimizer, 5.0)
  linear_regressor = tf.estimator.LinearRegressor(
      feature_columns=feature_columns,
      optimizer=my_optimizer
  )
  
  training_input_fn = lambda: my_input_fn(training_examples, 
                                          training_targets["Purchase"], 
                                          batch_size=batch_size)
  predict_training_input_fn = lambda: my_input_fn(training_examples, 
                                                  training_targets["Purchase"], 
                                                  num_epochs=1, 
                                                  shuffle=False)
  predict_validation_input_fn = lambda: my_input_fn(validation_examples, 
                                                    validation_targets["Purchase"], 
                                                    num_epochs=1, 
                                                    shuffle=False)

  # Train the model, but do so inside a loop so that we can periodically assess
  # loss metrics.
  print("Training model...")
  print("RMSE (on training data):")
  training_rmse = []
  validation_rmse = []
  for period in range (0, periods):
    # Train the model, starting from the prior state.
    linear_regressor.train(
        input_fn=training_input_fn,
        steps=steps_per_period
    )
    # Take a break and compute predictions.
    training_predictions = linear_regressor.predict(input_fn=predict_training_input_fn)
    training_predictions = np.array([item['predictions'][0] for item in training_predictions])
    validation_predictions = linear_regressor.predict(input_fn=predict_validation_input_fn)
    validation_predictions = np.array([item['predictions'][0] for item in validation_predictions])
    
    # Compute training and validation loss.
    training_root_mean_squared_error = math.sqrt(
        metrics.mean_squared_error(training_predictions, training_targets))
    validation_root_mean_squared_error = math.sqrt(
        metrics.mean_squared_error(validation_predictions, validation_targets))
    # Occasionally print the current loss.
    print("  period %02d : %0.2f" % (period, training_root_mean_squared_error))
    # Add the loss metrics from this period to our list.
    training_rmse.append(training_root_mean_squared_error)
    validation_rmse.append(validation_root_mean_squared_error)
  print("Model training finished.")

  
  # Output a graph of loss metrics over periods.
  plt.ylabel("RMSE")
  plt.xlabel("Periods")
  plt.title("Root Mean Squared Error vs. Periods")
  plt.tight_layout()
  plt.plot(training_rmse, label="training")
  plt.plot(validation_rmse, label="validation")
  plt.legend()

  return linear_regressor


"""
Feature sets
"""


def construct_feature_columns():
  """Construct the TensorFlow Feature Columns.

  Returns:
    A set of feature columns
  """ 
  occupation = tf.feature_column.numeric_column("Occupation")
  marital_status = tf.feature_column.numeric_column("Marital_Status")
  marital_status = tf.feature_column.numeric_column("num_gender")
  
  
  feature_columns = set([
    occupation,
    marital_status])
  
  return feature_columns

#_ = train_model(
#    learning_rate=1.0,
#    steps=500,
#    batch_size=100,
#    feature_columns=construct_feature_columns(),
#    training_examples=training_examples,
#    training_targets=training_targets,
#    validation_examples=validation_examples,
#    validation_targets=validation_targets)
#
#plt.subplot(1, 2, 2)
#_ = black_friday_dataframe["Occupation"].hist()
#_ = black_friday_dataframe["Marital_Status"].hist()
#_ = black_friday_dataframe["num_gender"].hist()
#_ = black_friday_dataframe["Purchase"].hist()

