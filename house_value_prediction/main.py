# coding: utf-8
from __future__ import print_function

import math

from IPython import display
from matplotlib import cm
from matplotlib import gridspec
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from sklearn import metrics
import tensorflow as tf

from train import train_model
from preprocess import preprocess_features, preprocess_targets

tf.logging.set_verbosity(tf.logging.ERROR)
pd.options.display.max_rows = 10
pd.options.display.float_format = '{:.1f}'.format
from sklearn.utils import shuffle

california_housing_dataframe = pd.read_csv("https://download.mlcc.google.cn/mledu-datasets/california_housing_train.csv", sep=",")

california_housing_dataframe = california_housing_dataframe.reindex(
    np.random.permutation(california_housing_dataframe.index))

training_examples = preprocess_features(california_housing_dataframe.head(12000))
training_targets = preprocess_targets(california_housing_dataframe.head(12000))
validation_examples = preprocess_features(california_housing_dataframe.tail(5000))
validation_targets = preprocess_targets(california_housing_dataframe.tail(5000))

# -----------------------------------------------
# train using validation

# linear_regressor = train_model(
#     learning_rate=0.00003,
#     steps=500,
#     batch_size=5,
#     training_examples=training_examples,
#     training_targets=training_targets,
#     validation_examples=validation_examples,
#     validation_targets=validation_targets)

# -----------------------------------------------
# multiple features

# minimal_features = [
#     "latitude",
#     "median_income",
#     "rooms_per_person"
# ]

# assert minimal_features, "You must select at least one feature!"

# minimal_training_examples = training_examples[minimal_features]
# minimal_validation_examples = validation_examples[minimal_features]

# train_model(
#     learning_rate=0.001,
#     steps=500,
#     batch_size=5,
#     training_examples=minimal_training_examples,
#     training_targets=training_targets,
#     validation_examples=minimal_validation_examples,
#     validation_targets=validation_targets)

# -----------------------------------------------
# one_hot

# import feature_combine
# from feature_combine import select_and_transform_features
# selected_training_examples = select_and_transform_features(training_examples)
# selected_validation_examples = select_and_transform_features(validation_examples)

# _ = train_model(
#     learning_rate=0.01,
#     steps=500,
#     batch_size=5,
#     training_examples=selected_training_examples,
#     training_targets=training_targets,
#     validation_examples=selected_validation_examples,
#     validation_targets=validation_targets)


# -----------------------------------------------
# buckets
import FTRL
from FTRL import construct_feature_columns
FTRL.train_model(
    learning_rate=1.0,
    steps=500,
    batch_size=100,
    feature_columns=construct_feature_columns(training_examples),
    training_examples=training_examples,
    training_targets=training_targets,
    validation_examples=validation_examples,
    validation_targets=validation_targets)