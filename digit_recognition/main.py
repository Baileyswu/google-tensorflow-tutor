from __future__ import print_function

import glob
import math
import os

from IPython import display
from matplotlib import cm
from matplotlib import gridspec
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn import metrics
import tensorflow as tf
from tensorflow.python.data import Dataset

tf.logging.set_verbosity(tf.logging.ERROR)
pd.options.display.max_rows = 10
pd.options.display.float_format = '{:.1f}'.format

mnist_dataframe = pd.read_csv(
  "https://download.mlcc.google.cn/mledu-datasets/mnist_train_small.csv",
  sep=",",
  header=None)

# Use just the first 10,000 records for training/validation.
mnist_dataframe = mnist_dataframe.head(10000)
mnist_dataframe = mnist_dataframe.reindex(np.random.permutation(mnist_dataframe.index))

from preprocess import parse_labels_and_features
training_targets, training_examples = parse_labels_and_features(mnist_dataframe[:7500])
validation_targets, validation_examples = parse_labels_and_features(mnist_dataframe[7500:10000])

# rand_example = np.random.choice(training_examples.index)
# _, ax = plt.subplots()
# ax.matshow(training_examples.loc[rand_example].values.reshape(28, 28)) # loc[行] loc[行，列]
# ax.set_title("Label: %i" % training_targets.loc[rand_example])
# ax.grid(False)

from train import train_linear_classification_model
classifier = train_linear_classification_model(
             learning_rate=0.02,
             steps=100,
             batch_size=10,
             training_examples=training_examples,
             training_targets=training_targets,
             validation_examples=validation_examples,
             validation_targets=validation_targets)