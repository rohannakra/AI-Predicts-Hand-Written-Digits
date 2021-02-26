# Begin cited code.
import tensorflow as tf    # https://www.tensorflow.org/
from sklearn.linear_model import Perceptron    # https://bit.ly/3pWvxTz
from sklearn.manifold import TSNE    # https://bit.ly/3stGnlr
from sklearn.preprocessing import MinMaxScaler    # https://bit.ly/2NGYyFG
from tensorflow.keras.models import Sequential    # https://www.tensorflow.org/api_docs/python/tf/keras/Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten    # https://bit.ly/2NJ1kdp
from tensorflow.keras.optimizers import SGD    # https://www.tensorflow.org/api_docs/python/tf/keras/optimizers/SGD
from tensorflow.keras.utils import to_categorical    # https://bit.ly/2O6ySC1
from tensorflow.keras.datasets import mnist    # https://www.tensorflow.org/api_docs/python/tf/keras/datasets/mnist
get_ipython().run_line_magic('matplotlib', 'inline')    # https://ipython.readthedocs.io/en/stable/interactive/plotting.html

# Import other modules.
import matplotlib.pyplot as plt    # https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.html
import numpy as np    # https://numpy.org/
from time import time    # https://docs.python.org/3/library/time.html
from plyer import notification    # https://github.com/kivy/plyer/blob/master/plyer/facades/notification.py
from random import choice    # https://docs.python.org/3/library/random.html
from IPython.display import clear_output    # https://bit.ly/3svyQml

# End of cited code.

# NOTE: To import 'plyer' module, you must use 'pip install plyer'
