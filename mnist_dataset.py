# Objective: Create a ML system that detects and classifies hand-written digits.

# -------------------------------------------------------------------------------------------

# ! IMPORT MODULES AND PREPARE DATASET

# Import sklearn modules.
from sklearn.decomposition import TruncatedSVD
from sklearn.model_selection import cross_validate, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.manifold import TSNE

# Import other modules.
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os

print(os.getcwd())

# Import dataset.
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

# NOTE: You can download these files @: https://www.kaggle.com/oddrationale/mnist-in-csv?select=mnist_train.csv

# X is the hand written digits, y is the actual digits (answers).
X_train, y_train = (train.loc[:, '1x1':'28x28'].to_numpy(), train['label'].to_numpy())
X_test, y_test = (test.loc[:, '1x1':'28x28'].to_numpy(), test['label'].to_numpy())

# X represents the hand written digits which are 28 x 28 in size.
print(f'X_train.shape: {X_train.shape}')    # -> (60,000, 784)

# Y is the actual digits they represent.
print(f'y_train.shape: {y_train.shape}')    # -> (60,000,)

print(f'X_test.shape: {X_test.shape}')    # -> (10,000, 784)
print(f'y_test.shape: {y_test.shape}')    # -> (10,000,)

# ------------------------------------------------------------------------------------------------------

# ! VISUALIZE THE DATA

tsne = TSNE()

X_test_trans = tsne.fit_transform(X_test[:5000])

scatter = plt.scatter(X_test_trans[:, 0], X_test_trans[:, 1], c=y_test[:5000])
plt.legend(*scatter.legend_elements())

# Data is formed in clusters and looks to be non-linear.

# --------------------------------------------------------------------------------------------------

# ! DO MORE DATA ANALYSIS

# Shorten amount of samples.
data = np.concatenate((X_train[:6000], X_test[:1000]))
target = np.concatenate((y_train[:6000], y_test[:1000]))

print(data.shape)    # -> (7000, 784)
print(target.shape)    # -> (7000,)

print(data[0].shape)    # -> (784,)

# Check how plt.imshow() works with data.
sample = data[0].reshape((28, 28))

plt.imshow(sample, cmap='binary')
plt.text(25, 25, target[0], fontsize=20, color='red')

# Checking one more time if the concatenation was correct:
print(target[6000] == y_test[0])    # -> True

# Check % of data that's 0.
print(np.sum(X_train == 0)/(60000*784))    # -> 80%
print(np.sum(X_train != 0)/(60000*784))    # -> 20%

# Check for null values.
print(np.isnan(np.sum(data)))    # -> False

# ---------------------------------------------------------------------------------------------

# ! APPLY MODEL TO THE DATA

# Use SVD for decomposition with pipeline.
pipe = Pipeline([
    ('svd', TruncatedSVD(n_components=149)),
    ('svm', SVC())
])

# NOTE: SVDs are good for dimensionality reduction when data has a lot of zeros.

cross_validate_args = {
    'cv': 5,
    'n_jobs': -1,
    'verbose': 100,
    'return_train_score': True,
    'return_estimator': True
}

svm = cross_validate(pipe, data, target, **cross_validate_args)

svm_scores = {
    'train': np.average(svm['train_score']),
    'test': np.average(svm['test_score'])
}

print('Train Score: {}'.format(svm_scores['train']))
print('Test Score: {}'.format(svm_scores['test']))

# -------------------------------------------------------------------------------------------------------

# ! CHECK HOW SVD AFFECTED THE DATA.

svd = pipe.named_steps['svd']

data_trans = svd.fit_transform(data)

print(data_trans.shape)    # -> (60000, 149)

zeros_before = np.sum(data == 0)/(data.size)
zeros_after = np.sum(data_trans == 0)/(data_trans.size)

nonzeros_before = np.sum(data != 0)/(data.size)
nonzeros_after = np.sum(data_trans != 0)/(data_trans.size)

print(f'% of Zeros in Data: Before SVD: {zeros_before:.2f}, After SVD: {zeros_after}')
print(f'% of Non Zero in Data: Before SVD: {nonzeros_before:.2f}, After SVD: {nonzeros_after}')

# -------------------------------------------------------------------------------------------------------------

# ! VISUALIZE THE RESULTS

# Checking how digits dataset displays images.
from sklearn.datasets import load_digits

dig_data = load_digits().data
dig_img = load_digits().images

print(images.shape)
print(data.shape)

# NOTE: Found out that images variable is 3 dimensional!

# Making data 3D so that imshow() can read data correctly as an img.

# NOTE: The reshaping of the array is to make the sample the shape of a sample.

'''

Ex: reshape(1, -1)

returns a np array with a column of 784 data points.

'''

cross_validate_args_pred = {
    'cv': 5,
    'n_jobs': -1,
    'verbose': 100,
}

fig, axs = plt.subplots(3, 3, subplot_kw={'yticks': (), 'xticks': ()}, figsize=(12.5, 12.5))

predictions = cross_val_predict(pipe, X_train, y_train, **cross_validate_args_pred)[:9]
axs = [ax for ax in axs.ravel()]
data_imgs = data[:9].reshape(9, 28, 28)

for ax, prediction, img in zip(axs, predictions, data_imgs):
    ax.imshow(img, cmap='binary')
    ax.text(23, 25, prediction, fontsize=11, color='red')

plt.show()
