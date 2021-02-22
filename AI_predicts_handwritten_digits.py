# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
from IPython import get_ipython

# %% [markdown]
# ## Objective - Create a ML system that detects and classifies hand-written digits
# %% [markdown]
# #### Import modules and prepare dataset

# %%
# Import sklearn/tensorflow modules.
import tensorflow as tf
from sklearn.linear_model import Perceptron
from sklearn.manifold import TSNE
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.datasets import mnist
get_ipython().run_line_magic('matplotlib', 'inline')

# Import other modules.
import matplotlib.pyplot as plt
import numpy as np
from time import time
from plyer import notification
from random import choice
from IPython.display import clear_output

# NOTE: To import 'plyer' module, you must use 'pip install plyer'

# Import dataset.
(X_train, y_train), (X_test, y_test) = mnist.load_data()

X_train = X_train.reshape(X_train.shape[0], -1)
X_test = X_test.reshape(X_test.shape[0], -1)

# X represents the hand written digits which are 28 x 28 in size.
print(f'X_train.shape: {X_train.shape}')    # -> (60,000, 784)

# Y is the actual digits they represent.
print(f'y_train.shape: {y_train.shape}')    # -> (60,000,)

print(f'X_test.shape: {X_test.shape}')    # -> (10,000, 784)
print(f'y_test.shape: {y_test.shape}')    # -> (10,000,)

# %% [markdown]
# #### Visualize the data using TSNE dimensionality technique

# %%
# Initializing data reduction algorithm
tsne = TSNE()

# Change the data accordingly to the algorithm.
X_test_trans = tsne.fit_transform(X_test[:2500])

# Scatter the data.
scatter = plt.scatter(X_test_trans[:, 0], X_test_trans[:, 1], c=y_test[:2500])
plt.legend(*scatter.legend_elements())
plt.show()

# Data is formed in clusters and looks to be linearly seperable.

# %% [markdown]
# #### Data analysis

# %%
# Concatenating both train and test datasets.
data = np.concatenate((X_train, X_test))
target = np.concatenate((y_train, y_test))

# Check % of data that's 0.
percent_of_zeros = np.sum(data == 0)/data.size    # -> 80%
percent_of_non_zeros = np.sum(data != 0)/data.size    # -> 20%

# Check for null values.
check_null = np.isnan(np.sum(data))    # -> False

# Create scaler.
scaler = MinMaxScaler()

# NOTE: MinMaxScaler() is used to scaler the data within a range of 0 to 1 relative to other values in the data.
#       The reason why this is used is to make the data easier to process by the model.

# Keep sample for comparison.
sample = X_train[0]

# Scale the data.
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Compare the two samples and their values
print('Before data transformation: {}'.format(sample[np.where(sample != 0)][:5]))
print('After data transformation:  {}'.format(X_train[np.where(X_train != 0)][:5]))

# Change to categorical data.
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

# Change data to image data.
X_train = X_train.reshape((60000, 28, 28, 1)).astype('float32')
X_test = X_test.reshape((10000, 28, 28, 1)).astype('float32')

# %% [markdown]
# #### Apply linear model to the data

# %%
# Creating simple linear model to learn more about data.
perceptron = Perceptron(max_iter=99999999)    # Perceptron only works when data is linear.

# NOTE: Perceptron is a neural network with only one hidden-layer.

# Create function that reforms the data for the algorithm.
def reform_sklearn(*arrays):
    
    reformed_arrays = []

    for array in arrays:

        # Array is a data variable.
        if array.ndim != 2:
            reformed_arrays.append(array.reshape(len(array), -1))
    
        # Array is a target variable
        else:
            reformed_arrays.append(np.array([np.argmax(sample) for sample in array]))
        
    return reformed_arrays

per_X_train, per_y_train, per_X_test, per_y_test = reform_sklearn(X_train, y_train, X_test, y_test)

# NOTE: perceptrons are not good for image processing,
#       meaning the data must be 1D, not 2D.

start_time = time()

perceptron.fit(per_X_train, per_y_train)

print(f'Perceptron train score: {perceptron.score(per_X_train, per_y_train)}')
print(f'Perceptron test score: {perceptron.score(per_X_test, per_y_test)}')
print(f'Iterations used: {perceptron.n_iter_}')
print(f'Elapsed: {(time() - start_time)/60:.2f} min.')

# NOTE: Based on the results on the perceptron and that the model used little iterations/epochs
#       to fit the data, we can assume the data is linearly seperable.

# %% [markdown]
# #### Apply model to the data

# %%
# Create convolutional neural network.

# NOTE: Convolutional neural networks (CNNs) are used as a machine learning model for processing images.
#       This algorithm goes thorugh different steps of processing the images in different ways,
#       eventually finding a pattern in the data.

def model():
    
    model = Sequential()
    model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', input_shape=(28, 28, 1)))
    model.add(MaxPooling2D((2, 2)))
    model.add(Flatten())
    model.add(Dense(100, activation='relu', kernel_initializer='he_uniform'))
    model.add(Dense(10, activation='softmax'))
	
    # Compile model.
    opt = SGD(lr=0.01, momentum=0.9)
    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# NOTE: Sequential() models allow you to make models layer-by-layer and is much simpler compared to Input() models.
#       Conv2D() is an input layer for converting an image in to a matrix.
#       MaxPooling2D() is another input layer.
#       Flatten() converts multi-dimensional data into a single vector to be processed. Ex: 28 x 28 -> 784
#       Dense() is the main hidden layer.
#       SGD() is the stochastic gradient descent optimization function to update weights.
#       relu activation function is a nonlinear function that is good for learning complex functions.
#       softmax activation function is used to quantify the output in classification examples.

# Create timer.
start_time = time()

model = model()
results = model.fit(X_train, y_train, epochs=5, batch_size=128).history

# Clear the output.
clear_output()

# NOTE: batch_size param means that the model will be tested on 64 samples at a time.
#       This save a ton of RAM during the training process.

print(train_score := 'CNN Train Score: {:.2f}'.format(model.evaluate(X_train, y_train, verbose=False)[1]*100))
print(test_score := 'CNN Test Score: {:.2f}'.format(model.evaluate(X_test, y_test, verbose=False)[1]*100))
print(elapsed := f'Elapsed: {(time() - start_time)/60:.2f} min.')

# Notify when done.
notification.notify(
    title='Neural Network Training Results',
    message=f'{train_score}\n{test_score}\n{elapsed}',
    app_icon='python_icon.ico'
)

plt.plot(range(5), results['loss'])
plt.title('Training Loss')

# %% [markdown]
# #### Test the model through visualizations

# %%
# Ask user for specific number to test model.
def pick_num(num):
    try:
        number = int(num)
    except ValueError:
        pick_num(input('Choose a number to test the model on -> '))
    else:
        return number

# Call function
number = pick_num(input('Choose a number to test the model on -> '))

# Revert the target data from to_categorical()
og_y_test = np.array([np.argmax(y, axis=None, out=None) for y in y_test])

# Get the random index for the sample and the sample target.
index = choice(np.where(og_y_test == number)[0])
sample = X_test[index].reshape(1, 28, 28, 1)
sample_target = og_y_test[index]

prediction = np.argmax(model.predict(sample))

sample_img = sample.reshape(28, 28)

plt.imshow(sample_img, cmap='binary')
plt.text(18, 24, f'Prediction: {prediction}', fontsize=11, color='red');
plt.text(18, 26, f'Target: {sample_target}', fontsize=11, color='red');


