# Objective: Create a ML system that detects and classifies hand-written digits.

# -------------------------------------------------------------------------------------------

# ! IMPORT MODULES AND PREPARE DATASET

# Import sklearn modules.
from sklearn.decomposition import TruncatedSVD
from sklearn.model_selection import cross_validate, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import Perceptron
from sklearn.manifold import TSNE
from sklearn.datasets import load_digits
from sklearn.preprocessing import StandardScaler
import pygame
from tkinter import *

# Import other modules.
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os

# Getting workspace/project to create a path that leads to the dataset.
print(os.getcwd())

# Import dataset.
train_path = os.path.join('Projects/AI Predicts Hand-Written Digits', 'train.csv')
test_path = os.path.join('Projects/AI Predicts Hand-Written Digits', 'test.csv')

train = pd.read_csv(train_path)
test = pd.read_csv(test_path)

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
plt.show()

# Data is formed in clusters and looks to be linearly seperable.

# --------------------------------------------------------------------------------------------------

# ! DO MORE DATA ANALYSIS

# Shorten amount of samples.
data = np.concatenate((X_train := X_train[:30000], X_test := X_test[:5000]))
target = np.concatenate((y_train := y_train[:30000], y_test := y_test[:5000]))

print(data.shape)    # -> (7000, 784)
print(target.shape)    # -> (7000,)
print(data.size)    # -> 5,488,000

# NOTE: The amount of features (784) is not ideal because it will
#       take lots of computational power.

print(data[0].shape)    # -> (784,)

# Check how plt.imshow() works with data.
sample = data[0].reshape((28, 28))

plt.imshow(sample, cmap='binary')
plt.text(25, 25, target[0], fontsize=20, color='red')
plt.show()

# Checking one more time if the concatenation was correct:
print(target[6000] == y_test[0])    # -> True

# Check % of data that's 0.
print(np.sum(X_train == 0)/(60000*784))    # -> 80%
print(np.sum(X_train != 0)/(60000*784))    # -> 20%

# Check for null values.
print(np.isnan(np.sum(data)))    # -> False

# --------------------------------------------------------------------

# ! APPLY LINEAR MODEL TO THE DATA (double checking if data is linear)

# Use SVD for decomposition with pipeline.
pipe_lin = Pipeline([
    ('svd', TruncatedSVD(n_components=149)),
    ('lin_svm', Perceptron())    # Preceptron models only work if data is linear.
])

# NOTE: SVDs are good for dimensionality reduction when data has a lot of zeros.

pipe_lin.fit(X_train, y_train)

print(pipe_lin.score(X_train, y_train))
print(pipe_lin.score(X_test, y_test))

# ---------------------------------------------------------------------------------------------

# ! APPLY MODEL TO THE DATA

mlp = Pipeline([
    ('svd', TruncatedSVD(n_components=149)),
    ('scaler', StandardScaler()),
    ('mlp', MLPClassifier(
        verbose=True,
        random_state=42,
        hidden_layer_sizes=(200,),
        alpha=0.1,
    ))
])

mlp.fit(X_train, y_train)

print(mlp.score(X_train, y_train))
print(mlp.score(X_test, y_test))

# -------------------------------------------------------------------------------------------------------

# ! CHECK HOW SVD AFFECTED THE DATA.

svd = mlp.named_steps['svd']

data_trans = svd.fit_transform(data)

print(data_trans.shape)    # -> (60000, 149)

zeros_before = np.sum(data == 0)/data.size
zeros_after = np.sum(data_trans == 0)/data_trans.size

nonzeros_before = np.sum(data != 0)/data.size
nonzeros_after = np.sum(data_trans != 0)/data_trans.size

print(f'% of Zeros in Data: Before SVD: {zeros_before:.2f}, After SVD: {zeros_after}')
print(f'% of Non Zero in Data: Before SVD: {nonzeros_before:.2f}, After SVD: {nonzeros_after}')

# -------------------------------------------------------------------------------------------------------------

# ! VISUALIZE THE RESULTS

# Checking how digits dataset displays images.
dig_data = load_digits().data
dig_img = load_digits().images

print(dig_data.shape)
print(dig_img.shape)

# NOTE: Found out that images variable is 3 dimensional!

# Making data 3D so that imshow() can read data correctly as an img.

# NOTE: The reshaping of the array is to make the sample the shape of a sample.

'''

Ex: reshape(1, -1)

returns a np array with a column of 784 data points.

'''

fig, axs = plt.subplots(3, 3, subplot_kw={'yticks': (), 'xticks': ()}, figsize=(12.5, 12.5))

predictions = mlp.predict(X_test[9:18])
axs = [ax for ax in axs.ravel()]
data_imgs = X_test[9:18].reshape(9, 28, 28)

for ax, prediction, img in zip(axs, predictions, data_imgs):
    ax.imshow(img, cmap='binary')
    ax.text(23, 25, prediction, fontsize=11, color='red')

plt.show()

# ---------------------------------------------------------------------------------------------------------------------

# ! MAKE YOUR OWN SAMPLES

model = mlp.named_steps['mlp'].fit(X_train, y_train)

# Create user plot mechanism using pygame.
class Pixel(object):
    def __init__(self, x, y, width, height):
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        self.color = (255, 255, 255)
        self.neighbors = []

    def draw(self, surface):
        pygame.draw.rect(surface, self.color, (self.x, self.y, self.x + self.width, self.y + self.height))

    def get_neighbor(self, g):
        # Get the neighbors of each pixel in the grid to create thicker lines.
        j = self.x // 20
        i = self.y // 20
        rows = 28
        cols = 28

        if i < cols - 1:  # Right
            self.neighbors.append(g.pixels[i + 1][j])
        if i > 0:  # Left
            self.neighbors.append(g.pixels[i - 1][j])
        if j < rows - 1:  # Up
            self.neighbors.append(g.pixels[i][j + 1])
        if j > 0:  # Down
            self.neighbors.append(g.pixels[i][j - 1])

        # Diagonal neighbors
        if j > 0 and i > 0:  # Top Left
            self.neighbors.append(g.pixels[i - 1][j - 1])

        if j + 1 < rows and i > -1 and i - 1 > 0:  # Bottom Left
            self.neighbors.append(g.pixels[i - 1][j + 1])

        if j - 1 < rows and i < cols - 1 and j - 1 > 0:  # Top Right
            self.neighbors.append(g.pixels[i + 1][j - 1])

        if j < rows - 1 and i < cols - 1:  # Bottom Right
            self.neighbors.append(g.pixels[i + 1][j + 1])


class Grid(object):
    pixels = []

    def __init__(self, row, col, width, height):
        self.rows = row
        self.cols = col
        self.len = row * col
        self.width = width
        self.height = height
        self.generate_pixels()
        pass

    def draw(self, surface):
        for row in self.pixels:
            for col in row:
                col.draw(surface)

    def generate_pixels(self):
        x_gap = self.width // self.cols
        y_gap = self.height // self.rows
        self.pixels = []
        for r in range(self.rows):
            self.pixels.append([])
            for c in range(self.cols):
                self.pixels[r].append(Pixel(x_gap * c, y_gap * r, x_gap, y_gap))

        for r in range(self.rows):
            for c in range(self.cols):
                self.pixels[r][c].get_neighbor(self)

    def clicked(self, pos):    # Return the position in the grid that user clicked on
        try:
            t = pos[0]
            w = pos[1]
            g1 = int(t) // self.pixels[0][0].width
            g1 = int(t) // self.pixels[0][0].width
            g2 = int(w) // self.pixels[0][0].height

            return self.pixels[g2][g1]
        except:
            pass

    def convert_binary(self):
        li = self.pixels

        newMatrix = [[] for x in range(len(li))]

        for i in range(len(li)):
            for j in range(len(li[i])):
                if li[i][j].color == (255, 255, 255):
                    newMatrix[i].append(0)
                else:
                    newMatrix[i].append(175)

        return np.array(newMatrix).reshape(1, -1)


def guess(li):
    prediction = model.predict(li)[0]
    print("I predict this number is a:", prediction)
    print('Shape of sample: ', li.shape)
    window = Tk()
    window.withdraw()
    window.destroy()
    #plt.imshow(li[0], cmap=plt.cm.binary)
    #plt.show()


def main():
    run = True

    while run:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run = False
            if event.type == pygame.KEYDOWN:
                li = g.convert_binary()
                guess(li)
                g.generate_pixels()
            if pygame.mouse.get_pressed()[0]:

                pos = pygame.mouse.get_pos()
                clicked = g.clicked(pos)
                clicked.color = (0, 0, 0)
                for n in clicked.neighbors:
                    n.color = (0, 0, 0)

            if pygame.mouse.get_pressed()[2]:
                try:
                    pos = pygame.mouse.get_pos()
                    clicked = g.clicked(pos)
                    clicked.color = (255, 255, 255)
                except:
                    pass

        g.draw(win)
        pygame.display.update()

pygame.init()
width = height = 560
win = pygame.display.set_mode((width, height))
pygame.display.set_caption("Number Guesser")
g = Grid(28, 28, width, height)
main()


pygame.quit()
quit()
