import math
import numpy as np
import csv
import os
import matplotlib as mpl
import matplotlib.pyplot as plt
from PIL import Image

def create_img_vector(start, arr):
    image_vector = np.zeros(shape=(28, 280))
    for i in range (0, 10):
        image_vector[:, i*28:((i*28)+28)] = arr[start + i].reshape(28, 28)
    return Image.fromarray(image_vector)

dir_path = os.path.dirname(os.path.realpath(__file__))

with open(dir_path + '\\train.csv', 'r', newline='') as mnist_train_file:
    mnist_train_data = list(csv.reader(mnist_train_file))

mnist_data = np.array(mnist_train_data[1:], dtype=int)
mnist_training = np.array(mnist_data[0:800, 1:])
mnist_training_labels = np.array(mnist_data[0:800, 0])
mnist_testing = np.array(mnist_data[800:1000, 1:])
mnist_testing_labels = np.array(mnist_data[800:1000, 0])

avg_images = [[] for i in range(10)]
for i in range(100):
    idx = mnist_training_labels[i]
    avg_images[idx].append(mnist_training[i])

np_avgs = [np.zeros(784) for i in range(10)]
for i in range(len(avg_images)):
    for j in range(len(avg_images[i])):
        np_avgs[i] = np.add(np_avgs[i], np.array(avg_images[i][j]))
    np_avgs[i] = np.divide(np_avgs[i], len(avg_images[i]))

x_vect = create_img_vector(0, np_avgs)
x_vect.show()

img_vect = create_img_vector(20, mnist_training)
img_vect.show()