import math
import numpy as np
import csv
import os
import matplotlib as mpl
import matplotlib.pyplot as plt
from PIL import Image

def scale_feature(feature):
    column_vector = np.array(feature)
    column_vector = np.subtract(column_vector, min(column_vector))
    denominator = max(column_vector) - min(column_vector)
    if denominator != 0:
        return column_vector / denominator
    else:
        return np.zeros(len(column_vector))

def normalize(feature):
    column_vector = np.array(feature)
    column_vector = np.delete(column_vector, np.argwhere(column_vector == 0))
    vector_mean = np.mean(column_vector)
    return np.where(feature == 0, vector_mean, feature)

dir_path = os.path.dirname(os.path.realpath(__file__))

with open(dir_path + '\\diabetes.csv', 'r', newline='') as dia_file:
    diabetes_file = list(csv.reader(dia_file))

diabetes_data = np.array(diabetes_file[1:], dtype=float)
diabetes_header = np.array(diabetes_file[0][:len(diabetes_file[0])-1])
diabetes_labels = np.array(diabetes_data[:, -1])
diabetes_data = np.delete(diabetes_data, len(diabetes_data[0])-1, axis=1)
diabetes_training = np.array(diabetes_data[int(len(diabetes_data)*0.2):])
dia_training_labels = np.array(diabetes_labels[int(len(diabetes_data)*0.2):])
diabetes_testing = np.array(diabetes_data[:int(len(diabetes_data)*0.2)])
dia_testing_labels = np.array(diabetes_labels[:int(len(diabetes_data)*0.2)])

for i in range(1, len(diabetes_training[0])):
    diabetes_training[:, i] = normalize(diabetes_training[:, i])
    diabetes_testing[:, i] = normalize(diabetes_testing[:, i])

for i in range(len(diabetes_training[0])):
    diabetes_training[:, i] = scale_feature(diabetes_training[:, i])
    diabetes_testing[:, i] = scale_feature(diabetes_testing[:, i])

marker_colors = ['g' if diabetes_labels[i] == 1 else 'r' for i in range(len(diabetes_labels))]

pregnancies_vector = diabetes_data[:, 0]
glucose_vector = normalize(diabetes_data[:, 1])

plt.scatter(pregnancies_vector, glucose_vector, alpha = 0.3, color=marker_colors)
plt.title("Glucose vs. Pregnancies")
plt.xlabel("Pregnancies")
plt.ylabel("Glucose")
plt.xticks(ticks=[0, 5, 10, 15, 20])
plt.show()

age_vector = diabetes_data[:, -1]
bmi_vector = normalize(diabetes_data[:, -3])

plt.scatter(bmi_vector, age_vector, alpha = 0.3, color=marker_colors)
plt.title("BMI vs. Age")
plt.xlabel("BMI")
plt.ylabel("Age")
plt.show()

plt.hist(diabetes_data[:, 0], alpha=0.3)
plt.title("Age Distribution")
plt.xlabel("Age Ranges")
plt.show()

data_present = [] * 7
num_zeroes = [] * 7

for i in range(1, 8):
    data_present.append(np.count_nonzero(diabetes_data[:, i]))
    num_zeroes.append(len(diabetes_data[:, 0]) - np.count_nonzero(diabetes_data[:, i]))

plt.bar(x=[i for i in range(1, 8)], height=num_zeroes, tick_label=diabetes_header[1:8])
plt.title("Missing Data by Feature")
plt.show()