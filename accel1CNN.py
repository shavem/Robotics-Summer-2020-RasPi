# import tensorflow as tf
# from tensorflow.keras import Sequential
# from tensorflow.keras.layers import Flatten, Dense, Dropout, BatchNormalization
# from tensorflow.keras.layers import Conv2D, MaxPool2D
# from tensorflow.keras.optimizers import Adam
# print(tf.__version__)

import pandas as pd
import numpy as np
# import matplotlib.pyplot as plt
import csv
from os import listdir
# import scipy.stats as stats
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import StandardScaler, LabelEncoder

#
# def load_dataset(directory):
#     accValues = []
#     for filename in listdir(directory):
#         if filename.endswith("csv"):
#             csvdata = csv.reader(open(filename, "r"), delimiter = ",") # opens training data
#             accValues.append(list(csvdata))
#     return accValues
#
# subjects = load_dataset(".")
# columns = ["time", "x", "y", "z", "label"]
# classes = ["still", "flip"]
#
# datasets = []
# for i in range(0, len(subjects)):
#     datasets.append(pd.DataFrame(data=subjects[i], columns=columns))


xyzdatasets= []
for filename in listdir("."):
    if filename.endswith("csv"):
        csvxyzdata = []
        for line in csv.reader(open(filename, "r"), delimiter = ","):
            csvxyzdata.append([line[1], line[2], line[3]])
        xyzdatasets.append(csvxyzdata)


print(xyzdatasets)