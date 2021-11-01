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
from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import StandardScaler, LabelEncoder


# xyzdatasets= []
# for filename in listdir("."):
#     if filename.endswith("csv"):
#         csvxyzdata = []
#         for line in csv.reader(open(filename, "r"), delimiter = ","):
#             csvxyzdata.append([line[1], line[2], line[3]])
#         xyzdatasets.append(csvxyzdata)
#
#
# print(xyzdatasets)
def load_datasets():
    subjects = list()
    for filename in listdir('.'):
        if filename.endswith("csv"):
            values = csv.reader(open(filename, "r"), delimiter = ",") # opens training data
            processedlist = []
            for row in values:
                temp = [row[0],row[1],row[2],row[3],row[4]]
                processedlist.append(temp)
            subjects.append(processedlist)
    return subjects

subjects = load_datasets()

columns = ["time", "x", "y", "z", "label"]
classes = ["still", "flip"]

datasets = []
for i in range(0,len(subjects)):
    datasets.append(pd.DataFrame(data = subjects[i], columns = columns))

def get_frames(df):
    frames = []
    labels = []
    for dataset in df:
        frame = []
        for i in range(0,len(dataset)):
            x = dataset['x'][i]
            y = dataset['y'][i]
            z = dataset['z'][i]

            frame.append([int(x), int(y), int(z)])
        frames.append(frame)
        labels.append(int(dataset["label"][0]))
    frames = np.asarray(frames)
    lables = np.asarray(frames)
    return frames, labels

X, Y = get_frames(datasets)
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=4)
X_train = X_train.reshape(len(X_train), 1000, 3, 1)
X_test = X_test.reshape(len(X_test), 1000, 3, 1)