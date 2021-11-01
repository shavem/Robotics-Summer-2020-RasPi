import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import csv
from os import listdir

model = tf.keras.models.load_model('./Model')
# Check its architecture
model.summary()