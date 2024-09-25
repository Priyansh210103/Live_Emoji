import os
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras import Input, Model
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import to_categorical

is_init = False
size = -1

label = []
dictionary = {}
c = 0

# Load data
for i in os.listdir():
    if i.split(".")[-1] == "npy" and not (i.split(".")[0] == "labels"):
        if not is_init:
            is_init = True
            X = np.load(i)
            size = X.shape[0]
            y = np.array([i.split('.')[0]] * size).reshape(-1, 1)
        else:
            X = np.concatenate((X, np.load(i)))
            y = np.concatenate((y, np.array([i.split('.')[0]] * size).reshape(-1, 1)))

        label.append(i.split('.')[0])
        dictionary[i.split('.')[0]] = c
        c += 1

# Convert labels to numeric format
for i in range(y.shape[0]):
    y[i, 0] = dictionary[y[i, 0]]

# Convert labels to categorical format (one-hot encoding)
y = to_categorical(y)

# Shuffle the data
X_new = X.copy()
y_new = y.copy()
counter = 0

cnt = np.arange(X.shape[0])
np.random.shuffle(cnt)

for i in cnt:
    X_new[counter] = X[i]
    y_new[counter] = y[i]
    counter += 1

# Define the model architecture
input_shape = X.shape[1]  # Ensure this is the correct input shape
ip = Input(shape=(input_shape,))  # Corrected the shape definition

m = Dense(512, activation="relu")(ip)
m = Dense(256, activation="relu")(m)

# Correct output layer size based on the number of classes (y.shape[1])
op = Dense(y.shape[1], activation="softmax")(m)

model = Model(inputs=ip, outputs=op)

# Compile the model
model.compile(optimizer='rmsprop', loss="categorical_crossentropy", metrics=['acc'])

# Fit the model
model.fit(X_new, y_new, epochs=50)

model.save("model.h5")
np.save("labels.npy",np.array(label))
