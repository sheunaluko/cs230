import util as u
import tensorflow as tf
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten

xs,ys = u.get_dataset()

#create model
model = Sequential()
#add model layers
model.add(Conv2D(64, kernel_size=3, activation='relu', input_shape=(512, 512, 3)))
model.add(Conv2D(32, kernel_size=3, activation='relu'))
model.add(Flatten())
model.add(Dense(4, activation = 'sigmoid'))

# https://towardsdatascience.com/building-a-convolutional-neural-network-cnn-in-keras-329fbbadc5f5


#compile model using accuracy to measure model performance
model.compile(optimizer='adam', loss='mean_squared_error', metrics=['accuracy'])

X_train  = xs[:30]
y_train = ys[:30]; y_train = y_train.reshape(30,4)/512
X_test = xs[30:39]
y_test = ys[30:39]; y_test = y_test.reshape(9,4)/512


model.fit(X_train, y_train, validation_data=(X_test, y_test), batch_size=1, epochs= 10)
