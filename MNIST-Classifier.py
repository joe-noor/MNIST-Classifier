from numpy import mean
from numpy import std
from matplotlib import pyplot
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from keras.datasets import mnist
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Dense
from keras.layers import Flatten
from keras.optimizers import SGD
from sklearn.model_selection import train_test_split
 
 #loads data
(trainX, trainY), (testX, testY) = mnist.load_data()

# reshape dataset to have a single channel
trainX = trainX.reshape((60000, 28, 28, 1)).astype('float32') / 255.0
testX = testX.reshape((10000, 28, 28, 1)).astype('float32') / 255.0

# creates 10 element binary vector
trainY = to_categorical(trainY)
testY = to_categorical(testY)

# splits the train and test arrays
trainX, X_val, trainY, Y_val = train_test_split(trainX, trainY, test_size=0.2, random_state=1)

# builds model
model = Sequential()
model.add(Conv2D(28, kernel_size = (3, 3), input_shape = (28,28,1), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(56, kernel_size = (3, 3), input_shape = (28,28,1), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Flatten())
model.add(Dense(100, activation='relu', kernel_initializer='he_uniform'))
model.add(Dense(10, activation='softmax'))

# compile model
opt = SGD(lr=0.007, momentum=0.903)
model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
model.evaluate(testX, testY)
history = model.fit(trainX, trainY, epochs=30, batch_size=50, validation_data=(X_val, Y_val))

# graphs model
loss_train = history.history['loss']
loss_val = history.history['val_loss']
epochs = range(1,31)
plt.plot(epochs, loss_train, 'g', label='Training loss')
plt.plot(epochs, loss_val, 'b', label='validation loss')
plt.title('Training and Validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()
