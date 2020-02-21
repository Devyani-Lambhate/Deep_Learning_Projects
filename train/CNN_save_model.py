from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.preprocessing.image import ImageDataGenerator 
from tensorflow.keras.models import Sequential 
from tensorflow.keras.layers import Conv2D, MaxPooling2D 
from tensorflow.keras.layers import Activation, Dropout, Flatten, Dense 
from keras.utils import to_categorical
import numpy as np
from matplotlib import pyplot as plt
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.optimizers import SGD

(xtrain,ytrain),(xtest,ytest)=fashion_mnist.load_data()
#print(xtrain.shape)

xtrain = xtrain.astype('float32')
xtest = xtest.astype('float32')
xtrain /= 255
xtest /= 255

num_classes=10

xtrain = xtrain.reshape(xtrain.shape[0], 28, 28, 1)
xtest = xtest.reshape(xtest.shape[0], 28, 28, 1)

model=Sequential()
model.add(Conv2D(32,activation='relu', kernel_size=(3,3),kernel_initializer='he_uniform', input_shape=(28,28,1)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(BatchNormalization())
model.add(Conv2D(64,activation='relu',kernel_size=(3,3),kernel_initializer='he_uniform'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(BatchNormalization())
model.add(Flatten()) # Flattening the 2D arrays for fully connected layers
#model.add(Dense(128, activation='relu'))
model.add(Dense(100, activation='relu', kernel_initializer='he_uniform'))
model.add(Dense(10,activation='softmax'))
print(model.summary())


model.compile(loss='sparse_categorical_crossentropy', optimizer='adam',metrics=['accuracy'])
history=model.fit(xtrain,ytrain,validation_split=0.33,epochs=20)

loss, accuracy = model.evaluate(xtest, ytest, verbose=0)
print('Accuracy: %f' % (accuracy))
print('Loss: %f' % (loss))

# summarize history for accuracy
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()
model.summary()
model.save('fashion_mnist_cnn.h5')
