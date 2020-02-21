from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.preprocessing.image import ImageDataGenerator 
from tensorflow.keras.models import Sequential 
from tensorflow.keras.layers import Conv2D, MaxPooling2D 
from tensorflow.keras.layers import Activation, Dropout, Flatten, Dense 
from keras.utils import to_categorical
import numpy as np
from matplotlib import pyplot as plt

(xtrain,ytrain),(xtest,ytest)=fashion_mnist.load_data()
#print(xtrain.shape)

xtrain = xtrain.astype('float32')
xtest = xtest.astype('float32')
xtrain /= 255
xtest /= 255

num_classes=10

model=Sequential()
model.add(Dense(512,input_shape=(28,28)))
model.add(Flatten())
model.add(Activation('relu'))

model.add(Dense(512))
model.add(Activation('relu'))

model.add(Dense(10))
model.add(Activation('sigmoid'))

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
model.save('fashion_mnist_fully_conn.h5')
