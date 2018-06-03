from keras.datasets import reuters
from keras.utils.np_utils import to_categorical  #keras function for one-hot encoding
from keras import layers
from keras import models
import numpy as np
import matplotlib.pyplot as plt

(train_data, train_labels), (test_data, test_labels) = reuters.load_data(num_words=10000)

# Net needs tensors... data encoding.
def vectorize_sequences(sequences, dimension=10000):
  results = np.zeros((len(sequences), dimension))     #all zero matrix of shape len(sequences), dimension)
  for i, sequence in enumerate(sequences):
    results[i, sequence] = 1.             #one-hot encoding according to the sequence. does it have the word?
  return results  

x_train = vectorize_sequences(train_data)
x_test = vectorize_sequences(test_data)

#one hot encode the label data
one_hot_train_labels = to_categorical(train_labels)
one_hot_test_labels = to_categorical(test_labels)

#Build the model
model = models.Sequential()
model.add(layers.Dense(64, activation='relu', input_shape=(10000,)))
#model.add(layers.Dense(46, activation='relu'))
model.add(layers.Dense(46, activation='softmax'))   #this outputs a vector...it's the probabliity distribution of how likely this input is each one of the classes.  Sum to 1.
#Compile the model
model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['accuracy']
)


#Validation.  Set aside 1000 data samples.
x_val = x_train[:1000]
partial_x_val = x_train[1000:]

y_val = one_hot_train_labels[:1000]
partial_y_val = one_hot_train_labels[1000:]
'''
#train the network for 20 epochs.
history = model.fit(partial_x_val,
                    partial_y_val,
                    epochs=20,
                    batch_size=512,
                    validation_data=(x_val,y_val)
                    )

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(1,len(loss) + 1)

plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and Validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.show()

This showed that the model was overfitting after 9 epochs.
'''
model.fit(partial_x_val,
          partial_y_val,
          epochs=9,
          batch_size=512,
          validation_data=(x_val,y_val)
          )

results = model.evaluate(x_test, one_hot_test_labels)
print(results)