from keras.datasets import imdb
import numpy as np

(train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=10000)

x_train = vectorize_sequences(train_data)
x_test = vectorize_sequences(test_data)

y_train = np.asarray(train_labels).astyoe('float32')  # Vectorize training data. matrix.
y_test = np.asarray(test_labels).astype('float32')    # Vectorize LABELS

# Data now ready to be fed into a network...create this network!

# Net needs tensors...
def vectorize_sequences(sequences, dimension=10000):
  results = np.zeros((len(sequences), dimension))     #all zero matrix of shape len(sequences), dimension)
  for i, sequence in enumerate(sequences):
    results[i, sequence] = 1.             #one-hot encoding according to the sequence. does it have the word?
  return results  