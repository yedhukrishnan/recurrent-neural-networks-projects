# Sentiment analysis (good or bad) from text

import numpy as np
import csv
import re

from keras.models import Model, load_model
from keras.layers import Dense, Input, Dropout, LSTM, Activation
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence
from keras.initializers import glorot_uniform

def convert_to_one_hot(Y, C):
    Y_hot = np.zeros((Y.shape[0], C))
    for index, val in enumerate(Y):
        Y_hot[index, val[0]] = 1
    return Y_hot

def good_or_bad(y):
    if y[0] == 1:
        return 'Bad'
    else:
        return 'Good'

x = []
y = []
train_data = [line.strip().split('\t') for line in open('amazon_cells_labelled.txt').readlines()]
train_data = np.array(train_data)
x = train_data[:, 0]
y = train_data[:, 1].astype(np.uint8)

x_train, y_train = x[:950], convert_to_one_hot(y[:950].reshape((950, 1)), C = 2)
x_test,  y_test  = x[950:], convert_to_one_hot(y[950:].reshape((50, 1)), C = 2)


max_len = len(max(x_train, key=len).split())

def read_glove_vecs(glove_file):
    with open(glove_file, 'r') as f:
        words = set()
        word_to_vec_map = {}
        for line in f:
            line = line.strip().split()
            curr_word = line[0]
            words.add(curr_word)
            word_to_vec_map[curr_word] = np.array(line[1:], dtype=np.float64)

        i = 1
        words_to_index = {}
        index_to_words = {}
        for w in sorted(words):
            words_to_index[w] = i
            index_to_words[i] = w
            i = i + 1
    return words_to_index, index_to_words, word_to_vec_map

word_to_index, index_to_word, word_to_vec_map = read_glove_vecs('glove.6B.50d.txt')



def sentences_to_indices(X, word_to_index, max_len):
    m = X.shape[0]
    X_indices = np.zeros((m, max_len))
    for i in range(m):
        sentence_words = X[i].lower().split()
        j = 0
        for w in sentence_words:
            w = re.sub(r"[^a-z]","", w.lower())
            try:
                X_indices[i, j] = word_to_index[w]
            except:
                # INACCURATE: Just for testing
                X_indices[i, j] = word_to_index['a']
            j = j + 1
    return X_indices

def pretrained_embedding_layer(word_to_vec_map, word_to_index):
    vocab_len = len(word_to_index) + 1
    emb_dim = word_to_vec_map["cucumber"].shape[0]
    emb_matrix = np.zeros((vocab_len, emb_dim))
    for word, index in word_to_index.items():
        emb_matrix[index, :] = word_to_vec_map[word]
    embedding_layer = Embedding(input_dim=vocab_len, output_dim=emb_dim, trainable = False)
    embedding_layer.build((None, ))
    embedding_layer.set_weights([emb_matrix])
    return embedding_layer

def sentiment_analyzer(input_shape, word_to_vec_map, word_to_index):
    sentence_indices = Input(shape=input_shape, dtype='int32')
    embedding_layer = pretrained_embedding_layer(word_to_vec_map, word_to_index)
    embeddings = embedding_layer(sentence_indices)
    X = LSTM(128, return_sequences=True)(embeddings)
    X = Dropout(0.5)(X)
    X = LSTM(128, return_sequences=False)(X)
    X = Dropout(0.5)(X)
    X = Dense(2)(X)
    X = Activation('softmax')(X)
    model = Model(inputs=sentence_indices, outputs=X)
    return model

model_available = False

def train_model():
    if model_available:
        model = load_model('sentiment_model.h5')
    else:
        model = sentiment_analyzer((max_len,), word_to_vec_map, word_to_index)
        model.summary()
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        x_train_indices = sentences_to_indices(x_train, word_to_index, max_len)
        model.fit(x_train_indices, y_train, epochs = 50, batch_size = 32, shuffle=True)
        model.save('sentiment_model.h5')
    return model


model = train_model()

x_test_indices = sentences_to_indices(x_test, word_to_index, max_len = max_len)
loss, acc = model.evaluate(x_test_indices, y_test)
print()
print("Test accuracy = ", acc)


x_test_indices = sentences_to_indices(x_test, word_to_index, max_len)
pred = model.predict(x_test_indices)
for i in range(len(x_test)):
    x = x_test_indices
    print(x_test[i] + "\n Expected: " + good_or_bad(y_test[i]) + ", Prediction: " + good_or_bad(np.round(pred[i])))


def predict_this(x_test):
    x_test = np.array([x_test])
    x_test_indices = sentences_to_indices(x_test, word_to_index, max_len)
    print(model.predict(x_test_indices))
    print(good_or_bad(np.round(model.predict(x_test_indices)[0])))

predict_this('Value for money')
predict_this('No value for money')
