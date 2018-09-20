#
# Example from Keras documentation
#
model = Sequential()
model.add(Embedding(1000, 64, input_length=10))
# the model will take as input an integer matrix of size (batch, input_length).
# the largest integer (i.e. word index) in the input should be no larger than 999 (vocabulary size).
# now model.output_shape == (None, 10, 64), where None is the batch dimension.

input_array = np.random.randint(1000, size=(32, 10))

model.compile('rmsprop', 'mse')
output_array = model.predict(input_array)
assert output_array.shape == (32, 10, 64)

#
# Simpler example
#
emb_matrix = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7,0.8,0.9]])
input_array = np.array([[2, 1], [1, 1], [1, 1]])

model = Sequential()
model.add(Embedding(3, 3))
model.set_weights([emb_matrix])
output_array = model.predict(input_array)

# output_array = 
# array([[[0.7, 0.8, 0.9], [0.4, 0.5, 0.6]],
#        [[0.4, 0.5, 0.6], [0.4, 0.5, 0.6]],
#        [[0.4, 0.5, 0.6], [0.4, 0.5, 0.6]]], dtype=float32)
