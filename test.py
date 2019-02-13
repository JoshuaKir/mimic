import tensorflow as tf
tf.enable_eager_execution()

import numpy as np
import os
import time

def letter_to_int(letter, vocab):
	i = 0
	while (i < len(vocab)):
		if (vocab[i] == letter):
			return int(i)
		i = i+1
	return -1

def int_to_letter(num, vocab):
	return vocab[int(num)]

def entire_text_to_int(text, vocab):
  int_list = []
  print("turning text to integers")
  i = 0
  while (i < len(text)):
    int_list.append(letter_to_int((text[i]), vocab))
    i+=1
    #print(i)
  int_array = np.array(int_list)
  print(int_array)
  return int_array

def entire_int_to_text(num_array, vocab):
  text_list = []
  i=0
  print("turning numbers to text")
  while (i < len(num_array)):
  	text_list.append(vocab[int(num_array[i])])
  	i+=1
  	#print(i)
  text_array = np.array(text_list)
  return text_array

def split_input_target(chunk):
    input_text = chunk[:-1]
    target_text = chunk[1:]
    return input_text, target_text

def build_model(vocab_size, embedding_dim, rnn_units, batch_size):
  model = tf.keras.Sequential([
    tf.keras.layers.Embedding(vocab_size, embedding_dim, batch_input_shape=[batch_size, None]),
    rnn(rnn_units,
        return_sequences=True, 
        recurrent_initializer='glorot_uniform',
        stateful=True),
    tf.keras.layers.Dense(vocab_size)
  ])
  return model

def loss(labels, logits):
  return tf.keras.losses.sparse_categorical_crossentropy(labels, logits, from_logits=True)

#path_to_file = './jp.txt'
path_to_file = './the_literal_entire_script_of_seinfeld.txt'
text = open(path_to_file, 'rb').read().decode(encoding='utf-8')
print('Length of text: {} characters'.format(len(text)))
print(text[:250])

vocab = sorted(set(text))
#maybe create a list of unwanted characters and filter them out of vocab like  for a sctipt
print ('{} unique characters'.format(len(vocab)))

print(vocab)

#percent = (letter_to_int('', vocab))
#index = int_to_letter(percent, vocab)

meme = entire_text_to_int(text, vocab)
print(meme)
#meme2 = entire_int_to_text(meme, vocab)
#print(meme2)

# The maximum length sentence we want for a single input in characters
seq_length = 100
examples_per_epoch = len(text)//seq_length

# Create training examples / targets
char_dataset = tf.data.Dataset.from_tensor_slices(meme)

for i in char_dataset.take(5):
  print(int_to_letter(i, vocab))

sequences = char_dataset.batch(seq_length+1, drop_remainder=True)

dataset = sequences.map(split_input_target)
print(dataset)
for input_example, target_example in  dataset.take(1):
	for number in input_example:
		print(int_to_letter(number, vocab))
	for other_number in target_example:
		print(int_to_letter(other_number, vocab))
# Batch size 
BATCH_SIZE = 64
steps_per_epoch = examples_per_epoch//BATCH_SIZE

# Buffer size to shuffle the dataset
# (TF data is designed to work with possibly infinite sequences, 
# so it doesn't attempt to shuffle the entire sequence in memory. Instead, 
# it maintains a buffer in which it shuffles elements).
BUFFER_SIZE = 10000

dataset = dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE, drop_remainder=True)
print(dataset)

# Length of the vocabulary in chars
vocab_size = len(vocab)

# The embedding dimension 
embedding_dim = 256

# Number of RNN units
rnn_units = 1024

if tf.test.is_gpu_available():
  rnn = tf.keras.layers.CuDNNGRU
else:
  import functools
  rnn = functools.partial(tf.keras.layers.GRU, recurrent_activation='sigmoid')

model = build_model(
  vocab_size = len(vocab), 
  embedding_dim=embedding_dim, 
  rnn_units=rnn_units, 
  batch_size=BATCH_SIZE)

print(model.summary())

for input_example_batch, target_example_batch in dataset.take(1): 
  example_batch_predictions = model(input_example_batch)
  print(example_batch_predictions.shape, "# (batch_size, sequence_length, vocab_size)")

sampled_indices = tf.random.multinomial(example_batch_predictions[0], num_samples=1)
sampled_indices = tf.squeeze(sampled_indices,axis=-1).numpy()
#print(sampled_indices)

print(input_example_batch)
#print(entire_int_to_text(input_example_batch[0], vocab))
print()
for item in input_example_batch[0]:
	print(int_to_letter(item, vocab), end='', flush=True)
print(entire_int_to_text(sampled_indices, vocab))

example_batch_loss  = loss(target_example_batch, example_batch_predictions)
print("Prediction shape: ", example_batch_predictions.shape, " # (batch_size, sequence_length, vocab_size)") 
print("scalar_loss:      ", example_batch_loss.numpy().mean())

model.compile(
    optimizer = tf.train.AdamOptimizer(),
    loss = loss)

# Directory where the checkpoints will be saved
checkpoint_dir = './training_checkpoints'
# Name of the checkpoint files
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt_{epoch}")

checkpoint_callback=tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_prefix,
    save_weights_only=True)

EPOCHS=5 #num of iterations

#If model.fir will make the model and save it to a directory. You can comment this out if you like your current model
#history = model.fit(dataset.repeat(), epochs=EPOCHS, steps_per_epoch=steps_per_epoch, callbacks=[checkpoint_callback])

tf.train.latest_checkpoint(checkpoint_dir)

model = build_model(vocab_size, embedding_dim, rnn_units, batch_size=1)

model.load_weights(tf.train.latest_checkpoint(checkpoint_dir))

model.build(tf.TensorShape([1, None]))

print(model.summary())

def generate_text(model, start_string):
  # Evaluation step (generating text using the learned model)

  # Number of characters to generate
  num_generate = 1000

  # Converting our start string to numbers (vectorizing) 
  input_eval = np.array([])
  for s in start_string:
  	input_eval = np.append(input_eval, letter_to_int(s, vocab))
  input_eval = tf.expand_dims(input_eval, 0)

  # Empty string to store our results
  text_generated = []

  # Low temperatures results in more predictable text.
  # Higher temperatures results in more surprising text.
  # Experiment to find the best setting.
  temperature = 1.0

  # Here batch size == 1
  model.reset_states()
  for i in range(num_generate):
      predictions = model(input_eval)
      # remove the batch dimension
      predictions = tf.squeeze(predictions, 0)

      # using a multinomial distribution to predict the word returned by the model
      predictions = predictions / temperature
      predicted_id = tf.multinomial(predictions, num_samples=1)[-1,0].numpy()
      
      # We pass the predicted word as the next input to the model
      # along with the previous hidden state
      input_eval = tf.expand_dims([predicted_id], 0)
      print("PREDICTED ID", predicted_id)
      text_generated.append(int_to_letter(predicted_id, vocab))
      print(text_generated)

  return (start_string + ''.join(text_generated))

while True:
  print(generate_text(model, input("Write something")))
