import os
import sys
import numpy as np
import random
import string
import tensorflow as tf
from tensorflow.models.rnn import rnn, rnn_cell
import collections
import urllib
import zipfile

url = 'http://mattmahoney.net/dc/'

def maybe_download(filename, expected_bytes):
  """Download a file if not present, and make sure it's the right size."""
  if not os.path.exists(filename):
    filename, _ = urllib.urlretrieve(url + filename, filename)
  statinfo = os.stat(filename)
  if statinfo.st_size == expected_bytes:
    print 'Found and verified', filename
  else:
    print statinfo.st_size
    raise Exception(
      'Failed to verify ' + filename + '. Can you get to it with a browser?')
  return filename

filename = maybe_download('text8.zip', 31344016)

def read_data(filename):
  f = zipfile.ZipFile(filename)
  for name in f.namelist():
    return f.read(name)
  f.close()
  
text = read_data(filename)
print "Data size", len(text)

valid_size = 1000
valid_text = text[:valid_size]
train_text = text[valid_size:]
train_size = len(train_text)
print train_size, train_text[:64]
print valid_size, valid_text[:64]

n_gram_size=2

def build_n_gram_dataset(text, n_gram_size):
  index = 0
  dictionary = dict()
  
  text_len = len(text)
  for i in xrange(text_len + n_gram_size):
    letters = []
    for j in xrange(n_gram_size):
      letter_idx = (i + j) % text_len
      letters.append(text[letter_idx])
    n_gram = ''.join(letters)
    
    if n_gram not in dictionary:
      dictionary[n_gram] = len(dictionary)
    index = dictionary[n_gram]
    
  reverse_dictionary = dict(zip(dictionary.values(), dictionary.keys()))    
  return dictionary, reverse_dictionary

dictionary, reverse_dictionary = build_n_gram_dataset(text, n_gram_size)
vocabulary_size = len(dictionary)

def n_gram_to_encoding(n_gram):
  id = dictionary[n_gram]
  
  encoding = np.zeros(shape=(vocabulary_size), dtype=np.float)
  encoding[id] = 1.0
  
  return encoding

def probs_to_ids(probabilities):
  return [c for c in np.argmax(probabilities, 1)]

def logprob(predictions, labels):
  """Log-probability of the true labels in a predicted batch."""
  predictions[predictions < 1e-10] = 1e-10
  return np.sum(np.multiply(labels, -np.log(predictions))) / labels.shape[0]

def sample(prediction, bottom_start=0):
  """Turn a (column) prediction into 1-hot encoded samples."""
  p = np.zeros(shape=[vocabulary_size], dtype=np.float)
  p[sample_distribution(prediction[0], bottom_start)] = 1.0
  return p

def sample_distribution(distribution, bottom_start=0):
  """Sample one element from a distribution assumed to be an array of normalized
  probabilities.
  """
  r = random.uniform(0, 1)
  s = 0
  for i in xrange(len(distribution)):
    s += distribution[i]
    if s >= r:
      return i
  return len(distribution) - 1

def random_distribution():
  """Generate a random column of probabilities."""
  b = np.random.uniform(0.0, 1.0, size=[1, vocabulary_size])
  return b/np.sum(b, 1)[:,None]

def prob_to_n_gram(probability):
  ngram_id = np.argmax(probability)
  ngram = reverse_dictionary[ngram_id]
  
  return ngram

def probs_2_n_gram_ids(probabilities):
  return [np.argmax(probability) for probability in probabilities]

def probabilities_to_n_grams(probabilities):
  return [prob_to_n_gram(x) for x in probabilities]

def n_gram_to_id(ngram):
  return dictionary[ngram]

def id_to_n_gram(id):
  return reverse_dictionary[id]

#print prob_to_n_gram(n_gram_to_encoding(" a"))
#enc = n_gram_to_encoding(" a")
#print enc
#print probabilities_to_n_grams([n_gram_to_encoding(" a"), n_gram_to_encoding("an")])

batch_size=64
num_unrollings=10

class BatchGenerator(object):
  def __init__(self, text, batch_size, num_unrollings, n_gram_size):
    self._text = text
    self._text_size = len(text)
    self._batch_size = batch_size
    self._num_unrollings = num_unrollings
    self._n_gram_size = n_gram_size
    segment = self._text_size / batch_size
    self._segment_size = segment
    self._cursor = [ offset * segment for offset in xrange(batch_size)]
    print self._cursor
    self._last_batch = self._next_batch()
  
  def _next_batch(self):
    """Generate a single batch from the current cursor position in the data."""
    batch = np.zeros(shape=(self._batch_size, vocabulary_size), dtype=np.float)
    for b in xrange(self._batch_size):
      batch[b, char2id(self._text[self._cursor[b]])] = 1.0
      self._cursor[b] = (self._cursor[b] + 1) % self._text_size
    return batch
  
  def _next_batch(self):
    """Generate a single batch from the current cursor position in the data."""
    batch = np.zeros(shape=(self._batch_size, vocabulary_size), dtype=np.float)
    
    for b in xrange(self._batch_size):
      letters = []
      for i in xrange(self._n_gram_size):
        letter_idx = (self._cursor[b] + i) % self._text_size
        letter = self._text[letter_idx]
        letters.append(letter)
      n_gram = ''.join(letters)
      n_gram_id = n_gram_to_id(n_gram)
      
      batch[b, n_gram_id] = 1.0
      self._cursor[b] = (self._cursor[b] + self._n_gram_size) % self._text_size
    return batch
  
  def next(self):
    """Generate the next array of batches from the data. The array consists of
    the last batch of the previous array, followed by num_unrollings new ones.
    """
    batches = [self._last_batch]
    for step in xrange(self._num_unrollings):
      batches.append(self._next_batch())
    self._last_batch = batches[-1]
    return batches

def characters(probabilities):
  """Turn a 1-hot encoding or a probability distribution over the possible
  characters back into its (mostl likely) character representation."""
  return [id_to_n_gram(c) for c in np.argmax(probabilities, 1)]

def batches2string(batches):
  """Convert a sequence of batches back into their (most likely) string
  representation."""
  s = [''] * batches[0].shape[0]
  for b in batches:
    s = [''.join(x) for x in zip(s, probabilities_to_n_grams(b))]
  return s

train_batches = BatchGenerator(train_text, batch_size, num_unrollings, n_gram_size)
valid_batches = BatchGenerator(valid_text, 1, 1, 2)

"""
print batches2string(train_batches.next())
print batches2string(train_batches.next())
print batches2string(valid_batches.next())
print batches2string(valid_batches.next())
"""

num_nodes = 64
embedding_size = 64	
num_steps = 24001

graph = tf.Graph()
with graph.as_default():
  
  # Parameters:
  # Variables saving state across unrollings.
  saved_output1 = tf.Variable(tf.zeros([batch_size, num_nodes]), trainable=False)
  saved_state1 = tf.Variable(tf.zeros([batch_size, num_nodes]), trainable=False)
  
  saved_output2 = tf.Variable(tf.zeros([batch_size, num_nodes]), trainable=False)
  saved_state2 = tf.Variable(tf.zeros([batch_size, num_nodes]), trainable=False)
  # Classifier weights and biases.
  w = tf.Variable(tf.truncated_normal([num_nodes, vocabulary_size], -0.1, 0.1))
  b = tf.Variable(tf.zeros([vocabulary_size]))
    
  # Defining matrices for: input gate, forget gate, memory cell, output gate
  m_rows = 4
  m_input_index = 0
  m_forget_index = 1
  m_update_index = 2
  m_output_index = 3
  m_input_w = tf.Variable(tf.truncated_normal([m_rows, embedding_size, num_nodes], -0.1, 0.1))
  m_middle = tf.Variable(tf.truncated_normal([m_rows, num_nodes, num_nodes], -0.1, 0.1))
  m_biases = tf.Variable(tf.truncated_normal([m_rows, 1, num_nodes], -0.1, 0.1))
  m_saved_output = tf.Variable(tf.zeros([m_rows, batch_size, num_nodes]), trainable=False)
  m_input = tf.Variable(tf.zeros([m_rows, batch_size, num_nodes]), trainable=False)
  
  # Variables.
  embeddings = tf.Variable(
    tf.random_uniform([vocabulary_size, embedding_size], -1.0, 1.0))
  # Dropout
  keep_prob = tf.placeholder(tf.float32) 

  # Definition of the 2nd LSTM layer
  m_input_w2 = tf.Variable(tf.truncated_normal([m_rows, embedding_size, num_nodes], -0.1, 0.1))
  m_middle_w2 = tf.Variable(tf.truncated_normal([m_rows, num_nodes, num_nodes], -0.1, 0.1))
  m_biases2 = tf.Variable(tf.truncated_normal([m_rows, 1, num_nodes], -0.1, 0.1))
  m_saved_output2 = tf.Variable(tf.zeros([m_rows, batch_size, num_nodes]), trainable=False)
  m_input2 = tf.Variable(tf.zeros([m_rows, batch_size, num_nodes]), trainable=False)
  
  # Definition of the cell computation.
  def lstm_cell_improved(i, o, state):
    m_input = tf.pack([i for _ in range(m_rows)])
    m_saved_output = tf.pack([o for _ in range(m_rows)])
    
    m_input = tf.nn.dropout(m_input, keep_prob)
    m_all = tf.batch_matmul(m_input, m_input_w) + tf.batch_matmul(m_saved_output, m_middle) + m_biases
    m_all = tf.unpack(m_all)
    
    input_gate = tf.sigmoid(m_all[m_input_index])
    forget_gate = tf.sigmoid(m_all[m_forget_index])
    update = m_all[m_update_index]
    state = forget_gate * state + input_gate * tf.tanh(update)
    output_gate = tf.sigmoid(m_all[m_output_index])
    
    return output_gate * tf.tanh(state), state
  
  def lstm_cell_2(i, o, state):
    """Create a LSTM cell. See e.g.: http://arxiv.org/pdf/1402.1128v1.pdf
    Note that in this formulation, we omit the various connections between the
    previous state and the gates."""    
    m_input2 = tf.pack([i for _ in range(m_rows)])
    m_saved_output2 = tf.pack([o for _ in range(m_rows)])
    
    m_input2 = tf.nn.dropout(m_input2, keep_prob)
    m_all = tf.batch_matmul(m_input2, m_input_w2) + tf.batch_matmul(m_saved_output2, m_middle_w2) + m_biases
    m_all = tf.unpack(m_all)
    
    input_gate = tf.sigmoid(m_all[m_input_index])
    forget_gate = tf.sigmoid(m_all[m_forget_index])
    update = m_all[m_update_index]
    state = forget_gate * state + input_gate * tf.tanh(update)
    output_gate = tf.sigmoid(m_all[m_output_index])
    
    return output_gate * tf.tanh(state), state
  
  # Input data.
  train_data = list()
  train_labels = list()
  
  for x in xrange(num_unrollings):
    train_data.append(
      tf.placeholder(tf.int32, shape=[batch_size]))
    train_labels.append(
      tf.placeholder(tf.float32, shape=[batch_size,vocabulary_size]))
  
  encoded_inputs = list()
  for bigram_batch in train_data:
    embed = tf.nn.embedding_lookup(embeddings, bigram_batch)
    encoded_inputs.append(embed)
  
  train_inputs = encoded_inputs

  # Unrolled LSTM loop.
  outputs = list()
  output1 = saved_output1
  output2 = saved_output2
  state1 = saved_state1
  state2 = saved_state2
  for i in train_inputs:
    output1, state1 = lstm_cell_improved(i, output1, state1)
    output2, state2 = lstm_cell_2(output1, output2, state2)
    outputs.append(output2)

  # State saving across unrollings.
  with tf.control_dependencies([saved_output1.assign(output1),
                                saved_state1.assign(state1),
                                saved_output2.assign(output2),
                                saved_state2.assign(state2)]):
    # Classifier.
    logits = tf.nn.xw_plus_b(tf.concat(0, outputs), w, b)
    loss = tf.reduce_mean(
      tf.nn.softmax_cross_entropy_with_logits(
        logits, tf.concat(0, train_labels)))

  # Optimizer.
  global_step = tf.Variable(0)
  learning_rate = tf.train.exponential_decay(
    10.0, global_step, num_steps / 2, 0.1, staircase=False)
  optimizer = tf.train.GradientDescentOptimizer(learning_rate)
  gradients, v = zip(*optimizer.compute_gradients(loss))
  gradients, _ = tf.clip_by_global_norm(gradients, 1.25)
  optimizer = optimizer.apply_gradients(
    zip(gradients, v), global_step=global_step)

  # Predictions.
  train_prediction = tf.nn.softmax(logits)
  
  # Sampling and validation eval: batch 1, no unrolling.
  sample_input = tf.placeholder(tf.int32, shape=[1])
  sample_embed = tf.nn.embedding_lookup(embeddings, sample_input)
  saved_sample_output1 = tf.Variable(tf.zeros([1, num_nodes]))
  saved_sample_state1 = tf.Variable(tf.zeros([1, num_nodes]))
  saved_sample_output2 = tf.Variable(tf.zeros([1, num_nodes]))
  saved_sample_state2 = tf.Variable(tf.zeros([1, num_nodes]))
  reset_sample_state = tf.group(
    saved_sample_output1.assign(tf.zeros([1, num_nodes])),
    saved_sample_state1.assign(tf.zeros([1, num_nodes])),
    saved_sample_output2.assign(tf.zeros([1, num_nodes])),
    saved_sample_state2.assign(tf.zeros([1, num_nodes])))
  sample_output1, sample_state1 = lstm_cell_improved(
    sample_embed, saved_sample_output1, saved_sample_state1)
  sample_output2, sample_state2 = lstm_cell_2(
    sample_output1, saved_sample_output2, saved_sample_state2)
  with tf.control_dependencies([saved_sample_output1.assign(sample_output1),
                                saved_sample_state1.assign(sample_state1),
                                saved_sample_output2.assign(sample_output2),
                                saved_sample_state2.assign(sample_state2)]):
    sample_prediction = tf.nn.softmax(tf.nn.xw_plus_b(sample_output2, w, b))
	
summary_frequency = 100

with tf.Session(graph=graph) as session:
  tf.initialize_all_variables().run()
  print 'Initialized'
  mean_loss = 0
  for step in xrange(num_steps):
    batches = train_batches.next()
    feed_dict = dict()
    
    # setup inputs
    for i in xrange(num_unrollings):
      data = probs_to_ids(batches[i])
      feed_dict[train_data[i]] = data
    
    # setup outputs  
    for i in xrange(1, num_unrollings + 1, 1):
      feed_dict[train_labels[i-1]] = batches[i]
    
    # setup dropout
    feed_dict[keep_prob] = 0.8
    _, l, predictions, lr = session.run(
      [optimizer, loss, train_prediction, learning_rate], feed_dict=feed_dict)
    mean_loss += l
    if step % summary_frequency == 0:
      if step > 0:
        mean_loss = mean_loss / summary_frequency
      # The mean loss is an estimate of the loss over the last few batches.
      print 'Average loss at step', step, ':', mean_loss, 'learning rate:', lr
      mean_loss = 0
      labels = np.concatenate(list(batches)[1:])
      print 'Minibatch perplexity: %.2f' % float(
        np.exp(logprob(predictions, labels)))
      
      
      if step % (summary_frequency * 10) == 0:
        # Generate some samples.
        print '=' * 80
        for _ in xrange(5):
          feed = sample(random_distribution())
          sentence = characters([feed])[0]
          feed = probs_to_ids([feed])
          reset_sample_state.run()
          for _ in xrange(79):
            prediction = sample_prediction.eval({sample_input: feed, keep_prob: 1.0})
            feed = sample(prediction)
            sentence += characters([feed])[0]
            feed = probs_to_ids([feed])
          print sentence
        print '=' * 80
      # Measure validation set perplexity.
      reset_sample_state.run()
      valid_logprob = 0
      for _ in xrange(valid_size):
        b = valid_batches.next()
        feed = probs_to_ids(b[0])
        predictions = sample_prediction.eval({sample_input: feed, keep_prob: 1.0})
        valid_logprob = valid_logprob + logprob(predictions, b[1])
      print 'Validation set perplexity: %.2f' % float(np.exp(
        valid_logprob / valid_size))