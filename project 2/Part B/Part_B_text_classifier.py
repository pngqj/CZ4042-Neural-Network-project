import numpy as np
import pandas
import tensorflow as tf
import csv
import matplotlib.pyplot as plt
import time
import datetime
from tqdm import tqdm

MAX_DOCUMENT_LENGTH = 100
N_FILTERS = 10
HIDDEN_SIZE = 20 #RNN
POOLING_WINDOW = 4 #CNN
POOLING_STRIDE = 2 #CNN
MAX_LABEL = 15
batch_size = 128
EMBEDDING_SIZE = 20
lr = 0.01
dropout = 0.9
no_epochs = 500

tf.logging.set_verbosity(tf.logging.ERROR)
seed = 10
tf.set_random_seed(seed)

"""# CNN"""

def cnn_model(x,with_dropout,idType):

  if idType == 'char':
    input_layer = tf.reshape(
        tf.one_hot(x, 256), [-1, MAX_DOCUMENT_LENGTH, 256, 1])
    FILTER_SHAPE1 = [20, 256]
    FILTER_SHAPE2 = [20, 1]
  elif idType == 'word':
    word_vectors = tf.contrib.layers.embed_sequence(x, vocab_size=n_words, embed_dim=EMBEDDING_SIZE)
    input_layer = tf.reshape(word_vectors, [-1, MAX_DOCUMENT_LENGTH, EMBEDDING_SIZE, 1])
    FILTER_SHAPE1 = [20,20]
    FILTER_SHAPE2 = [20,1]
    
  with tf.variable_scope('CNN_Layer1'):
    conv1 = tf.layers.conv2d(
        input_layer,
        filters=N_FILTERS,
        kernel_size=FILTER_SHAPE1,
        padding='VALID',
        activation=tf.nn.relu)
    pool1 = tf.layers.max_pooling2d(
        conv1,
        pool_size=POOLING_WINDOW,
        strides=POOLING_STRIDE,
        padding='SAME')
    if with_dropout:
      pool1 = tf.nn.dropout(pool1, dropout)
    
  with tf.variable_scope('CNN_Layer2'):
    conv2 = tf.layers.conv2d(
        pool1,
        filters=N_FILTERS,
        kernel_size=FILTER_SHAPE2,
        padding='VALID',
        activation=tf.nn.relu)
    pool2 = tf.layers.max_pooling2d(
        conv2,
        pool_size=POOLING_WINDOW,
        strides=POOLING_STRIDE,
        padding='SAME')
    
    if with_dropout:
        pool2 = tf.nn.dropout(pool2, dropout)

    pool2 = tf.squeeze(tf.reduce_max(pool2, 1), squeeze_dims=[1])

  logits = tf.layers.dense(pool2, MAX_LABEL, activation=None)

  return input_layer, logits

"""# RNN"""

def create_RNN_cell(cell_type):
      if cell_type == 'GRU':
          cell = tf.nn.rnn_cell.GRUCell(HIDDEN_SIZE)
      elif cell_type == 'RNN':
          cell = tf.nn.rnn_cell.BasicRNNCell(HIDDEN_SIZE)
      elif cell_type == 'LSTM':
          cell = tf.nn.rnn_cell.LSTMCell(HIDDEN_SIZE)
      return cell

def rnn_model(x, with_dropout,idType,cell_type,num_layers,gradient_clipping):
      if idType == "char":
          id_vectors = tf.one_hot(x, 256)
          id_list = tf.unstack(id_vectors, axis=1)
      elif idType == "word":
          id_vectors = tf.contrib.layers.embed_sequence(x, vocab_size=n_words, embed_dim=EMBEDDING_SIZE)
          id_list = tf.unstack(id_vectors, axis=1)

      if num_layers>1:
          cell = tf.contrib.rnn.MultiRNNCell([create_RNN_cell(cell_type) for _ in range(num_layers)])
      else:
          cell = create_RNN_cell(cell_type)

      if with_dropout:
          cell = tf.contrib.rnn.DropoutWrapper(cell, input_keep_prob=dropout, output_keep_prob=dropout)

      _, encoding = tf.nn.static_rnn(cell, id_list, dtype=tf.float32)
      if isinstance(encoding, tuple):
                encoding = encoding[-1]

      logits = tf.layers.dense(encoding, MAX_LABEL, activation=None)

      return logits, id_list

"""# Read Characters"""

def read_data_chars():
  
  x_train, y_train, x_test, y_test = [], [], [], []

  with open('train_medium.csv', encoding='utf-8') as filex:
    reader = csv.reader(filex)
    for row in reader:
      x_train.append(row[1])
      y_train.append(int(row[0]))

  with open('test_medium.csv', encoding='utf-8') as filex:
    reader = csv.reader(filex)
    for row in reader:
      x_test.append(row[1])
      y_test.append(int(row[0]))
  
  x_train = pandas.Series(x_train)
  y_train = pandas.Series(y_train)
  x_test = pandas.Series(x_test)
  y_test = pandas.Series(y_test)
  
  
  char_processor = tf.contrib.learn.preprocessing.ByteProcessor(MAX_DOCUMENT_LENGTH)
  x_train = np.array(list(char_processor.fit_transform(x_train)))
  x_test = np.array(list(char_processor.transform(x_test)))
  y_train = y_train.values
  y_test = y_test.values
  
  return x_train, y_train, x_test, y_test

"""# Read Words"""

def read_data_words():
  
  x_train, y_train, x_test, y_test = [], [], [], []
  
  with open('train_medium.csv', encoding='utf-8') as filex:
    reader = csv.reader(filex)
    for row in reader:
      x_train.append(row[2])
      y_train.append(int(row[0]))

  with open("test_medium.csv", encoding='utf-8') as filex:
    reader = csv.reader(filex)
    for row in reader:
      x_test.append(row[2])
      y_test.append(int(row[0]))
  
  x_train = pandas.Series(x_train)
  y_train = pandas.Series(y_train)
  x_test = pandas.Series(x_test)
  y_test = pandas.Series(y_test)
  y_train = y_train.values
  y_test = y_test.values
  
  vocab_processor = tf.contrib.learn.preprocessing.VocabularyProcessor(
      MAX_DOCUMENT_LENGTH)

  x_transform_train = vocab_processor.fit_transform(x_train)
  x_transform_test = vocab_processor.transform(x_test)

  x_train = np.array(list(x_transform_train))
  x_test = np.array(list(x_transform_test))

  no_words = len(vocab_processor.vocabulary_)
  print('Total words: %d' % no_words)

  return x_train, y_train, x_test, y_test, no_words

"""# Plot Graph"""

def get_title(idType,networkType, with_dropout,cell_type=None,num_layers=1,gradient_clipping=False):
  title = networkType
  if networkType == "RNN":
    title += " (" + cell_type + ")"
  title += " " + idType + " Classifier"
  if num_layers>1:
    title+= " (" + str(num_layers) + " layers)"

  if gradient_clipping:
    title +=  " with gradient clipping" 


  if gradient_clipping and with_dropout:
    title+= " and dropout"
  elif with_dropout:
    title+= " with dropout"

  return title

def plotGraph(loss,test_acc,idType,networkType, with_dropout,cell_type,num_layers,gradient_clipping,time_taken):
  plt.figure(1)
  
  title = get_title(idType,networkType, with_dropout,cell_type,num_layers,gradient_clipping)
  max_acc = max(test_acc)      
  plt.suptitle(title)
  plt.title("time taken: " + time_taken + " best acc: " + str(max_acc), fontsize=9)
  plt.xlabel(str(no_epochs) + ' iterations')
  plt.ylabel('Error')
  plt.plot(range(no_epochs), loss, c="r")
  plt.plot(range(no_epochs), test_acc, c="b")
  plt.legend(["train error", "test accuracy"],loc='upper left')
  title=title.replace(" ", "_") + ".png"
  plt.savefig(title)
  plt.close()

"""# main"""

def textClassifier(idType,networkType, with_dropout,cell_type=None,num_layers=1,gradient_clipping=False):
  global n_words

  if idType == "char":
    x_train, y_train, x_test, y_test = read_data_chars()
  elif idType == "word":
    x_train, y_train, x_test, y_test, n_words= read_data_words()

  print("length of train data:", len(x_train))
  print("length of test data:", len(x_test))
  print("Number of epochs:", no_epochs)

  # Create the model
  x = tf.placeholder(tf.int64, [None, MAX_DOCUMENT_LENGTH])
  y_ = tf.placeholder(tf.int64)

  if networkType == "CNN":
    inputs, logits = cnn_model(x, with_dropout,idType)
  elif networkType == "RNN":
    logits, word_list = rnn_model(x, with_dropout,idType,cell_type,num_layers,gradient_clipping)

  # Optimizer
  entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=tf.one_hot(y_, MAX_LABEL), logits=logits))
  
  if gradient_clipping:
    optimizer = tf.train.AdamOptimizer(learning_rate=lr)
    gvs = optimizer.compute_gradients(entropy)
    capped_gvs = [(tf.clip_by_value(grad, -2., 2.), var) for grad, var in gvs]
    train_op = optimizer.apply_gradients(capped_gvs)

  else:
    train_op = tf.train.AdamOptimizer(lr).minimize(entropy)

  # Accuracy
  correct_prediction = tf.cast(tf.equal(tf.argmax(logits, 1), y_), tf.float32)
  accuracy = tf.reduce_mean(correct_prediction)

  # training
  loss = []
  test_acc = []
  index = np.arange(len(x_train))

  timer = time.time()
  sess = tf.Session()
  sess.run(tf.global_variables_initializer())

  for e in tqdm(range(no_epochs)):
    np.random.shuffle(index)
    x_train, y_train = x_train[index], y_train[index]

    for s in range(0, x_train.shape[0]-batch_size, batch_size):
      sess.run(train_op, {x: x_train[s:s+batch_size], y_: y_train[s:s+batch_size]})
    
    loss_ = sess.run(entropy, {x: x_train, y_: y_train})
    test_acc_ = accuracy.eval(session=sess, feed_dict={x:x_test, y_: y_test})
    
    loss.append(loss_)
    test_acc.append(test_acc_)
  
  sess.close()
  time_taken = time.time() - timer
  time_taken = str(datetime.timedelta(seconds=time_taken)).split(".")[0]
  print('Time Taken: ' + time_taken)

  plotGraph(loss,test_acc,idType,networkType, with_dropout,cell_type,num_layers,gradient_clipping,time_taken)