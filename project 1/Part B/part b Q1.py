# part b Q1
import math
import tensorflow as tf
import numpy as np
import pylab as plt

seed = 10
tf.set_random_seed(seed)


def get_admission_data(small = None):
  
    np.random.seed(seed) 
    #read and divide data into test and train sets
    admit_data = np.genfromtxt('admission_predict.csv', delimiter= ',')
    X_data, Y_data = admit_data[1:,1:8], admit_data[1:,-1]
    Y_data = Y_data.reshape(Y_data.shape[0], 1)

    idx = np.arange(X_data.shape[0])
    np.random.shuffle(idx)
    X_data, Y_data = X_data[idx], Y_data[idx]

    if small is not None:
        # experiment with small datasets
        X_data = X_data[:small]
        Y_data = Y_data[:small]
    
    X_data = (X_data- np.mean(X_data, axis=0))/ np.std(X_data, axis=0)
    
    split = 0.7
    #create train set
    trainX = X_data[:int(X_data.shape[0]*split)]
    trainY = Y_data[:int(Y_data.shape[0]*split)]

    #create test set
    testX = X_data[:int(X_data.shape[0]*split*-1)]
    testY = Y_data[:int(Y_data.shape[0]*split*-1)]
    
    #randomly choose 50 samples from test set
    rand = np.random.choice(testX.shape[0], 50, replace=True)
    testX = testX[rand]
    testY = testY[rand]
    return trainX, trainY, testX, testY

def gradientDescentOptimizer(trainX, trainY, testX, testY, print_result = False, epochs = 1000, num_layer = 3, num_neuron = 10, with_dropouts = False):
    # initialise constants
    learning_rate = 10**-3
    beta = 10**-3
    batch_size = 8
    NUM_FEATURES = np.size(trainX,1)
    dropout = 0.8
    
    # Create the model
    x = tf.placeholder(tf.float32, [None, NUM_FEATURES])
    y_ = tf.placeholder(tf.float32, [None, 1])
    
    # Build the graph for the deep net
    stddev = 1.0/math.sqrt(NUM_FEATURES)
    
    # hidden layer 1
    seed=10
    weights_h1 = tf.Variable(tf.truncated_normal([NUM_FEATURES,num_neuron], 
                             stddev=stddev, dtype=tf.float32,seed=seed), 
                             name='weights')
    biases_h1 = tf.Variable(tf.zeros([num_neuron]))
    h1 = tf.nn.relu(tf.matmul(x, weights_h1) + biases_h1)
    if with_dropouts:
        h1 = tf.nn.dropout(h1, dropout)

    # hidden layer 2
    seed=20 
    weights_h2 = tf.Variable(tf.truncated_normal([num_neuron,num_neuron], 
                             stddev=stddev, dtype=tf.float32,seed=seed), 
                             name='weights')
    biases_h2 = tf.Variable(tf.zeros([num_neuron]))
    h2 = tf.nn.relu(tf.matmul(h1, weights_h2) + biases_h2)
    if with_dropouts:
        h2 = tf.nn.dropout(h2, dropout)

    # hidden layer 3
    seed=30
    weights_h3 = tf.Variable(tf.truncated_normal([num_neuron,num_neuron], 
                            stddev=stddev, dtype=tf.float32,seed=seed), 
                             name='weights')
    biases_h3 = tf.Variable(tf.zeros([num_neuron]))
    h3 = tf.nn.relu(tf.matmul(h2, weights_h3) + biases_h3)
    if with_dropouts:
        h3 = tf.nn.dropout(h3, dropout)
        
        
    #output layer
    seed=40
    weights = tf.Variable(tf.truncated_normal([num_neuron, 1], stddev=stddev, dtype=tf.float32, seed=seed), name='weights')
    biases = tf.Variable(tf.zeros([1]), dtype=tf.float32, name='biases')

    if num_layer == 3:
        y = tf.matmul(h1, weights) + biases
        regularization = tf.nn.l2_loss(weights) + tf.nn.l2_loss(weights_h1)
    elif num_layer == 4:
        y = tf.matmul(h2, weights) + biases
        regularization = tf.nn.l2_loss(weights) + tf.nn.l2_loss(weights_h1) + tf.nn.l2_loss(weights_h2)
    elif num_layer == 5:
        y = tf.matmul(h3, weights) + biases
        regularization = tf.nn.l2_loss(weights) + tf.nn.l2_loss(weights_h1) + tf.nn.l2_loss(weights_h2) + tf.nn.l2_loss(weights_h3)

    ridge_loss = tf.square(y_ - y)
    loss = tf.reduce_mean(ridge_loss + beta*regularization)

    #Create the gradient descent optimizer with the given learning rate.
    optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    train_op = optimizer.minimize(loss)
    error = tf.reduce_mean(tf.square(y_ - y))

    train_errs = []
    test_errs = []
    n = trainX.shape[0]
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        for i in range(epochs):
          # shuffle
          idx = np.arange(trainX.shape[0])
          np.random.shuffle(idx)
          trainX, trainY = trainX[idx], trainY[idx]

          # implementing mini-batch GD
          for s in range(0, n-batch_size, batch_size):
              train_op.run(feed_dict={x: trainX[s:s+batch_size], y_: trainY[s:s+batch_size]})
          train_err = error.eval(feed_dict={x: trainX, y_:trainY})
          train_errs.append(train_err)
          test_err = error.eval(feed_dict={x: testX, y_:testY})
          test_errs.append(test_err)

          if print_result:
              print("training iteration %d" %i, end="\r")
              if i % 100 == 0:
                  print('iter %d: train error %g'%(i, train_errs[i]))
                  print('iter %d: test error %g'%(i, test_errs[i]))

        pred = sess.run(y, feed_dict={x:testX})
    
    if print_result:
        if epochs >= 20000:
        # plot learning curves
          plt.figure(1)
          plt.xlabel(str(epochs) + ' iterations')
          plt.ylabel('Error')
          z = 10000
          plt.plot(range(z,epochs), train_errs[z:], c="r")
          plt.plot(range(z,epochs), test_errs[z:], c="b")
          plt.legend(["train error", "test error"],loc='upper left')

        plt.figure(2)
        plt.xlabel(str(epochs) + ' iterations')
        plt.ylabel('Error')
        z = 0
        plt.plot(range(z,epochs), train_errs[z:], c="r")
        plt.plot(range(z,epochs), test_errs[z:], c="b")
        plt.legend(["train error", "test error"],loc='upper left')

        plt.figure(3)
        plt.xlabel('Sample number')
        plt.ylabel('Chance to Admit')
        plt.plot(testY, c="r")
        plt.plot(pred, c="b")
        plt.legend(["Target", "Predicted"],loc='upper left')
        plt.show()
    
    return test_errs[len(test_errs) -1]

trainX, trainY, testX, testY = get_admission_data()
gradientDescentOptimizer(trainX, trainY, testX, testY, epochs = 1000, print_result = True, num_layer = 3, num_neuron = 10, with_dropouts = False)
