#
# Project 1, starter code part a
#
import math
import tensorflow as tf
import numpy as np
import pylab as plt
import os
from sklearn.model_selection import KFold, train_test_split

if not os.path.isdir('figures'):
    print('creating the figures folder')
    os.makedirs('figures')

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)


# scale data
def scale(X, X_min, X_max):
    return (X - X_min) / (X_max - X_min)


NUM_FEATURES = 21
NUM_CLASSES = 3

learning_rate = 0.01
epochs = 1200
batch_size = 8
num_neurons = 20
seed = 10
beta = tf.constant(1e-6)
tf.set_random_seed(seed)


def fnn(x, hidden_units):
    # Hidden 1
    h_weights = tf.Variable(
        tf.random.truncated_normal([NUM_FEATURES, hidden_units], stddev=1.0 / math.sqrt(float(NUM_FEATURES))),
        name='weights')
    h_biases = tf.Variable(tf.zeros([hidden_units]), name='biases')

    h = tf.nn.relu(tf.matmul(x, h_weights) + h_biases)

    # Output layer
    weights = tf.Variable(
        tf.random.truncated_normal([hidden_units, NUM_CLASSES], stddev=1.0 / math.sqrt(float(NUM_FEATURES))),
        name='weights')
    biases = tf.Variable(tf.zeros([NUM_CLASSES]), name='biases')
    logits = tf.matmul(h, weights) + biases

    return logits, h_weights, weights


def main():
    # read train data

    train_input = np.genfromtxt('ctg_data_cleaned.csv', delimiter=',')
    trainX, train_Y = train_input[1:, :21], train_input[1:, -1].astype(int)
    trainX = scale(trainX, np.min(trainX, axis=0), np.max(trainX, axis=0))

    trainY = np.zeros((train_Y.shape[0], NUM_CLASSES))
    trainY[np.arange(train_Y.shape[0]), train_Y - 1] = 1  # one hot matrix

    # experiment with small datasets
    # trainX = trainX[:1000]
    # trainY = trainY[:1000]

    # split the test and training data into 70:30
    trainX, testX, trainY, testY = train_test_split(trainX, trainY, test_size=0.3, shuffle=True)

    n = trainX.shape[0]
    print(n)

    # Create the model
    x = tf.placeholder(tf.float32, [None, NUM_FEATURES])
    y_ = tf.placeholder(tf.float32, [None, NUM_CLASSES])

    logits, h_weights, weights = fnn(x, num_neurons)

    # Build the graph for the deep net

    cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(labels=y_, logits=logits)
    L2_regularization = tf.nn.l2_loss(h_weights) + tf.nn.l2_loss(weights)
    loss = tf.reduce_mean(cross_entropy + beta * L2_regularization)

    # Create the gradient descent optimizer with the given learning rate.
    optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    # global_step = tf.Variable(0, name='global_step', trainable=False)
    # train_op = optimizer.minimize(loss, global_step=global_step)
    train_op = optimizer.minimize(loss)

    correct_prediction = tf.cast(tf.equal(tf.argmax(logits, 1), tf.argmax(y_, 1)), tf.float32)
    accuracy = tf.reduce_mean(correct_prediction)

    N = len(trainX)
    idx = np.arange(N)
    print(idx)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        train_acc = []
        test_acc = []

        for i in range(epochs):
            np.random.shuffle(idx)
            trainX = trainX[idx]
            trainY = trainY[idx]

            for start, end in zip(range(0, N, batch_size), range(batch_size, N, batch_size)):
                train_op.run(feed_dict={x: trainX[start:end], y_: trainY[start:end]})

            train_acc.append(accuracy.eval(feed_dict={x: trainX, y_: trainY}))
            test_acc.append(accuracy.eval(feed_dict={x: testX, y_: testY}))

            if i % 100 == 0:
                print('iter %d: training accuracy %g' % (i, train_acc[i]))
                print('iter %d: test accuracy %g' % (i, test_acc[i]), '\n')

    # plot learning curves

    plt.figure(1)
    plt.plot(range(epochs), train_acc)
    plt.plot(range(epochs), test_acc)
    plt.xlabel(str(epochs) + ' iterations')
    plt.ylabel('Accuracy')
    plt.title('Training & Testing Accuracy with 20 neurons')
    plt.legend(['Training Accuracy', 'Testing Accuracy'], loc='lower right')
    plt.savefig('./figures/PartA_Qn3C_Training_Testing_Accuracy.png')

    plt.show()


if __name__ == '__main__':
    main()
