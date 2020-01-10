#
# Project 1, starter code part a
#
import math
import tensorflow as tf
import numpy as np
import pylab as plt
import os
from sklearn.model_selection import KFold, train_test_split
import time
from multiprocessing import Pool

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
batch_size = [4, 8, 16, 32, 64]
num_neurons = 10
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


def main(batch_size):
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

    # n = trainX.shape[0]
    # print(n)

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
    global_step = tf.Variable(0, name='global_step', trainable=False)
    train_op = optimizer.minimize(loss, global_step=global_step)

    correct_prediction = tf.cast(tf.equal(tf.argmax(logits, 1), tf.argmax(y_, 1)), tf.float32)
    accuracy = tf.reduce_mean(correct_prediction)

    kf = KFold(n_splits=5, shuffle=True)
    kf.get_n_splits(trainX)
    # print(kf)

    acc_per_fold = []
    test_acc_per_fold = []

    for train_index, test_index in kf.split(trainX):
        # print("TRAIN:", train_index, "TEST:", test_index)
        trainX_, testX_ = trainX[train_index], trainX[test_index]
        trainY_, testY_ = trainY[train_index], trainY[test_index]

        N = len(trainX_)

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            train_acc = []
            test_acc = []

            for i in range(epochs):
                for start, end in zip(range(0, N, batch_size), range(batch_size, N, batch_size)):
                    train_op.run(feed_dict={x: trainX_[start:end], y_: trainY_[start:end]})

                train_acc.append(accuracy.eval(feed_dict={x: trainX_, y_: trainY_}))
                test_acc.append(accuracy.eval(feed_dict={x: testX_, y_: testY_}))

                if i % 100 == 0:
                    print('iter %d: training accuracy %g' % (i, train_acc[i]))
                    print('iter %d: test accuracy %g' % (i, test_acc[i]), '\n')

            acc_per_fold.append(train_acc)
            # print('hey hey',acc_per_fold)
            test_acc_per_fold.append(test_acc)

        # find the mean accuracy of the 5 folds
        mean_acc = np.mean(np.array(acc_per_fold), axis=0)
        test_mean_acc = np.mean(np.array(test_acc_per_fold), axis=0)
        # print(mean_acc)

    return train_acc, test_acc, mean_acc, test_mean_acc


if __name__ == '__main__':
    p = Pool(5)
    paras = p.map(main, batch_size)
    paras = np.array(paras)
    train_acc, test_acc, mean_acc, test_mean_acc = paras[:, 0], paras[:, 1], paras[:, 2], paras[:,3]
    p.close()
    p.join()

    # for no_batch in range(len(batch_size)):
    #     print('validating with batch size of:', batch_size[no_batch])
    #     train_acc, test_acc, mean_acc, test_mean_acc = main(batch_size[no_batch])
    for i in range(5):
        # plot learning curves
        plt.figure(1)
        plt.plot(range(epochs), mean_acc[i])
        plt.legend(["batch size 4", "batch size 8", "batch size 16", "batch size 32", "batch size 64"],
                   loc='lower right')
        plt.title('Training Accuracy')
        plt.xlabel(str(epochs) + ' Epochs')
        plt.ylabel('Cross Validation Accuracy')
        plt.savefig('./figures/PartA_Qn2a_trainAcc.png')

        plt.figure(2)
        plt.plot(range(epochs), test_mean_acc[i])
        plt.legend(["batch size 4", "batch size 8", "batch size 16", "batch size 32", "batch size 64"],
                   loc='lower right')
        plt.title('Testing Accuracy')
        plt.xlabel(str(epochs) + ' Epochs')
        plt.ylabel('Cross Validation Accuracy')
        plt.savefig('./figures/PartA_Qn2a_testAcc.png')

        # plt.figure(2)
        # plt.plot(range(epochs), train_acc)
        # plt.plot(range(epochs), test_acc)
        # plt.legend(["Train acc", "Test acc"])
        # plt.xlabel(str(epochs) + 'Epochs')
        # plt.ylabel('Accuracy')

    plt.show()
