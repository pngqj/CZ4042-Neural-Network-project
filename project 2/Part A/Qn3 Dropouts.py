import math
import tensorflow as tf
import numpy as np
import pylab as plt
import pickle
import os

if not os.path.isdir('figures'):
    print('creating the figures folder')
    os.makedirs('figures')

NUM_CLASSES = 10
IMG_SIZE = 32
NUM_CHANNELS = 3
learning_rate = 0.001
no_epochs = 1000
batch_size = 128
features_map_1 = 100
features_map_2 = 100

seed = 10
np.random.seed(seed)
tf.set_random_seed(seed)


def load_data(file):
    with open(file, 'rb') as fo:
        try:
            samples = pickle.load(fo)
        except UnicodeDecodeError:  # python 3.x
            fo.seek(0)
            samples = pickle.load(fo, encoding='latin1')

    data, labels = samples['data'], samples['labels']

    data = np.array(data, dtype=np.float32)
    labels = np.array(labels, dtype=np.int32)

    labels_ = np.zeros([labels.shape[0], NUM_CLASSES])
    labels_[np.arange(labels.shape[0]), labels - 1] = 1

    return data, labels_


def cnn(images):
    images = tf.reshape(images, [-1, IMG_SIZE, IMG_SIZE, NUM_CHANNELS])

    # Conv 1
    W_conv1 = weight_variable([9, 9, NUM_CHANNELS, features_map_1], stddev=1.0 / np.sqrt(NUM_CHANNELS * 9 * 9),
                              name='h_weight1')
    b_conv1 = bias_variable([features_map_1], name='h_biases1')
    u_conv1 = tf.nn.conv2d(images, W_conv1, strides=[1, 1, 1, 1], padding='VALID') + b_conv1
    h_conv1 = tf.nn.relu(u_conv1)

    # Pooling layer, max pool
    h_pool1 = tf.nn.max_pool(h_conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID', name='h_pool1')

    # Conv 2
    W_conv2 = weight_variable([5, 5, features_map_1, features_map_2], stddev=1.0 / np.sqrt(features_map_1 * 5 * 5),
                              name='h_weight2')
    b_conv2 = bias_variable([features_map_2], name='h_biases2')
    u_conv2 = tf.nn.conv2d(h_pool1, filter=W_conv2, strides=[1, 1, 1, 1], padding='VALID') + b_conv2
    h_conv2 = tf.nn.relu(u_conv2)

    # Pooling layer 2, max pool
    h_pool2 = tf.nn.max_pool(h_conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='VALID', name='h_pool2')

    # Flatten the output
    dim = h_pool2.get_shape()[1].value * h_pool2.get_shape()[2].value * h_pool2.get_shape()[3].value
    pool_2_flat = tf.reshape(h_pool2, [-1, dim])

    # Full connected layer
    W_fc1 = weight_variable([dim, 300], 1.0 / np.sqrt(dim), name='weights_fc1')
    b_fc1 = bias_variable([300], name='bias_fc1')
    u_fc1 = tf.matmul(pool_2_flat, W_fc1) + b_fc1
    h_fc1 = tf.nn.relu(u_fc1)

    # Dropout - controls the complexity of the model, prevents co-adaptation of
    # features.

    keep_prob = tf.placeholder(tf.float32)
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

    # Softmax
    W_fc2 = weight_variable([300, NUM_CLASSES], stddev=1.0 / np.sqrt(dim), name='weights_fc2')
    b_fc2 = tf.Variable(tf.zeros([NUM_CLASSES]), name='bias_fc2')

    y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

    return W_conv1, h_conv1, h_pool1, h_conv2, h_pool2, y_conv, keep_prob


def weight_variable(shape, stddev, name):
    initial = tf.truncated_normal(shape, stddev=stddev)
    return tf.Variable(initial, name=name)


def bias_variable(shape, name):
    return tf.Variable(tf.zeros(shape), name=name)


def main():
    trainX, trainY = load_data('./data/data_batch_1')
    print(trainX.shape, trainY.shape)

    testX, testY = load_data('./data/test_batch_trim')
    print(testX.shape, testY.shape)

    trainX = (trainX - np.min(trainX, axis=0)) / np.max(trainX, axis=0)
    testX = (testX - np.min(testX, axis=0)) / np.max(testX, axis=0)

    # Create the model
    x = tf.placeholder(tf.float32, [None, IMG_SIZE * IMG_SIZE * NUM_CHANNELS])
    y_ = tf.placeholder(tf.float32, [None, NUM_CLASSES])

    W_conv1, h_conv1, h_pool1, h_conv2, h_pool2, y_conv, keep_prob = cnn(x)

    cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(labels=y_, logits=y_conv)
    loss = tf.reduce_mean(cross_entropy)

    # 3A
    train_step2 = tf.train.MomentumOptimizer(learning_rate, 0.1).minimize(loss)
    # 3B
    train_step3 = tf.train.RMSPropOptimizer(learning_rate).minimize(loss)
    # 3C
    train_step4 = tf.train.AdamOptimizer(learning_rate).minimize(loss)
    # 3D
    train_step5 = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)

    correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
    correct_prediction = tf.cast(correct_prediction, tf.float32)
    accuracy = tf.reduce_mean(correct_prediction)

    N = len(trainX)
    idx = np.arange(N)
    with tf.Session() as sess:

        # 3A
        print('momentum...')
        sess.run(tf.global_variables_initializer())

        test_acc2 = []
        train_cost2 = []
        for i in range(no_epochs):
            np.random.shuffle(idx)
            trainX, trainY = trainX[idx], trainY[idx]

            for start, end in zip(range(0, N, batch_size), range(batch_size, N, batch_size)):
                train_step2.run(feed_dict={x: trainX[start:end], y_: trainY[start:end], keep_prob: 0.8})

            test_acc2.append(accuracy.eval(feed_dict={x: testX, y_: testY, keep_prob: 1.0}))
            train_cost2.append(loss.eval(feed_dict={x: trainX, y_: trainY, keep_prob: 1.0}))

            if i % 100 == 0:
                print('iter %d: test accuracy %g' % (i, test_acc2[i]))
                print('iter %d: training cost %g' % (i, train_cost2[i]), '\n')

        plt.figure(1)
        plt.plot(np.arange(no_epochs), test_acc2, label='Momentum with Dropouts')
        plt.plot(np.arange(no_epochs), train_cost2)

        plt.xlabel('epochs')
        plt.ylabel('Accuracy & Training Cost')
        plt.legend(['Test Accuracy', 'Training Cost'], loc='upper right')
        plt.savefig('./figures/3A_dropouts.png')

        # 3B
        print('RMSProp...')
        sess.run(tf.global_variables_initializer())

        test_acc3 = []
        train_cost3 = []
        for i in range(no_epochs):
            np.random.shuffle(idx)
            trainX, trainY = trainX[idx], trainY[idx]

            for start, end in zip(range(0, N, batch_size), range(batch_size, N, batch_size)):
                train_step3.run(feed_dict={x: trainX[start:end], y_: trainY[start:end], keep_prob: 0.8})

            test_acc3.append(accuracy.eval(feed_dict={x: testX, y_: testY, keep_prob: 1.0}))
            train_cost3.append(loss.eval(feed_dict={x: trainX, y_: trainY, keep_prob: 1.0}))

            if i % 100 == 0:
                print('iter %d: test accuracy %g' % (i, test_acc3[i]))
                print('iter %d: training cost %g' % (i, train_cost3[i]), '\n')

        plt.figure(2)
        plt.plot(np.arange(no_epochs), test_acc3, label='RMSProp with Dropouts')
        plt.plot(np.arange(no_epochs), train_cost3)

        plt.xlabel('epochs')
        plt.ylabel('Accuracy & Training Cost')
        plt.legend(['Test Accuracy', 'Training Cost'], loc='upper right')
        plt.savefig('./figures/3B_dropouts.png')

        # 3C
        print('AdamOp...')
        sess.run(tf.global_variables_initializer())

        test_acc4 = []
        train_cost4 = []
        for i in range(no_epochs):
            np.random.shuffle(idx)
            trainX, trainY = trainX[idx], trainY[idx]

            for start, end in zip(range(0, N, batch_size), range(batch_size, N, batch_size)):
                train_step4.run(feed_dict={x: trainX[start:end], y_: trainY[start:end], keep_prob: 0.8})

            test_acc4.append(accuracy.eval(feed_dict={x: testX, y_: testY, keep_prob: 1.0}))
            train_cost4.append(loss.eval(feed_dict={x: trainX, y_: trainY, keep_prob: 1.0}))

            if i % 100 == 0:
                print('iter %d: test accuracy %g' % (i, test_acc4[i]))
                print('iter %d: training cost %g' % (i, train_cost4[i]), '\n')

        plt.figure(3)
        plt.plot(np.arange(no_epochs), test_acc4, label='AdamOp with Dropouts')
        plt.plot(np.arange(no_epochs), train_cost4)

        plt.xlabel('epochs')
        plt.ylabel('Accuracy & Training Cost')
        plt.legend(['Test Accuracy', 'Training Cost'], loc='upper right')
        plt.savefig('./figures/3C_dropouts.png')

        # 3D
        print('gd with dropout...')
        sess.run(tf.global_variables_initializer())

        test_acc5 = []
        train_cost5 = []

        for i in range(no_epochs):
            np.random.shuffle(idx)
            trainX, trainY = trainX[idx], trainY[idx]

            for start, end in zip(range(0, N, batch_size), range(batch_size, N, batch_size)):
                train_step5.run(feed_dict={x: trainX[start:end], y_: trainY[start:end], keep_prob: 0.8})

            train_cost5.append(loss.eval(feed_dict={x: trainX, y_: trainY, keep_prob: 1.0}))
            test_acc5.append(accuracy.eval(feed_dict={x: testX, y_: testY, keep_prob: 1.0}))

            if i % 100 == 0:
                print('iter %d: test accuracy %g' % (i, test_acc5[i]))
                print('iter %d: training cost %g' % (i, train_cost5[i]), '\n')

        plt.figure(4)
        plt.plot(np.arange(no_epochs), test_acc5, label='GD with Dropouts')
        plt.plot(np.arange(no_epochs), train_cost5)

        plt.xlabel('epochs')
        plt.ylabel('Accuracy & Training Cost')
        plt.legend(['Test Accuracy', 'Training Cost'], loc='upper right')
        plt.savefig('./figures/3D_dropouts.png')

        # Test Accuracies vs Epochs
        plt.figure(5)
        plt.plot(np.arange(no_epochs), test_acc2, label='Momentum y=0.1')
        plt.plot(np.arange(no_epochs), test_acc3, label='RMSProp')
        plt.plot(np.arange(no_epochs), test_acc4, label='Adam')
        plt.plot(np.arange(no_epochs), test_acc5, label='GD')
        plt.xlabel('epochs')
        plt.ylabel('Test Accuracies')
        plt.title('Test Accuracies vs Epochs')
        plt.legend(loc='best')
        plt.savefig('./figures/Q3_TestAccuracy_vs_Epochs_dropouts.png')

        # Training Costs vs Epochs
        plt.figure(6)
        plt.plot(np.arange(no_epochs), train_cost2, label='Momentum y=0.1')
        plt.plot(np.arange(no_epochs), train_cost3, label='RMSProp')
        plt.plot(np.arange(no_epochs), train_cost4, label='Adam')
        plt.plot(np.arange(no_epochs), train_cost5, label='GD')
        plt.xlabel('epochs')
        plt.ylabel('Training Costs')
        plt.title('Training Costs vs Epochs')
        plt.legend(loc='best')
        plt.savefig('./figures/Q3_TrainCosts_vs_Epochs_dropouts.png')

        plt.show()


if __name__ == '__main__':
    main()
