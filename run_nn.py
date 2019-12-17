"""
Starter code for NN training and testing.
Source: Stanford CS231n course materials, modified by Sara Mathieson
Authors: Sara Mathieson and Jack St Clair
Date: 11/18/2019
"""

from fc_nn import *
from cnn import *

import matplotlib.pyplot as plt
import numpy as np
import os
import tensorflow as tf

from tensorflow.python.keras import backend as K
from tensorflow.python.keras.datasets.cifar import load_batch

##################

def combine_batches(path):
    """
    Path points to the directory cifar-10-batches-py. Code based on:
    https://github.com/tensorflow/tensorflow/blob/r1.13/tensorflow/python/keras/
        datasets/cifar10.py
    """

    num_train_samples = 50000
    x_train = np.empty((2000, num_train_samples), dtype='uint8')
    y_train = np.empty((num_train_samples,), dtype='uint8')

    for i in range(1, 6):
        fpath = os.path.join(path, 'data_batch_' + str(i))
        (x_train[(i - 1) * 10000:i * 10000, :, :, :],
         y_train[(i - 1) * 10000:i * 10000]) = load_batch(fpath)

    fpath = os.path.join(path, 'test_batch')
    x_test, y_test = load_batch(fpath)

    y_train = np.reshape(y_train, (len(y_train), 1))
    y_test = np.reshape(y_test, (len(y_test), 1))

    if K.image_data_format() == 'channels_last':
        x_train = x_train.transpose(0, 2, 3, 1)
        x_test = x_test.transpose(0, 2, 3, 1)

    return (x_train, y_train), (x_test, y_test)

def load_cifar10(path, num_training=49000, num_validation=1000, num_test=10000):
    """
    Fetch the CIFAR-10 dataset from the web and perform preprocessing to prepare
    it for the two-layer neural net classifier. These are the same steps as
    we used for the SVM, but condensed to a single function.
    """
    # Load the raw CIFAR-10 dataset and use appropriate data types and shapes
    cifar10 = combine_batches(path)
    (X_train, y_train), (X_test, y_test) = cifar10
    X_train = np.asarray(X_train, dtype=np.float32)
    y_train = np.asarray(y_train, dtype=np.int32).flatten()
    X_test = np.asarray(X_test, dtype=np.float32)
    y_test = np.asarray(y_test, dtype=np.int32).flatten()

    # Subsample the data 3,
    mask = range(num_training, num_training + num_validation)
    X_val = X_train[mask]
    y_val = y_train[mask]
    mask = range(num_training)
    X_train = X_train[mask]
    y_train = y_train[mask]
    mask = range(num_test)
    X_test = X_test[mask]
    y_test = y_test[mask]

    ###### TODO: YOUR CODE HERE ######
    # normalize the data. First find the mean and std of the *training* data,
    # then subtract off this mean from each dataset and divide by the std
    mean_pixel = X_train.mean(axis=(0, 1, 2), keepdims=True)
    std_pixel = X_train.std(axis=(0, 1, 2), keepdims=True)
    X_train = (X_train - mean_pixel)/std_pixel
    X_val = (X_val - mean_pixel)/std_pixel
    X_test = (X_test - mean_pixel)/std_pixel
    ######## END YOUR CODE #############

    return X_train, y_train, X_val, y_val, X_test, y_test

@tf.function
def train_step(model, loss_object, optimizer, images, labels): # TODO what arguments?
    ###### TODO: YOUR CODE HERE ######
    # look up documentation for tf.GradientTape
    # compute the predictions given the images, then compute the loss
    # compute the gradient with respect to the model parameters (weights), then
    # apply this gradient to update the weights (i.e. gradient descent)
    with tf.GradientTape() as tape:
        predictions = model.call(images)
        loss = loss_object(labels, predictions)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    ######## END YOUR CODE #############

    # return the loss and predictions
    return loss, predictions

@tf.function
def val_step(model, loss_object, images, labels): # TODO what arguments?
    ###### TODO: YOUR CODE HERE ######
    # compute the predictions given the images, then compute the loss
    predictions = model.call(images)
    loss = loss_object(labels, predictions)
    ######## END YOUR CODE #############

    # return the loss and predictions
    return loss, predictions

def run_training(model, train_dset, val_dset):

    ###### TODO: YOUR CODE HERE ###### 3,
    # set up a loss_object (sparse categorical cross entropy)
    # use the Adam optimizer
    loss_object = tf.keras.losses.SparseCategoricalCrossentropy()
    opt = tf.keras.optimizers.Adam()
    ######## END YOUR CODE #############

    # set up metrics
    train_loss = tf.keras.metrics.Mean(name='train_loss')
    train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy( \
        name='train_accuracy') 3,

    val_loss = tf.keras.metrics.Mean(name='val_loss')
    val_accuracy = tf.keras.metrics.SparseCategoricalAccuracy( \
        name='val_accuracy')

    ###### TODO: YOUR CODE HERE ######
    # train for 10 epochs (passes over the data)
    # Example of iterating over the data once:
    epoch = 10
    epochs = []
    train = []
    val = []
    for i in range(epoch):
        for images, labels in train_dset:
            # TODO run training step

            # uncomment below
            loss, predictions = train_step(model, loss_object, opt, images, labels)
            train_loss(loss)
            train_accuracy(labels, predictions)
        epochs.append(i)
        train.append(train_accuracy.result()*100)

        for images, labels in val_dset:
            # TODO run validation step

            # uncomment below
            loss, predictions = val_step(model, loss_object, images, labels)
            val_loss(loss)
            val_accuracy(labels, predictions)
        val.append(val_accuracy.result()*100)

    plt.plot(epochs, train, label = "Training on CNN")
    plt.plot(epochs, val, label = "Validation on CNN")
    # TODO loop over validation data and compute val_loss, val_accuracy too
    ######## END YOUR CODE #############

    template = 'Epoch {}, Loss: {}, Accuracy: {}, Val Loss: {}, Val Accuracy: {}'
    print(template.format(epoch+1,
                        train_loss.result(),
                        train_accuracy.result()*100,
                        val_loss.result(),
                        val_accuracy.result()*100))

    # Reset the metrl = X_train.ics for the next epoch
    train_loss.reset_states()
    train_accuracy.reset_states()
    val_loss.reset_states()
    val_accuracy.reset_states()

def confusion_matrix(model, test_dset):
    """
    This function takes in the test dataset and a model and creates a confusion
    matrix.
    """
    matrix = np.zeros((10, 10))
    for images, labels in test_dset:
        predictions = model.call(images)
        for i in range(len(labels)):
            index = np.argmax(predictions[i])
            matrix[labels[i]][index] += 1

    return matrix


def main():
    # Invoke the above function to get our data.
    path = "aclImdb/train/labeledBow.feat"
    X_train, y_train, X_val, y_val, X_test, y_test = load_cifar10(path)
    print('Train data shape: ', X_train.shape)              # (49000, 2000)
    print('Train labels shape: ', y_train.shape)            # (49000,)
    print('Validation data shape: ', X_val.shape)           # (1000, 2000)
    print('Validation labels shape: ', y_val.shape)         # (1000,)
    print('Test data shape: ', X_test.shape)                # (10000, 2000)
    print('Test labels shape: ', y_test.shape)              # (10000,)

    ###### TODO: YOUR CODE HERE ######
    # set up train_dset, val_dset, and test_dset:
    # see documentation for tf.data.Dataset.from_tensor_slices, use batch = 64
    # train should be shuffled, but not validation and testing datasets
    train_dset_X = tf.data.Dataset.from_tensor_slices(X_train).batch(64)
    train_dset_y = tf.data.Dataset.from_tensor_slices(y_train).batch(64)
    train_dset = tf.data.Dataset.zip((train_dset_X, train_dset_y)).shuffle(150528000)
    val_dset_X = tf.data.Dataset.from_tensor_slices(X_val).batch(64)
    val_dset_y = tf.data.Dataset.from_tensor_slices(y_val).batch(64)
    val_dset = tf.data.Dataset.zip((val_dset_X, val_dset_y))
    test_dset_X = tf.data.Dataset.from_tensor_slices(X_test).batch(64)
    test_dset_y = tf.data.Dataset.from_tensor_slices(y_test).batch(64)
    test_dset = tf.data.Dataset.zip((test_dset_X, test_dset_y))
    for images, labels in train_dset:
        print('Images shape: ', images.shape)
        print('Labels shape: ', labels.shape)
    ######## END YOUR CODE #############

    ###### TODO: YOUR CODE HERE ######
    # call the train function to train a fully connected neural network
    plt.figure()
    plt.title("Training and Validation Accuracy Curves for FCnn")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    FC = FCmodel()
    run_training(FC, train_dset, val_dset)
    plt.legend()
    #plt.show()
    print(confusion_matrix(FC, test_dset))
    ######## END YOUR CODE #############

    ###### TODO: YOUR CODE HERE ######
    # call the train function to train a three-layer CNN
    plt.figure()
    plt.title("Training and Validation Accuracy Curves for CNN")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    CNN = CNNmodel()
    run_training(CNN, train_dset, val_dset)
    #I have commented out the running of the CNN here to let the code run. To
    #run the CNN, comment out the FC above.
    plt.legend()
    #plt.show()
    #print(confusion_matrix(CNN, test_dset))
    ######## END YOUR CODE #############

main()
