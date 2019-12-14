"""
Convolutional neural network architecture.
Author: Sara Mathieson and Jack St Clair
Date:11/18/2019
"""

import numpy as np
import tensorflow as tf

from tensorflow.keras.layers import Dense, Flatten, Conv2D
from tensorflow.keras import Model

##################

class CNNmodel(Model):
    """
    A convolutional neural network; the architecture is:
    Conv -> ReLU -> Conv -> ReLU -> Dense
    Note that we only need to define the forward pass here; TensorFlow will take
    care of computing the gradients for us.
    """
    def __init__(self):
        super(CNNmodel, self).__init__()
        # TODO complete constructor
        # First conv layer: 32 filters, each 5x5
        # Second conv layer: 16 filters, each 3x3
        self.d1 = Conv2D(32, (5, 5), activation = tf.keras.activations.relu, \
        data_format = 'channels_last', input_shape = (64, 32, 32, 3))
        self.d2 = Conv2D(16, (3, 3), activation = tf.keras.activations.relu)
        self.d3 = Dense(10, activation = tf.keras.activations.softmax)
        self.flat = Flatten()

    def call(self, x):
        # TODO complete call method
        x1 = self.d1(x)
        x2 = self.d2(x1)
        x3 = self.flat(x2)
        x4 = self.d3(x3)
        return x4

def three_layer_convnet_test():
    """Test function to make sure the dimensions are working"""

    # Create an instance of the model
    cnn_model = CNNmodel()

    # TODO try out both the options below (all zeros and random)
    # shape is: number of examples (mini-batch size), width, height, depth
    #x_np = np.zeros((64, 32, 32, 3))
    x_np = np.random.rand(64, 32, 32, 3)

    # call the model on this input and print the result
    output = cnn_model.call(x_np)
    print(output) # TODO what shape is this? does it make sense?

    # TODO look at the model parameter shapes, do they make sense?
    for v in cnn_model.trainable_variables:
        print("Variable:", v.name)
        print("Shape:", v.shape)

def main():
    # test three layer function
    three_layer_convnet_test()

if __name__ == "__main__":
    main()
