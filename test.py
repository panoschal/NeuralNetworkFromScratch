# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
import torch
import torchvision
import torchvision.transforms as transforms
from time import time
from matplotlib import pyplot as plt
import numpy as np
from more_itertools import pairwise, prepend
from operator import itemgetter
from tqdm import tqdm
import tensorflow.compat.v2 as tf

# %%
transform = transforms.Compose([transforms.ToTensor(),
                              transforms.Normalize((0.5,), (0.5,)),
                              ])

def labels_to_one_hot(y):
    return (np.arange(10) == y[:, None]).astype(np.float32)

trainset = torchvision.datasets.MNIST('trainset', download=True, train=True, transform=transform)
valset = torchvision.datasets.MNIST('valset', download=True, train=False, transform=transform)

X_train = trainset.data.numpy()
Y_train = trainset.targets.numpy()
X_val = valset.data.numpy()
Y_val = valset.targets.numpy()

Y_train = labels_to_one_hot(Y_train)
Y_val = labels_to_one_hot(Y_val)

# %%
def show_image(image):
    plt.imshow(image.squeeze(), interpolation='nearest')
    plt.show()

def show_many_images(images):
    fig, axes = plt.subplots(5,5, figsize=(8,8))
    for i,ax in enumerate(axes.flat):
        ax.imshow(images[i].squeeze())


# %%
# show_image(trainset[0][0])


# %%
# show_many_images(trainset.data[0:25])


# %%
trainset


# %%
valset

# %%
class GradientsAveragingAccumulator():
    '''
    we assume all items have the same structure
    '''

    def __init__(self):
        self.accumulation = None
        self.count = 0

    def accumulate(self, item):
        if self.accumulation is None:
            self.accumulation = item
        else:
            for key in self.accumulation:
                for layer in self.accumulation[key]:
                    self.accumulation[key][layer] += item[key][layer]

        self.count += 1
    
    def result(self):
        average = {}
        for key in self.accumulation:
            average[key] = {}
            for layer in self.accumulation[key]:
                average[key][layer] = self.accumulation[key][layer] / self.count
        
        return average

class SimpleAveragingAccumulator():
    '''
    we assume all items have the same structure
    '''

    def __init__(self):
        self.accumulation = None
        self.count = 0

    def accumulate(self, item):
        if self.accumulation is None:
            self.accumulation = item
        else:
            self.accumulation += item

        self.count += 1
    
    def result(self):
        return self.accumulation / self.count

# %%
def sigmoid(X):
    return 1 / (1 + np.exp(-X))

def sigmoid_derivative(X):
    return sigmoid(X)*(1-sigmoid(X))

def relu(X):
    return np.maximum(0, X)

def softmax(x):
    return np.exp(x)/np.sum(np.exp(x))

def softmax_derivative(x):
    return softmax(x)*(1-softmax(x)) # supposing i = j

class NeuralNetwork():
    def __init__(self, input_size, layer_sizes):
        '''
        for example, mnist is input_size=28*28=784 and layers=(one hidden, one hidden, output layer), you dont count the first one, but the layer is at the right. 4 rows of vertices, is 3 layers
        the last element of layer_sizes has to be 10 for mnist, to predict the 10 classes
        '''
        
        self.layers = []
        for previous_size, this_size in pairwise(prepend(input_size, layer_sizes)):
            # each neuron has weights with all the previous neurons, and 1 bias
            # from 7 inputs, a layer with 9 neurons has 7*9 weights and 9 biases
            # in order to be aligned for operations, weights shape is 16 neurons * 784 weights each
            initializer = tf.keras.initializers.GlorotUniform()

            weights = initializer(shape=(this_size, previous_size)).numpy()
            biases = np.zeros(this_size,)
            self.layers.append([weights, biases])
        

    def forward_one_example(self, x):
        '''
        input is a numpy array of shape (784,)
        return a numpy array of shape (10,) the output neuron activations
        '''

        # activations = x
        # for weights, biases in self.layers:
        #     z = weights @ activations + biases
        #     activations = sigmoid( z )

        # return activations
        final_layer = len(self.layers) - 1
        
        assert x.shape == (784,)

        activations = {}
        z = {}

        activations[-1] = x

        for l, (weights, biases) in enumerate(self.layers):
            assert isinstance(l,int) and isinstance(weights, np.ndarray) and isinstance(biases, np.ndarray) and len(self.layers) < 5

            this_layer = l
            previous_layer = l-1

            z[this_layer] = weights @ activations[previous_layer] + biases
            activation_function = sigmoid if this_layer<final_layer else softmax
            activations[this_layer] = activation_function( z[this_layer] )

            assert z[this_layer].shape == activations[this_layer].shape

        return z, activations

        # weights is (16,784) and activations is (784,)
        # the matmul result is (16,)

    def cost_one_example(self, y_pred, y_true):
        '''
        the loss is mean squared loss
        arguments are numpy arrays of shape (10,)
        return a single number
        '''

        return np.sum( (y_pred - y_true)**2 )

    def train(self, X_train, Y_train, epochs=5, learning_rate=0.1, batch_size=64):
        final_layer = len(self.layers) - 1

        for epoch in tqdm(range(epochs)):
            # one gradient descent step

            # calculate the negative gradient of the cost function
            # a vector with 13002 elements
            # each corresponds to one model parameter (weight or bias)
            # shows how sensitive the cost function is, to this parameter (nudge in this parameter, yields big or small change in the cost function result)

            total_loss = SimpleAveragingAccumulator()
            accuracy = SimpleAveragingAccumulator()
            accuracies = []
            for batch in range(len(X_train) % batch_size):

                # slice only one batch
                X_batch = X_train[batch*batch_size:(batch+1)*batch_size]
                Y_batch = Y_train[batch*batch_size:(batch+1)*batch_size]
                assert len(X_batch) == len(Y_batch) == batch_size

                negative_gradients_accumulator = GradientsAveragingAccumulator()
                for x, y_true in zip(X_batch, Y_batch):
                    assert x.shape == (1,28,28) or x.shape == (28,28)
                    x = x.reshape(784)
                    if torch.is_tensor(x):
                        x = x.numpy()
                    assert y_true.shape == (10,) and np.sum(y_true)==1

                    z, activations = self.forward_one_example(x)
                    y_pred = activations[final_layer]
                    accuracy.accumulate(y_pred.argmax() == y_true.argmax())
                    accuracies.append(y_pred.argmax() == y_true.argmax())
                    assert y_pred.shape == (10,)
                    loss = self.cost_one_example(y_pred, y_true)
                    total_loss.accumulate(loss)

                    # C_over_activations[i] means: partial derivatives of C over activations of layer i (some of the elements that will go into the gradient)

                    # initializations
                    # I use dicts in order to have whatever indices are convenient, e.g. -1 and to not initialize them
                    C_over_weights = {}
                    C_over_biases = {}
                    C_over_activations = {}
                    C_over_activations[final_layer] = 2*(activations[final_layer] - y_true) # imaginary initial derivatives for the one past final layer
                    activations[-1] = x
                    weights = [weights for weights, biases in self.layers]
                    biases = [biases for weights, biases in self.layers]

                    for l, layer in list(enumerate(self.layers))[::-1]:
                        assert isinstance(l, int) and isinstance(layer, list)
                        this_layer = l
                        previous_layer = l-1

                        # function that takes ndarray of shape (neurons_in_this_layer,) and returns ndarray of same shape, with the partial derivatives over each element
                        derivative_of_activation_function = sigmoid_derivative if this_layer < final_layer else softmax_derivative

                        C_over_weights[this_layer] = (
                                                    C_over_activations[this_layer] *
                                                    derivative_of_activation_function( z[this_layer] ) 
                                                    ).reshape(-1,1) @ activations[previous_layer].reshape(1,-1)

                        assert C_over_weights[this_layer].shape == weights[this_layer].shape

                        C_over_biases[this_layer] = (C_over_activations[this_layer] *
                                                    derivative_of_activation_function( z[this_layer] ) *
                                                    1)

                        assert C_over_biases[this_layer].shape == biases[this_layer].shape

                        # if previous_layer >= 0: # we dont care about C over activations of the -1 layer (inputs)
                        C_over_activations[previous_layer] = ((
                                                    C_over_activations[this_layer] *
                                                    derivative_of_activation_function( z[this_layer] ) 
                                                    ).reshape(1,-1) @ weights[this_layer] # cant directly influence this, will backpropagate to find C over other weights and biases
                        ).reshape(-1)
                        assert C_over_activations[previous_layer].shape == activations[previous_layer].shape

                    # gather all partial derivatives in one negative gradient object
                    negative_gradient = {
                        'over_weights': {layer: -partial_derivatives for layer, partial_derivatives in C_over_weights.items()}, 
                        'over_biases': {layer: -partial_derivatives for layer, partial_derivatives in C_over_biases.items()}
                    }
                    negative_gradients_accumulator.accumulate(negative_gradient)

                # nudge each model parameter to the direction this showed
                # multiplied by the learning rate
                negative_gradient_all_batch = negative_gradients_accumulator.result()

                self.update_model_parameters(learning_rate, negative_gradient_all_batch)

            print('train average loss', total_loss.result())
            print('train accuracy', accuracy.result(), sum(accuracies)/len(accuracies))

    def update_model_parameters(self, learning_rate, negative_gradient_all_batch):
        for l, layer in enumerate(self.layers):
            assert isinstance(l, int) and isinstance(layer, list)
            layer[0] += learning_rate * negative_gradient_all_batch['over_weights'][l]
            layer[1] += learning_rate * negative_gradient_all_batch['over_biases'][l]

    def inference(self, x):
        x = x.reshape(784)
        if torch.is_tensor(x):
            x = x.numpy()
        z, activations = self.forward_one_example(x)
        return activations[len(self.layers)-1]

# %%

nn = NeuralNetwork(input_size=784, layer_sizes=(64, 10))
nn.train(X_train[0:10000], Y_train[0:10000], 
    epochs=60, learning_rate=0.1, batch_size=64)
# %%
nn.inference(trainset.data[20000])
# %%

# with keras

model = tf.keras.models.Sequential([
  tf.keras.layers.Flatten(input_shape=(28, 28, 1)),
  tf.keras.layers.Dense(64, activation='sigmoid'),
  tf.keras.layers.Dense(10, activation='softmax')
])
optimizer = tf.keras.optimizers.SGD(learning_rate=0.1)
model.compile(
    loss='mse',
    optimizer=optimizer,
    metrics=['accuracy'],
)
model.fit(
    X_train[0:10000],
    Y_train[0:10000],
    epochs=60,
    batch_size=64
    # validation_data=,
)
# %%
