# Neural Network from scratch

In this project, I implemented a neural network from scratch in Python, without using a library like PyTorch or TensorFlow. I used numpy for efficient computations.

This helped me understand backpropagation and the math behind gradient descent better.

# Review

While deep learning libraries make the NN design a very easy process, requiring only a few lines of code, a simple NN is not that hard to create from scratch. By doing this, you get a better understanding of their inner workings.

The main concept here is gradient descent / backpropagation.

One main goal of this project was to write expressive and readable code to make the algorithm easy to understand by reading it. This is, of course, one of the advantages of Python.

The first draft of the training loop was this:

```py
for epoch in range(epochs):
	# one gradient descent step

	# calculate the negative gradient of the cost function
	# a vector with 13002 elements
	# each corresponds to one model parameter (weight or bias)
	# shows how sensitive the cost function is, to this parameter (nudge in this parameter, yields big or small change in the cost function result)

	for x, y_true in trainset:
		y_pred = self.forward_one_example(x)
		cost = self.cost_one_example(y_pred, y_true)

		# this is the partial derivative (one element of the gradient)
		# average it over all training examples
		C_over_weights_this_layer = (
									2*(activations_this_layer - y_true) *
									sigmoid_derivative( z_this_layer ) 
									).reshape((None,1)) @ activations_previous_layer.reshape((1,None))

		C_over_biases_this_layer = (2*(activations_this_layer - y_true) *
									sigmoid_derivative( z_this_layer ) *
									1)

		C_over_activations_previous_layer = (
									2*(activations_this_layer - y_true) *
									sigmoid_derivative( z_this_layer ) 
									).reshape((1,None)) @ weights_this_layer # cant directly influence this, will backpropagate to find C over other weights and biases

		
	# nudge each model parameter to the direction this showed
	# multiplied by the learning rate
```