# chRUSTmas

A Rust implementation from scratch of a classifier that can identify images of cats and non-cats.

First, I have implemented a form of what’s called **forward propagation**, wherein a neural network takes input data and makes a prediction. 
It’s called this because you’re propagating activations forward through the network. 

An **activation function** decides whether and how strongly a neuron should “fire”. Without activation functions Neural networks become just linear models and cannot learn complex patterns.
Activation functions introduce non-linearity and shapes the model’s capacity. The activation function can be any non-linear function applied element-wise to the elements of the logit matrix. 
In our model, we will use the relu activation function for all intermediate layers and sigmoid for the last layer (classifier layer), but not tanh.

Next important question is a **loss function**. A loss function measures how wrong a model’s prediction is compared to the correct answer. Then training process tries to minimize the loss.
Without a loss function, the model has no signal telling it how to improve. The loss function provides the feedback used by optimization algorithms (like Gradient Descent). Since we have a simple binary classification problem, the loss function is given as below:

XX

The purpose of the loss function is to measure the discrepancy between the predicted labels and the true labels. By minimizing this loss, we aim to make our model’s predictions as close as possible to the ground truth.
To train the model and minimize the loss, we employ a technique called **backward propagation**. This technique calculates **the gradients of the cost function** with respect to the weights and biases, which indicates the direction and magnitude of adjustments required for each parameter. The gradient computations are performed using the following equations for each layer:

XX

To better understand:
- Loss function → error for one example
- Cost function → error for the whole dataset (or batch), usually an average or sum of losses

A loss function measures how wrong the model is for a single training example. Defined per sample i.e.: **Loss = local error**.

A cost function aggregates the losses over many samples. Most commonly Cost = average (or sum) of losses. Gradient descent minimizes not individual losses, but optimizes at the dataset (or batch) level i.e. **Cost = global error**. It happens because individual losses are noisy and the model must perform well on average, so Cost gives a more stable optimization signal.

Once we have calculated the gradients, we can adjust the weights and biases to minimize the loss. The following equations are used for updating the parameters using a learning rate alpha:

XX

So whole pipeline looks like this:
Inputs
  ↓
Linear combination (Wx + b)
  ↓
Activation function
  ↓
Prediction ŷ
  ↓
Loss function (per example)
  ↓
Cost function (aggregate)
  ↓
Backpropagation

where:
- Activation functions shape predictions
- Loss functions measure error
- Cost function is what we actually minimize

Keep in mind that in practice (PyTorch, TensorFlow, papers) people often say “loss function” when they really mean "cost over the batch", but the distinction is frequently ignored.

## Dependencies

cargo add polars -F lazy
cargo add ndarray -F serde
cargo add rand
cargo add num-integer