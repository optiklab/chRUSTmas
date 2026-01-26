# chRUSTmas

A Rust implementation from scratch of a classifier that can identify images of cats and non-cats.

## To run

```
cargo install --path .
ch_rust_mas_app.exe
```

Results:
```
ch_rust_mas_app.exe
Epoch : 0/1000    Cost: 0.693141
Epoch : 100/1000    Cost: 0.6493075
Epoch : 200/1000    Cost: 0.6268611
Epoch : 300/1000    Cost: 0.5944505
Epoch : 400/1000    Cost: 0.5521986
Epoch : 500/1000    Cost: 0.5064041
Epoch : 600/1000    Cost: 0.47501507
Epoch : 700/1000    Cost: 0.4300008
Epoch : 800/1000    Cost: 0.3909312
Epoch : 900/1000    Cost: 0.34643888
Training Set Accuracy: 88.03828%
Test Set Accuracy: 56%
```

## Explainer

First, I have implemented a form of what’s called **forward propagation**, wherein a neural network takes input data and makes a prediction. 
It’s called this because you’re propagating activations forward through the network. 

An **activation function** decides whether and how strongly a neuron should “fire”. Without activation functions Neural networks become just linear models and cannot learn complex patterns.
Activation functions introduce non-linearity and shapes the model’s capacity. The activation function can be any non-linear function applied element-wise to the elements of the logit matrix. 
In our model, we will use the relu activation function for all intermediate layers and sigmoid for the last layer (classifier layer), but not tanh.

Next important question is a **loss function**. A loss function measures how wrong a model’s prediction is compared to the correct answer. Then training process tries to minimize the loss.
Without a loss function, the model has no signal telling it how to improve. The loss function provides the feedback used by optimization algorithms (like Gradient Descent). 

A loss function measures how wrong the model is for a single training example. Defined per sample i.e.: **Loss = local error**.

The purpose of the loss function is to measure the discrepancy between the predicted labels and the true labels. By minimizing this loss, we aim to make our model’s predictions as close as possible to the ground truth.
To train the model and minimize the loss, we employ a technique called **backward propagation**. This technique calculates **the gradients of the cost function** with respect to the weights and biases, which indicates the direction and magnitude of adjustments required for each parameter. So whole pipeline looks like this:

```
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
```

where:
- Activation functions shape predictions.
- Loss functions measure error.
- Cost function is what we actually minimize.

Keep in mind that in practice (PyTorch, TensorFlow, papers) people often say “loss function” when they really mean "cost over the batch", but the distinction is frequently ignored, so:
- Loss function → error for one example.
- Cost function → error for the whole dataset (or batch), usually an average or sum of losses.

**A cost function** aggregates the losses over many samples. Most commonly Cost = average (or sum) of losses. Gradient descent minimizes not individual losses, but optimizes at the dataset (or batch) level i.e. **Cost = global error**. It happens because individual losses are noisy and the model must perform well on average, so Cost gives a more stable optimization signal. Since we have a simple **binary classification problem**, we are using typical loss function given as below:

```
J(w, b) = −(1 / m) · [ Ŷ · log(A^[L]) + (1 − Ŷ) · log(1 − A^[L]) ]
```

See `fn cost` function in code.

This is the binary cross-entropy (log loss) cost function, typically used with Sigmoid activation in the output layer and Binary classification problems, where:
- J(w, b) — cost function
- m — number of training examples
- Ŷ (Y_hat) — true labels (ground truth)
- A^[L] (AL) — output of the last (L-th) layer, i.e. predicted probability
- log — natural logarithm

The gradient computations are performed using the following equations for each layer:

First, apply the chain rule through the activation function. First, `dZ` at layer `l` equals `dA` at layer `l` element‑wise multiplied by the derivative of the activation function evaluated at `Z[l]`:
```
dZ^[l] = dA^[l] ⊙ g′(Z^[l])
```

Next, compute the gradient of the weights. `dW` at layer `l` equals one over m times `dZ` at layer `l` multiplied by the transpose of `A` from the previous layer:
```
dW^[l] = (1 / m) · dZ^[l] · (A^[l−1])^T
```

Then, compute the gradient of the bias (sum over axis = 1). `db` at layer `l` equals one over `m` times the sum of `dZ` at layer `l` across all training examples:
```
db^[l] = (1 / m) · Σ(dZ^[l])  
```

Finally, propagate the gradient back to the previous layer. `dA` of the previous layer equals the transpose of `W` at layer `l` multiplied by `dZ` at layer `l`: 
```
dA^[l−1] = (W^[l])^T · dZ^[l] 
```

Meaning of symbols (quick glossary):
- `l` — layer index
- `m` — number of training examples
- `Z[l]` — linear output (W[l]A[l−1] + b[l])
- `A[l]` — activation output of layer l
- `g′(Z[l])` — derivative of activation function
- `W[l]` — weight matrix
- `⊙` — element‑wise (Hadamard) product/multiplication

Derivations of these equations can be found [here](https://res.cloudinary.com/dltwftrgc/image/upload/v1684930865/Blogs/rust_dnn_4/derivations_lylhqq.png). These equations are the core of vectorized backpropagation used in deep neural networks.

Once we have calculated the gradients, we can adjust the weights and biases to minimize the loss. The following equations are used for updating the Gradient‑descent parameters using a learning rate alpha for each layer `l`:
```
W^[l] = W^[l] − α · dW^[l]

b^[l] = b^[l] − α · db^[l]
```

Where:
- `α` (alpha) is the learning rate
- `dW^[l]` is the gradient of the cost w.r.t. the weights
- `db^[l]` is the gradient of the cost w.r.t. the bias

These equations update the weights and biases of each layer based on their respective gradients. By iteratively performing the forward and backward passes, and updating the parameters using the gradients, we allow the model to learn and improve its performance over time.

These gradients then used in the optimization step to update the parameters and minimize the cost.

[See full](https://www.akshaymakes.com/blogs/rust_dnn_part4).

## Dependencies

```
cargo add polars -F lazy
cargo add ndarray -F serde
cargo add rand
cargo add num-integer
```

### Polars explainer

```
cargo add polars -F lazy
```

-F (--features)

Enables the Lazy API in Polars — lazy evaluation (similar to Spark / DataFrames):
- you build a computation graph
- the query optimizer performs filter pushdown, projection pushdown, etc.
- execution starts only when .collect() is called

Example (lazy execution):
```
    use polars::prelude::*;

    fn main() -> PolarsResult<()> {
        let df = LazyFrame::scan_csv("data.csv", Default::default())?
            .filter(col("size").gt(10))           // не выполняется
            .select([col("name"), col("size")])   // не выполняется
            .collect()?; // запуск оптимизатора и выполнения

        println!("{df}");
        Ok(())
```

Alternative: eager approach (immediate execution)
```
use polars::prelude::*;

let df = CsvReader::from_path("data.csv")?
    .finish()?;        // загрузка уже выполнена

let filtered = df.filter(&col("age").gt(30))?;   // выполняется сразу
```
Why you should almost always use Lazy? Because the lazy plan knows the entire operation graph, so it can:
- drop unused columns before reading the file
- push down filters (filter pushdown)
- combine expressions
- eliminate duplicate computations
- optimize joins
- reorder operations for maximum efficiency

In essence, enabling the lazy feature turns Polars into a full‑fledged query optimizer, on par with Spark, DuckDB, and DataFusion.

### ndarray & serde explainer

```
cargo add ndarray -F serde
```

The serde feature allows arrays (Array, Array2, ArrayD, …) to be serialized into JSON, CBOR, and other formats, and deserialized back.
Without serde, ndarray cannot perform serialization.
The serde feature adds Serialize / Deserialize implementations for ndarray types.