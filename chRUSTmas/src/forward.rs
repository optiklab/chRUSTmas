use ndarray::prelude::*;

use crate::activations::{relu_activation, sigmoid_activation};
use crate::cache::{ActivationCache, LinearCache};

/// Takes the activation matrix `a`, weight matrix `w`, and bias matrix `b` as inputs with the goal to compute the logit matrix `z` for a specific layer in a neural network.
/// It calculates the logit matrix for each layer using the following expression:
///    ```text
///    Z[l] = W[l]A[l-1] + b[l]
///    ```
/// In simpler terms, the logit matrix for layer `l` is obtained by taking the dot product of the weight matrix `W[l]` and the activation matrix `A[l-1]` from the previous layer, and then adding the bias matrix `b[l]`. 
/// This step represents the linear transformation of the inputs for the current layer by calculating the dot product of `w` and `a`, and then adding `b` to the result. 
/// The resulting matrix `z` represents the logits of the layer. The function returns `z` along with a LinearCache struct that stores the input matrices for later use in backward propagation.
pub fn linear_forward(
    a: &Array2<f32>,
    w: &Array2<f32>,
    b: &Array2<f32>,
) -> (Array2<f32>, LinearCache) {
    let z = w.dot(a) + b;

    let cache = LinearCache {
        a: a.clone(),
        w: w.clone(),
        b: b.clone(),
    };
    return (z, cache);
}

/// Takes the activation matrix a, weight matrix w, bias matrix b and additional activation parameter indicating the activation function to be applied as inputs.
/// To perform forward propagation, we need to calculate logit matrix `z` using linear_forward function and then apply the specified activation function to compute the activation matrix `a_next` and the activation cache. 
/// i.e. we need to follow these two steps for each layer:
/// 1. Calculate the logit matrix z for each layer using the following expression:
///    ```text
///    Z[l] = W[l]A[l-1] + b[l]
///    ```
///    In simpler terms, the logit matrix for layer `l` is obtained by taking the dot product of the weight matrix `W[l]` and the activation matrix `A[l-1]` from the previous layer, and then adding the bias matrix `b[l]`. This step represents the linear transformation of the inputs for the current layer.
/// 
/// 2. Calculate the activation matrix from the logit matrix using an activation function:
///    A[l] = ActivationFunction(Z[l])
/// 
///    Here, the activation function can be any non-linear function applied element-wise to the elements of the logit matrix. Popular activation functions include sigmoid, tanh, and relu. 
///    In our model, we will use the relu activation function for all intermediate layers and sigmoid for the last layer (classifier layer). 
///    This step introduces non-linearity into the network, allowing it to learn and model complex relationships in the data.
/// 
/// For `n[l]` number of hidden units in layer `l` and m number of examples, these are the shapes of each matrix:
/// ```text
/// Z[l] ⇾ [n[l] x m]
/// W[l] ⇾ [n[l] x n[l-1]] - i.e. number of connections between hidden units in current layer (rows) and previous layer (columns)
/// b[l] ⇾ [n[l] x 1]
/// A[l] ⇾ [n[l] x m]
/// ```
/// The function returns a_next along with a tuple of the linear cache and activation cache, wrapped in a Result enum. 
/// If the specified activation function is not supported, an error message is returned.
pub fn linear_forward_activation(
    a: &Array2<f32>,
    w: &Array2<f32>,
    b: &Array2<f32>,
    activation: &str,
) -> Result<(Array2<f32>, (LinearCache, ActivationCache)), String> {
    match activation {
        "sigmoid" => {
            let (z, linear_cache) = linear_forward(a, w, b);
            let (a_next, activation_cache) = sigmoid_activation(z);
            return Ok((a_next, (linear_cache, activation_cache)));
        }
        "relu" => {
            let (z, linear_cache) = linear_forward(a, w, b);
            let (a_next, activation_cache) = relu_activation(z);
            return Ok((a_next, (linear_cache, activation_cache)));
        }
        _ => return Err("wrong activation string".to_string()),
    }
}
