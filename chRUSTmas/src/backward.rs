use ndarray::prelude::*;

use crate::activations::{relu_prime, sigmoid_prime};
use crate::cache::{ActivationCache, LinearCache};

/// Backward propagation for sigmoid activation function.
/// Computes the backward propagation (i.e. the gradient of the loss) with respect to the input of the sigmoid activation function.
/// It takes the derivative of the cost function with respect to the activation da and the activation cache activation_cache.
/// It performs an element-wise multiplication between da and the derivative of the sigmoid function applied to the values in the activation cache, activation_cache.z.
pub fn sigmoid_backward(da: &Array2<f32>, activation_cache: ActivationCache) -> Array2<f32> {
    da * activation_cache.z.mapv(|x| sigmoid_prime(&x))
}

/// Backward propagation for ReLU activation function.
/// Computes the backward propagation for the ReLU activation function. 
/// It takes the derivative of the cost function with respect to the activation da and the activation cache activation_cache. 
/// It performs an element-wise multiplication between da and the derivative of the ReLU function applied to the values in the activation cache, activation_cache.z.
pub fn relu_backward(da: &Array2<f32>, activation_cache: ActivationCache) -> Array2<f32> {
    da * activation_cache.z.mapv(|x| relu_prime(&x))
}

/// Calculates the backward propagation for the linear component of a layer. 
/// It takes the gradient of the cost function with respect to the linear output `dz` and the linear cache linear_cache. 
/// It returns the gradients with respect to the previous layer's activation `da_prev`, the weights `dw`, and the biases `db`.
/// 
/// The function first extracts the previous layer's activation `a_prev`, the weight matrix `w`, and the bias matrix `_b` from the linear cache. 
/// It computes the number of training examples `m` by accessing the shape of `a_prev` and dividing the number of examples by `m`.
/// 
/// The function then calculates the gradient of the weights `dw` using the dot product between `dz` and the transposed `a_prev`, scaled by `1/m`. 
/// It computes the gradient of the biases `db` by summing the elements of `dz` along Axis(1) and scaling the result by `1/m`. 
/// Finally, it computes the gradient of the previous layer's activation `da_prev` by performing the dot product between the transposed `w` and `dz`.
///
/// The function returns `da_prev`, `dw`, and `db` as a tuple.
pub fn linear_backward(
    dz: &Array2<f32>,
    linear_cache: LinearCache,
) -> (Array2<f32>, Array2<f32>, Array2<f32>) {
    let (a_prev, w, _b) = (linear_cache.a, linear_cache.w, linear_cache.b);
    let m = a_prev.shape()[1] as f32;
    let dw = (1.0 / m) * (dz.dot(&a_prev.reversed_axes()));
    let db_vec = ((1.0 / m) * dz.sum_axis(Axis(1))).to_vec();
    let db = Array2::from_shape_vec((db_vec.len(), 1), db_vec).unwrap();
    let da_prev = w.reversed_axes().dot(dz);

    (da_prev, dw, db)
}

/// Combines the linear backward propagation and activation backward propagation steps for a layer in a neural network. 
/// It takes the derivative of the cost function with respect to the activation `da`, a tuple containing the linear cache and activation cache, and a string indicating the activation function used.
/// Depending on the specified activation function, it computes the derivative of the cost function with respect to the linear output `dz` using the appropriate backward activation function (sigmoid or ReLU). 
/// It then calls the `linear_backward` function to compute the gradients with respect to the previous layer's activation `da_prev`, weights `dw`, and biases `db`.
/// The function returns `da_prev`, `dw`, and `db` as a tuple.
pub fn linear_backward_activation(
    da: &Array2<f32>,
    cache: (LinearCache, ActivationCache),
    activation: &str,
) -> (Array2<f32>, Array2<f32>, Array2<f32>) {
    let (linear_cache, activation_cache) = cache;

    match activation {
        "sigmoid" => {
            let dz = sigmoid_backward(da, activation_cache);
            linear_backward(&dz, linear_cache)
        }
        "relu" => {
            let dz = relu_backward(da, activation_cache);
            linear_backward(&dz, linear_cache)
        }
        _ => panic!("wrong activation string"),
    }
}
