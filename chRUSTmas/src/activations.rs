use ndarray::prelude::*;
use std::f32::consts::E;

use crate::cache::ActivationCache;

// Activation functions below introduce non-linearity to neural networks and play a crucial role in the forward propagation process. 
// The code provides implementations for two commonly used activation functions: sigmoid and relu.

/// Non-linear activation function sigmoid.
/// The sigmoid function maps the input value to a range between 0 and 1, enabling the network to model non-linear relationships.
pub fn sigmoid(z: &f32) -> f32 {
    1.0 / (1.0 + E.powf(-z))
}

/// Non-linear activation function relu (Rectified Linear Unit).
/// If `z` is greater than zero, the function returns `z`; otherwise, it returns zero. 
/// ReLU is a popular activation function that introduces non-linearity and helps the network learn complex patterns.
pub fn relu(z: &f32) -> f32 {
    match *z > 0.0 {
        true => *z,
        false => 0.0,
    }
}

// ----------------------------------------------------- Activation functions for forward propagation -----------------------------------------------------

/// Matrix-based sigmoid activation function. Takes a 2D matrix `z` as input and apply the respective activation function element-wise using the mapv function.
/// The resulting activation matrix is returned along with an ActivationCache struct that stores the corresponding logit matrix.
pub fn sigmoid_activation(z: Array2<f32>) -> (Array2<f32>, ActivationCache) {
    (z.mapv(|x| sigmoid(&x)), ActivationCache { z })
}

/// Matrix-based ReLU activation function. Takes a 2D matrix `z` as input and apply the respective activation function element-wise using the mapv function.
/// The resulting activation matrix is returned along with an ActivationCache struct that stores the corresponding logit matrix.
pub fn relu_activation(z: Array2<f32>) -> (Array2<f32>, ActivationCache) {
    (z.mapv(|x| relu(&x)), ActivationCache { z })
}

// ----------------------------------------------------- Derivative functions for backward propagation -----------------------------------------------------

/// Derivative of sigmoid activation function.
/// Calculates the derivative of the sigmoid activation function.
/// It takes the input `z` and returns the derivative value, which is computed as the sigmoid of `z` multiplied by 1.0 minus the sigmoid of `z`.
pub fn sigmoid_prime(z: &f32) -> f32 {
    sigmoid(z) * (1.0 - sigmoid(z))
}

/// Derivative of ReLU activation function.
/// Computes the derivative of the ReLU activation function. It takes the input `z` and returns 1.0 if `z` is greater than 0, and 0.0 otherwise.
pub fn relu_prime(z: &f32) -> f32 {
    match *z > 0.0 {
        true => 1.0,
        false => 0.0,
    }
}
