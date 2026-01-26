use ndarray::prelude::*;

/// Stores the intermediate values needed for each layer. It includes the activation matrix a, weight matrix w, and bias matrix b. 
/// These matrices are used to calculate the logit matrix z in the forward propagation process.
#[derive(Clone, Debug)]
pub struct LinearCache {
    pub a: Array2<f32>, // A[l]: Activation matrix for layer l. It represents the output or activation values of the neurons in a specific layer.
    pub w: Array2<f32>, // W[l]: Weights matrix for layer l. It contains the weights connecting the neurons of layer l-1 to the neurons of layer l.
    pub b: Array2<f32>, // b[l]: Bias matrix for layer l. It contains the bias values added to the linear transformation of the inputs for layer l.
}

/// Stores the logit matrix z for each layer. 
/// This cache is essential for later stages, such as backpropagation, where the stored values are required.
#[derive(Clone, Debug)]
pub struct ActivationCache {
    pub z: Array2<f32>, // `Z[l]`: Logit Matrix for layer `l`. It represents the linear transformation of the inputs for a particular layer.
}
