use ndarray::prelude::*;
use polars::prelude::*; // Polars is a DataFrame library for Rust. It is based on Apache Arrow’s memory model. 
                        // Apache Arrow provides very cache efficient columnar data structures and 
                        // is becoming the defacto standard for columnar data.

                        // Polars is named after polar bears which are one of the fastest and most powerful animals 
                        // in their environment - just like Polars aims to be the fastest DataFrame library.

                        // Besides that, all Polars functions return the same error type, making error handling predictable.
use rand::distr::Uniform;
use rand::prelude::*;
use std::collections::HashMap;
use std::fs::File;
use std::path::PathBuf;
use std::f32::consts::E;
use num_integer::Roots;

// Converts a Polars DataFrame to a 2D ndarray Array of f32 type.
pub fn array_from_dataframe(df: &DataFrame) -> Array2<f32> {
    df.to_ndarray::<Float32Type>(IndexOrder::C).unwrap().reversed_axes()
}

// Trait to compute natural logarithm of each element in a 2D array.
trait Log {
    fn log(&self) -> Array2<f32>;
}

// Implementation of the Log trait for 2D ndarray Array of f32 type.
impl Log for Array2<f32> {
    fn log(&self) -> Array2<f32> {
        self.mapv(|x| x.log(std::f32::consts::E))
    }
}

pub fn dataframe_from_csv(file_path: PathBuf) -> PolarsResult<(DataFrame, DataFrame)> {
    let file = File::open(file_path).map_err(|e| PolarsError::ComputeError(e.to_string().into()))?;
    let data = CsvReader::new(file).finish()?;

    // Now, splitting loaded DataFrame (data) into features and labels
    // to prepare data for machine learning.

    // Create a new DataFrame that contains all the columns from the original CSV
    // except the column named "y". This represents input features (X).
    let training_dataset = data.drop("y")?;

    // Create a separate DataFrame that contains only the column named "y". 
    // This represents target variable (labels).
    let training_labels = data.select(["y"])?;

    return Ok((training_dataset, training_labels));
}

// Stores the intermediate values needed for each layer. It includes the activation matrix a, weight matrix w, and bias matrix b. 
// These matrices are used to calculate the logit matrix z in the forward propagation process.
#[derive(Clone, Debug)]
pub struct LinearCache {
    pub a: Array2<f32>, // A[l]: Activation matrix for layer l. It represents the output or activation values of the neurons in a specific layer.
    pub w: Array2<f32>, // W[l]: Weights matrix for layer l. It contains the weights connecting the neurons of layer l-1 to the neurons of layer l.
    pub b: Array2<f32>, // b[l]: Bias matrix for layer l. It contains the bias values added to the linear transformation of the inputs for layer l.
}

// Takes the activation matrix `a`, weight matrix `w`, and bias matrix `b` as inputs with the goal to compute the logit matrix `z` for a specific layer in a neural network.
// It calculates the logit matrix for each layer using the following expression:
//    ```
//    Z[l] = W[l]A[l-1] + b[l]
//    ```
// In simpler terms, the logit matrix for layer `l` is obtained by taking the dot product of the weight matrix `W[l]` and the activation matrix `A[l-1]` from the previous layer, and then adding the bias matrix `b[l]`. 
// This step represents the linear transformation of the inputs for the current layer by calculating the dot product of `w` and `a`, and then adding `b` to the result. 
// The resulting matrix `z` represents the logits of the layer. The function returns `z` along with a LinearCache struct that stores the input matrices for later use in backward propagation.
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

// Activation functions below introduce non-linearity to neural networks and play a crucial role in the forward propagation process. 
// The code provides implementations for two commonly used activation functions: sigmoid and relu.

// Non-linear activation function sigmoid.
// The sigmoid function maps the input value to a range between 0 and 1, enabling the network to model non-linear relationships.
pub fn sigmoid(z: &f32) -> f32 {
    1.0 / (1.0 + E.powf(-z))
}

// Non-linear activation function relu (Rectified Linear Unit).
// If `z` is greater than zero, the function returns `z`; otherwise, it returns zero. 
// ReLU is a popular activation function that introduces non-linearity and helps the network learn complex patterns.
pub fn relu(z: &f32) -> f32 {
    match *z > 0.0 {
        true => *z,
        false => 0.0,
    }
}

// Matrix-based sigmoid activation function. Takes a 2D matrix `z` as input and apply the respective activation function element-wise using the mapv function.
// The resulting activation matrix is returned along with an ActivationCache struct that stores the corresponding logit matrix.
pub fn sigmoid_activation(z: Array2<f32>) -> (Array2<f32>, ActivationCache) {
    (z.mapv(|x| sigmoid(&x)), ActivationCache { z })
}

// Matrix-based ReLU activation function. Takes a 2D matrix `z` as input and apply the respective activation function element-wise using the mapv function.
// The resulting activation matrix is returned along with an ActivationCache struct that stores the corresponding logit matrix.
pub fn relu_activation(z: Array2<f32>) -> (Array2<f32>, ActivationCache) {
    (z.mapv(|x| relu(&x)), ActivationCache { z })
}

// Stores the logit matrix z for each layer. 
// This cache is essential for later stages, such as backpropagation, where the stored values are required.
#[derive(Clone, Debug)]
pub struct ActivationCache {
    pub z: Array2<f32>, // `Z[l]`: Logit Matrix for layer `l`. It represents the linear transformation of the inputs for a particular layer.
}

// Takes the activation matrix a, weight matrix w, bias matrix b and additional activation parameter indicating the activation function to be applied as inputs.
// To perform forward propagation, we need to calculate logit matrix `z` using linear_forward function and then apply the specified activation function to compute the activation matrix `a_next` and the activation cache. 
// i.e. we need to follow these two steps for each layer:
// 1. Calculate the logit matrix z for each layer using the following expression:
//    ```
//    Z[l] = W[l]A[l-1] + b[l]
//    ```
//    In simpler terms, the logit matrix for layer `l` is obtained by taking the dot product of the weight matrix `W[l]` and the activation matrix `A[l-1]` from the previous layer, and then adding the bias matrix `b[l]`. This step represents the linear transformation of the inputs for the current layer.
// 
// 2. Calculate the activation matrix from the logit matrix using an activation function:
//    A[l] = ActivationFunction(Z[l])
// 
//    Here, the activation function can be any non-linear function applied element-wise to the elements of the logit matrix. Popular activation functions include sigmoid, tanh, and relu. 
//    In our model, we will use the relu activation function for all intermediate layers and sigmoid for the last layer (classifier layer). 
//    This step introduces non-linearity into the network, allowing it to learn and model complex relationships in the data.
// 
// For `n[l]` number of hidden units in layer `l` and m number of examples, these are the shapes of each matrix:
// 
// Z[l] ⇾ [n[l] x m]
// W[l] ⇾ [n[l] x n[l-1]] - i.e. number of connections between hidden units in current layer (rows) and previous layer (columns)
// b[l] ⇾ [n[l] x 1]
// A[l] ⇾ [n[l] x m]
// 
// The function returns a_next along with a tuple of the linear cache and activation cache, wrapped in a Result enum. 
// If the specified activation function is not supported, an error message is returned.
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

// ------------------------------------- Backward Activations / Backward propagation functions below -------------------------------------

// Derivative of sigmoid activation function.
// Calculates the derivative of the sigmoid activation function.
// It takes the input `z` and returns the derivative value, which is computed as the sigmoid of `z` multiplied by 1.0 minus the sigmoid of `z`.
pub fn sigmoid_prime(z: &f32) -> f32 {
    sigmoid(z) * (1.0 - sigmoid(z))
}

// Derivative of ReLU activation function.
// Computes the derivative of the ReLU activation function. It takes the input `z` and returns 1.0 if `z` is greater than 0, and 0.0 otherwise.
pub fn relu_prime(z: &f32) -> f32 {
    match *z > 0.0 {
        true => 1.0,
        false => 0.0,
    }
}

// Backward propagation for sigmoid activation function.
// Computes the backward propagation (i.e. the gradient of the loss) with respect to the input of the sigmoid activation function.
// It takes the derivative of the cost function with respect to the activation da and the activation cache activation_cache.
// It performs an element-wise multiplication between da and the derivative of the sigmoid function applied to the values in the activation cache, activation_cache.z.
pub fn sigmoid_backward(da: &Array2<f32>, activation_cache: ActivationCache) -> Array2<f32> {
    da * activation_cache.z.mapv(|x| sigmoid_prime(&x))
}

// Backward propagation for ReLU activation function.
// Computes the backward propagation for the ReLU activation function. 
// It takes the derivative of the cost function with respect to the activation da and the activation cache activation_cache. 
// It performs an element-wise multiplication between da and the derivative of the ReLU function applied to the values in the activation cache, activation_cache.z.
pub fn relu_backward(da: &Array2<f32>, activation_cache: ActivationCache) -> Array2<f32> {
    da * activation_cache.z.mapv(|x| relu_prime(&x))
}

// Calculates the backward propagation for the linear component of a layer. 
// It takes the gradient of the cost function with respect to the linear output `dz` and the linear cache linear_cache. 
// It returns the gradients with respect to the previous layer’s activation `da_prev`, the weights `dw`, and the biases `db`.
// 
// The function first extracts the previous layer’s activation `a_prev`, the weight matrix `w`, and the bias matrix `_b` from the linear cache. 
// It computes the number of training examples `m` by accessing the shape of `a_prev` and dividing the number of examples by `m`.
// 
// The function then calculates the gradient of the weights `dw` using the dot product between `dz` and the transposed `a_prev`, scaled by `1/m`. 
// It computes the gradient of the biases `db` by summing the elements of `dz` along Axis(1) and scaling the result by `1/m`. 
// Finally, it computes the gradient of the previous layer’s activation `da_prev` by performing the dot product between the transposed `w` and `dz`.
//
// The function returns `da_prev`, `dw`, and `db` as a tuple.
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

// ------------------------------------- End of Backward Activations / Backward propagation functions below -------------------------------------

struct DeepNeuralNetwork {
    pub layers: Vec<usize>,
    pub learning_rate: f32,
}

impl DeepNeuralNetwork {
    /// Initializes the parameters of the neural network.
    ///
    /// ### Returns
    /// a Hashmap dictionary of randomly initialized weights and biases.
    pub fn initialize_parameters(&self) -> HashMap<String, Array2<f32>> {
        let between = Uniform::new(-1.0f32, 1.0f32).unwrap(); // random number between -1 and 1
        let mut rng = rand::rng(); // random number generator

        let number_of_layers = self.layers.len();

        let mut parameters: HashMap<String, Array2<f32>> = HashMap::new();

        // Zeroth layer is the input layer.
        // Start the loop from the first hidden layer (not zeroth input layer) to the output layer. 
        for l in 1..number_of_layers {
            // Create 2D array with values randomly initialized between -1 and 1.
            let weight_matrix = Array::from_shape_fn(
                (self.layers[l], self.layers[l - 1]),
                |_| between.sample(&mut rng),
            );

            let bias_matrix = Array::zeros((self.layers[l], 1)); // bias matrix [n, 1] initialized to zero

            let weight_string = ["W", &l.to_string()].join("").to_string();
            let biases_string = ["b", &l.to_string()].join("").to_string();

            parameters.insert(weight_string, weight_matrix);
            parameters.insert(biases_string, bias_matrix);
        }
        
        parameters
    }

    // Forward propagation implementation.
    // Takes the input matrix x and the parameters (weights and biases) as inputs.
    // By performing forward propagation, our neural network takes the input data through all the layers,
    // applying linear transformations and activation functions, and eventually produces a prediction or output at the final layer.
    // During the forward propagation process, we will store the weight matrix, bias matrix, and logit matrix as cache. 
    // This stored information will prove useful in the subsequent step of backward propagation, where we update the model’s parameters based on the computed gradients.
    pub fn forward(
        &self,
        x: &Array2<f32>,
        parameters: &HashMap<String, Array2<f32>>,
    ) -> (Array2<f32>, HashMap<String, (LinearCache, ActivationCache)>) {
        let number_of_layers = self.layers.len()-1;

        let mut a = x.clone(); // Initializes the a matrix as a copy of x (input data)
        let mut caches = HashMap::new(); // Creates an empty hashmap caches to store the caches for each layer.

        for l in 1..number_of_layers { // Iterates over each layer (except the last layer) in a for loop

            // For each layer, it retrieves the corresponding weights w and biases b from the parameters using string concatenation.
            let w_string = ["W", &l.to_string()].join("").to_string();
            let b_string = ["b", &l.to_string()].join("").to_string();

            let w = &parameters[&w_string];
            let b = &parameters[&b_string];

            // Apply linear transformations and activation functions to the layer
            let (a_temp, cache_temp) = linear_forward_activation(&a, w, b, "relu").unwrap();

            a = a_temp; // current activation matrix will be used as input activation matrix for next layer

            // The resulting activation matrix a_temp and the cache cache_temp are stored in the caches hashmap using the layer index as the key. 
            caches.insert(l.to_string(), cache_temp);
        }

        // We are done processing all intermediate layers with ReLU. Now, let's compute activation of last layer with sigmoid
        let weight_string = ["W", &(number_of_layers).to_string()].join("").to_string();
        let bias_string = ["b", &(number_of_layers).to_string()].join("").to_string();

        let w = &parameters[&weight_string];
        let b = &parameters[&bias_string];

        let (al, cache) = linear_forward_activation(&a, w, b, "sigmoid").unwrap();
        caches.insert(number_of_layers.to_string(), cache);

        // Finally, the method returns the final activation matrix al and the caches hashmap containing all the caches for each layer. 
        // Here al is the activation of the final layer and will be used to make the predictions during the inference part of our process.
        return (al, caches);
    }

    // Computes the cost of the predictions made by the neural network.
    // It takes the final activation matrix al (predicted outputs) and the true labels y as inputs.
    // The cost is calculated using the cross-entropy loss function, which measures the difference between the predicted probabilities and the true labels.
    pub fn cost(&self, al: &Array2<f32>, y: &Array2<f32>) -> f32 {
        let m = y.shape()[1] as f32;
        let cost = -(1.0 / m)
            * (y.dot(&al.clone().reversed_axes().log())
                + (1.0 - y).dot(&(1.0 - al).reversed_axes().log()));

        return cost.sum();
    }
    
    // Performs the backward propagation algorithm to calculate the gradients of the cost function with respect to the parameters (weights and biases) of each layer.
    // The method takes the final activation `al` obtained from the forward propagation, the true labels `y`, and the caches containing the linear and activation values for each layer.
    // Returns the grads map containing the gradients of the cost function with respect to each parameter of the neural network.
    // These gradients will be used in the optimization step to update the parameters and minimize the cost.
	pub fn backward(
        &self,
        al: &Array2<f32>,
        y: &Array2<f32>,
        caches: HashMap<String, (LinearCache, ActivationCache)>,
    ) -> HashMap<String, Array2<f32>> {
        
        // First, it initializes an empty HashMap called grads to store the gradients. 
        let mut grads = HashMap::new();

        let num_of_layers = self.layers.len() - 1;

        // Compute the initial derivative of the cost function with respect to `al` using the provided formula.
        let dal = -(y / al - (1.0 - y) / (1.0 - al));

        // Starting from the last layer (output layer), it retrieves the cache for the current layer...
        let current_cache = caches[&num_of_layers.to_string()].clone();

        //... and calls the linear_backward_activation function to calculate the gradients of the cost function with respect to the parameters of that layer. 
        // The activation function used is “sigmoid” for the last layer. The computed gradients for weights, biases, and activation are stored in the grads map.
        let (mut da_prev, mut dw, mut db) =
            linear_backward_activation(&dal, current_cache, "sigmoid");

        let weight_string = ["dW", &num_of_layers.to_string()].join("").to_string();
        let bias_string = ["db", &num_of_layers.to_string()].join("").to_string();
        let activation_string = ["dA", &num_of_layers.to_string()].join("").to_string();

        grads.insert(weight_string, dw);
        grads.insert(bias_string, db);
        grads.insert(activation_string, da_prev.clone());
 
        // Iterate over the remaining layers in reverse order.
        for l in (1..num_of_layers).rev() {

            // For each layer, retrieve the cache, 
            let current_cache = caches[&l.to_string()].clone();

            // call the linear_backward_activation function to calculate the gradients, 
            (da_prev, dw, db) =
                linear_backward_activation(&da_prev, current_cache, "relu");

            // and store the computed gradients in the grads map.
            let weight_string = ["dW", &l.to_string()].join("").to_string();
            let bias_string = ["db", &l.to_string()].join("").to_string();
            let activation_string = ["dA", &l.to_string()].join("").to_string();

            grads.insert(weight_string, dw);
            grads.insert(bias_string, db);
            grads.insert(activation_string, da_prev.clone());
        }

        // Return the grads map containing the gradients of the cost function with respect to each parameter of the neural network.
        grads
    }

    // Go through each layer and update the parameters in the HashMap for each layer by using the HashMap of gradients in that layer. 
    // Return the updated parameters.
    pub fn update_parameters(
        &self,
        params: &HashMap<String, Array2<f32>>,
        grads: HashMap<String, Array2<f32>>,
        m: f32, 
        learning_rate: f32,

    ) -> HashMap<String, Array2<f32>> {
        let mut parameters = params.clone();
        let num_of_layers = self.layer_dims.len() - 1;
        for l in 1..num_of_layers + 1 {
            let weight_string_grad = ["dW", &l.to_string()].join("").to_string();
            let bias_string_grad = ["db", &l.to_string()].join("").to_string();
            let weight_string = ["W", &l.to_string()].join("").to_string();
            let bias_string = ["b", &l.to_string()].join("").to_string();

            *parameters.get_mut(&weight_string).unwrap() = parameters[&weight_string].clone()
                - (learning_rate * (grads[&weight_string_grad].clone() + (self.lambda/m) *parameters[&weight_string].clone()) );
            *parameters.get_mut(&bias_string).unwrap() = parameters[&bias_string].clone()
                - (learning_rate * grads[&bias_string_grad].clone());
        }
        parameters
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dataframe_from_csv() {
        let file_path = PathBuf::from("datasets/training_set.csv");
        let result = dataframe_from_csv(file_path);
        assert!(result.is_ok());
        let (training, labels) = result.unwrap();
        assert!(!training.is_empty());
        assert!(!labels.is_empty());
    }

    #[test]
    fn test_initialize_parameters() {
        // Create a network with 3 layers: input (784), hidden (128), output (10)
        let network = DeepNeuralNetwork {
            layers: vec![784, 128, 10],
            learning_rate: 0.01,
        };

        let parameters = network.initialize_parameters();

        // Should have 4 entries: W1, b1, W2, b2
        assert_eq!(parameters.len(), 4);

        // Check W1 shape: (128, 784)
        let w1 = parameters.get("W1").expect("W1 should exist");
        assert_eq!(w1.shape(), &[128, 784]);

        // Check b1 shape: (128, 1)
        let b1 = parameters.get("b1").expect("b1 should exist");
        assert_eq!(b1.shape(), &[128, 1]);

        // Check W2 shape: (10, 128)
        let w2 = parameters.get("W2").expect("W2 should exist");
        assert_eq!(w2.shape(), &[10, 128]);

        // Check b2 shape: (10, 1)
        let b2 = parameters.get("b2").expect("b2 should exist");
        assert_eq!(b2.shape(), &[10, 1]);

        // Verify weights are in range [-1, 1]
        for val in w1.iter() {
            assert!(*val >= -1.0 && *val <= 1.0, "Weight value out of range: {}", val);
        }

        // Verify biases are initialized to 0
        for val in b1.iter() {
            assert_eq!(*val, 0.0, "Bias should be initialized to 0");
        }
        for val in b2.iter() {
            assert_eq!(*val, 0.0, "Bias should be initialized to 0");
        }
    }
}
