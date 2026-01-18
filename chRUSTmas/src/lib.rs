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

pub fn array_from_dataframe(df: &DataFrame) -> Array2<f32> {
    df.to_ndarray::<Float32Type>(IndexOrder::C).unwrap().reversed_axes()
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

// Takes the activation matrix a, weight matrix w, and bias matrix b as inputs with the goal to compute the logit matrix z for a specific layer in a neural network.
// It calculates the logit matrix for each layer using the following expression:
//    Z[l] = W[l]A[l-1] + b[l]
// In simpler terms, the logit matrix for layer l is obtained by taking the dot product of the weight matrix W[l] and the activation matrix A[l-1] from the previous layer, and then adding the bias matrix b[l]. 
// This step represents the linear transformation of the inputs for the current layer by calculating the dot product of w and a, and then adding b to the result. 
// The resulting matrix z represents the logits of the layer. The function returns z along with a LinearCache struct that stores the input matrices for later use in backward propagation.
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
// If z is greater than zero, the function returns z; otherwise, it returns zero. 
// ReLU is a popular activation function that introduces non-linearity and helps the network learn complex patterns.
pub fn relu(z: &f32) -> f32 {
    match *z > 0.0 {
        true => *z,
        false => 0.0,
    }
}

// Matrix-based sigmoid activation function. Takes a 2D matrix z as input and apply the respective activation function element-wise using the mapv function.
// The resulting activation matrix is returned along with an ActivationCache struct that stores the corresponding logit matrix.
pub fn sigmoid_activation(z: Array2<f32>) -> (Array2<f32>, ActivationCache) {
    (z.mapv(|x| sigmoid(&x)), ActivationCache { z })
}

// Matrix-based ReLU activation function. Takes a 2D matrix z as input and apply the respective activation function element-wise using the mapv function.
// The resulting activation matrix is returned along with an ActivationCache struct that stores the corresponding logit matrix.
pub fn relu_activation(z: Array2<f32>) -> (Array2<f32>, ActivationCache) {
    (z.mapv(|x| relu(&x)), ActivationCache { z })
}

// Stores the logit matrix z for each layer. 
// This cache is essential for later stages, such as backpropagation, where the stored values are required.
#[derive(Clone, Debug)]
pub struct ActivationCache {
    pub z: Array2<f32>, // Z[l]: Logit Matrix for layer l. It represents the linear transformation of the inputs for a particular layer.
}

// Takes the activation matrix a, weight matrix w, bias matrix b and additional activation parameter indicating the activation function to be applied as inputs.
// To perform forward propagation, we need to calculate logit matrix z using linear_forward function and then apply the specified activation function to compute the activation matrix a_next and the activation cache. 
// i.e. we need to follow these two steps for each layer:
// 1. Calculate the logit matrix z for each layer using the following expression:
//    Z[l] = W[l]A[l-1] + b[l]
//    In simpler terms, the logit matrix for layer l is obtained by taking the dot product of the weight matrix W[l] and the activation matrix A[l-1] from the previous layer, and then adding the bias matrix b[l]. This step represents the linear transformation of the inputs for the current layer.
// 
// 2. Calculate the activation matrix from the logit matrix using an activation function:
//    A[l] = ActivationFunction(Z[l])
// 
//    Here, the activation function can be any non-linear function applied element-wise to the elements of the logit matrix. Popular activation functions include sigmoid, tanh, and relu. 
//    In our model, we will use the relu activation function for all intermediate layers and sigmoid for the last layer (classifier layer). 
//    This step introduces non-linearity into the network, allowing it to learn and model complex relationships in the data.
// 
// For n[l] number of hidden units in layer l and m number of examples, these are the shapes of each matrix:
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
