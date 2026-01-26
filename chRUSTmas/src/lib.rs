use ndarray::prelude::*;
use rand::distr::Uniform;
use rand::prelude::*;
use std::collections::HashMap;
use num_integer::Roots;

// Module declarations
pub mod activations;
pub mod backward;
pub mod cache;
pub mod file_helpers;
pub mod array_helpers;
pub mod forward;

// Re-export public items for convenience
pub use activations::{relu, relu_activation, relu_prime, sigmoid, sigmoid_activation, sigmoid_prime};
pub use backward::{linear_backward, linear_backward_activation, relu_backward, sigmoid_backward};
pub use cache::{ActivationCache, LinearCache};
pub use file_helpers::{dataframe_from_csv, write_parameters_to_json_file};
pub use array_helpers::{array_from_dataframe, Log};
pub use forward::{linear_forward, linear_forward_activation};

#[derive(Clone, Debug)]
pub struct DeepNeuralNetwork {
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

            // Create a flattened weights array of (N * M) values
            let weight_array: Vec<f32> = (0..self.layers[l]*self.layers[l-1])
                .map(|_| between.sample(&mut rng))
                .collect();

            // Creates an n-dimensional (2D, actually) array rom a flat Vec by reshaping it into the specified dimensions (2)
            // Rows = self.layers[l] (number of neurons in current layer)
            // Columns = self.layers[l - 1] (number of neurons in previous layer)
            let weight_matrix = 
                Array2::from_shape_vec((self.layers[l], self.layers[l - 1]), weight_array).unwrap()
                    / (self.layers[l - 1]).sqrt() as f32;

            let bias_array: Vec<f32> = (0..self.layers[l]).map(|_| 0.0).collect();
            let bias_matrix = Array2::from_shape_vec((self.layers[l], 1), bias_array).unwrap();

            let weight_string = ["W", &l.to_string()].join("").to_string();
            let biases_string = ["b", &l.to_string()].join("").to_string();

            parameters.insert(weight_string, weight_matrix);
            parameters.insert(biases_string, bias_matrix);
        }
        
        parameters
    }

    /// Forward propagation implementation.
    /// Takes the input matrix x and the parameters (weights and biases) as inputs.
    /// By performing forward propagation, our neural network takes the input data through all the layers,
    /// applying linear transformations and activation functions, and eventually produces a prediction or output at the final layer.
    /// During the forward propagation process, we will store the weight matrix, bias matrix, and logit matrix as cache. 
    /// This stored information will prove useful in the subsequent step of backward propagation, where we update the model's parameters based on the computed gradients.
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

    /// Computes the cost of the predictions made by the neural network.
    /// It takes the final activation matrix al (predicted outputs) and the true labels y as inputs.
    /// The cost is calculated using the cross-entropy loss function, which measures the difference between the predicted probabilities and the true labels.
    pub fn cost(&self, al: &Array2<f32>, y: &Array2<f32>) -> f32 {
        let m = y.shape()[1] as f32;
        let cost = -(1.0 / m)
            * (y.dot(&al.clone().reversed_axes().log())
                + (1.0 - y).dot(&(1.0 - al).reversed_axes().log()));

        return cost.sum();
    }
    
    /// Performs the backward propagation algorithm to calculate the gradients of the cost function with respect to the parameters (weights and biases) of each layer.
    /// The method takes the final activation `al` obtained from the forward propagation, the true labels `y`, and the caches containing the linear and activation values for each layer.
    /// Returns the grads map containing the gradients of the cost function with respect to each parameter of the neural network.
    /// These gradients will be used in the optimization step to update the parameters and minimize the cost.
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
        // The activation function used is "sigmoid" for the last layer. The computed gradients for weights, biases, and activation are stored in the grads map.
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

    /// Go through each layer and update the parameters in the HashMap for each layer by using the HashMap of gradients in that layer. 
    /// Return the updated parameters.
    pub fn update_parameters(
        &self,
        params: &HashMap<String, Array2<f32>>,
        grads: HashMap<String, Array2<f32>>,
        learning_rate: f32,
    ) -> HashMap<String, Array2<f32>> {
        let mut parameters = params.clone();
        let num_of_layers = self.layers.len() - 1;
        for l in 1..num_of_layers + 1 {
            let weight_string_grad = ["dW", &l.to_string()].join("").to_string();
            let bias_string_grad = ["db", &l.to_string()].join("").to_string();
            let weight_string = ["W", &l.to_string()].join("").to_string();
            let bias_string = ["b", &l.to_string()].join("").to_string();

            *parameters.get_mut(&weight_string).unwrap() = parameters[&weight_string].clone()
                - (learning_rate * (grads[&weight_string_grad].clone()));
            *parameters.get_mut(&bias_string).unwrap() = parameters[&bias_string].clone()
                - (learning_rate * grads[&bias_string_grad].clone());
        }
        parameters
    }

    /// Training loop.
    /// Takes in 
    /// the training data: x_train_data, 
    /// training labels: y_train_data, 
    /// the parameters dictionary: parameters, 
    /// the number of training loop iterations: iterations,
    /// the learning_rate. 
    /// Returns the new parameters after a training iteration.
    pub fn train_model(
        &self,
        x_train_data: &Array2<f32>,
        y_train_data: &Array2<f32>,
        mut parameters: HashMap<String, Array2<f32>>,
        iterations: usize,
        learning_rate: f32,
    ) -> HashMap<String, Array2<f32>> {

        // It initializes an empty vector to store the cost values for each iteration.
        let mut costs: Vec<f32> = vec![];

        // In each iteration of the training loop, it performs the following steps:
        for i in 0..iterations {
            // Performs forward propagation to obtain the final activation al and the caches.
            let (al, caches) = self.forward(&x_train_data, &parameters);
            // Calculates the cost.
            let cost = self.cost(&al, &y_train_data);
            // Performs backward propagation to compute the gradients.
            let grads = self.backward(&al, &y_train_data, caches);
            // Updates the parameters with the computed gradients and the learning rate.
            parameters = self.update_parameters(&parameters, grads.clone(), learning_rate);

            // If the current iteration is a multiple of 100, it appends the cost value to the costs vector and prints the current epoch number and cost value.
            if i % 100 == 0 {
                costs.append(&mut vec![cost]);
                println!("Epoch : {}/{}    Cost: {:?}", i, iterations, cost);
            }
        }

        // Returns the updated parameters.
        parameters
    }

    /// Makes predictions on the test data using the trained parameters.
    pub fn predict(
        &self,
        x_test_data: &Array2<f32>,
        parameters: &HashMap<String, Array2<f32>>,
    ) -> Array2<f32> {
        // Obtain the final activation al.
        let (al, _) = self.forward(&x_test_data, &parameters);
        // Apply a threshold of 0.5 to the elements of al using the map method, converting values greater than 0.5 to 1.0 and values less than or equal to 0.5 to 0.0.
        let y_hat = al.map(|x| (x > &0.5) as i32 as f32);
        // Return the predicted labels as y_hat.
        y_hat
    }

    /// Calculates the accuracy score of the predicted labels compared to the actual test labels.
    pub fn score(&self, y_hat: &Array2<f32>, y_test_data: &Array2<f32>) -> f32 {
        
        // Calculates the element-wise absolute difference between the predicted labels y_hat and the actual test labels y_test_data.
        // Then sums the absolute differences using the sum method.
        // Then divides the sum by the number of examples (y_test_data.shape()[1]) and multiplies by 100.0 to get the error percentage.
        let error =
            (y_hat - y_test_data).map(|x| x.abs()).sum() / y_test_data.shape()[1] as f32 * 100.0;

        // Finally, subtracts the error percentage from 100.0 to get the accuracy score and returns it.
        100.0 - error
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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
