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

struct DeepNeuralNetwork{
    pub layers: Vec<usize>,
    pub learning_rate: f32,
}

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
            // Create 2D array with values between -1 and 1.
            let weight_matrix = Array::from_shape_fn(
                (self.layers[l], self.layers[l - 1]),
                |_| between.sample(&mut rng),
            );

            let bias_matrix = Array::zeros((self.layers[l], 1));

            let weight_string = ["W", &l.to_string()].join("").to_string();
            let biases_string = ["b", &l.to_string()].join("").to_string();

            parameters.insert(weight_string, weight_matrix);
            parameters.insert(biases_string, bias_matrix);
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
