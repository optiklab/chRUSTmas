use ch_rust_mas::*;
use std::env;
fn main() {
    // SAFETY: This is called at the start of main before any threads are spawned
    unsafe { env::set_var("RUST_BACKTRACE", "1") };

    // Set the neural network layers, learning rate, and number of iterations.
    let neural_network_layers: Vec<usize> = vec![12288, 20, 7, 5, 1];
    let learning_rate = 0.0075;
    let iterations = 1000;

    // Load the training and test data from CSV files.
    let (training_data, training_labels) =
        dataframe_from_csv("datasets/training_set.csv".into()).unwrap();
    // Convert the dataframes to arrays and normalize the pixel values to the range [0, 1].
    let (test_data, test_labels) = dataframe_from_csv("datasets/test_set.csv".into()).unwrap();

    let training_data_array = array_from_dataframe(&training_data)/255.0;
    let training_labels_array = array_from_dataframe(&training_labels);
    let test_data_array = array_from_dataframe(&test_data)/255.0;
    let test_labels_array = array_from_dataframe(&test_labels);

    // Create an instance of the DeepNeuralNetwork struct with the specified layers and learning rate.
    let model = DeepNeuralNetwork {
        layers: neural_network_layers,
        learning_rate,
    };

    // Initialize the parameters.
    let parameters = model.initialize_parameters();

    // Train the model, passing in the training data, training labels, initial parameters, iterations, and learning rate.
    let parameters = model.train_model(
        &training_data_array,
        &training_labels_array,
        parameters,
        iterations,
        model.learning_rate,
    );

    // Write the trained parameters to a JSON file.
    write_parameters_to_json_file(&parameters, "model.json".into());

    // Predict the labels for the training and test data.
    let training_predictions = model.predict(&training_data_array, &parameters);
    println!(
        "Training Set Accuracy: {}%",
        model.score(&training_predictions, &training_labels_array)
    );

    let test_predictions = model.predict(&test_data_array, &parameters);
    
    // Calculate and print the accuracy scores for the training and test predictions.
    println!(
        "Test Set Accuracy: {}%",
        model.score(&test_predictions, &test_labels_array)
    );
}