use ndarray::prelude::*;
use polars::prelude::*;
use std::collections::HashMap;
use std::fs::File;
use std::fs::OpenOptions;
use std::path::PathBuf;

/// Reads a CSV file from the specified file path and returns a tuple containing two Polars DataFrames:
/// 1. training_dataset: A DataFrame containing all columns except the "y" column (features).
/// 2. training_labels: A DataFrame containing only the "y" column (labels).
/// This function is useful for preparing data for machine learning tasks, where features and labels need to be separated.
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

/// Writes the parameters (weights and biases) of the neural network to a JSON file at the specified file path.
/// The parameters are stored in a HashMap where the keys are strings representing the parameter names (e.g., "W1", "b1") 
/// and the values are 2D ndarray Arrays of f32 type representing the parameter values. 
pub fn write_parameters_to_json_file(
    parameters: &HashMap<String, Array2<f32>>,
    file_path: PathBuf,
) {
    let file = OpenOptions::new()
        .create(true)
        .write(true)
        .truncate(true)
        .open(file_path)
        .unwrap();

    _ = serde_json::to_writer(file, parameters);
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
}
