use ndarray::prelude::*;
use polars::prelude::*;

/// Converts a Polars DataFrame to a 2D ndarray Array of f32 type.
pub fn array_from_dataframe(df: &DataFrame) -> Array2<f32> {
    df.to_ndarray::<Float32Type>(IndexOrder::C).unwrap().reversed_axes()
}

// Trait to compute natural logarithm of each element in a 2D array.
pub trait Log {
    fn log(&self) -> Array2<f32>;
}

// Implementation of the Log trait for 2D ndarray Array of f32 type.
impl Log for Array2<f32> {
    fn log(&self) -> Array2<f32> {
        self.mapv(|x| x.log(std::f32::consts::E))
    }
}