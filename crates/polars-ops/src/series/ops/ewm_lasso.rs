pub use polars_compute::ewm::EwmLassoOptions;
use polars_compute::ewm::ewm_lasso as kernel_ewm_lasso;
use polars_core::prelude::*;
use polars_error::PolarsResult;

pub fn ewm_lasso(x: &Series, y: &Series, options: EwmLassoOptions) -> PolarsResult<Series> {
    let x = x.cast(&DataType::List(Box::new(DataType::Float64)))?;
    let y = y.cast(&DataType::Float64)?;

    let x = x.list()?;
    let y = y.f64()?;
    let x = x.rechunk();
    let y = y.rechunk();

    let x_arr = x.downcast_iter().next().unwrap();
    let y_arr = y.downcast_iter().next().unwrap();

    let result = kernel_ewm_lasso(x_arr, y_arr, options)?;
    let dtype = DataType::List(Box::new(DataType::Float64));
    let name = x.name().clone();
    unsafe {
        Ok(Series::from_chunks_and_dtype_unchecked(
            name,
            vec![result.boxed()],
            &dtype,
        ))
    }
}
