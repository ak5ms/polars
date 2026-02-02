use arrow::array::{Array, ListArray, PrimitiveArray};
use arrow::offset::{Offsets, OffsetsBuffer};
use arrow::types::NativeType;
use polars_error::{polars_bail, polars_err, PolarsResult};

use super::options::EwmLassoOptions;

#[derive(Debug, Clone)]
pub struct EwmLassoState<T> {
    n_features: usize,
    decay: T,
    alpha: T,
    max_iter: usize,
    tol: T,
    s: Vec<T>,
    z: Vec<T>,
    coef: Vec<T>,
    m: Vec<T>,
    wsum: T,
}

impl<T> EwmLassoState<T>
where
    T: NativeType
        + num_traits::Float
        + std::ops::AddAssign
        + std::ops::DivAssign
        + std::ops::MulAssign,
{
    pub fn new(n_features: usize, options: EwmLassoOptions) -> Self {
        Self {
            n_features,
            decay: T::from(options.decay).unwrap(),
            alpha: T::from(options.alpha).unwrap(),
            max_iter: options.max_iter,
            tol: T::from(options.tol).unwrap(),
            s: vec![T::zero(); n_features * n_features],
            z: vec![T::zero(); n_features],
            coef: vec![T::zero(); n_features],
            m: vec![T::zero(); n_features],
            wsum: T::zero(),
        }
    }

    pub fn coefficients(&self) -> &[T] {
        &self.coef
    }

    pub fn update_row(&mut self, x: &[T], y: T) {
        self.wsum = self.decay * self.wsum + T::one();
        for j in 0..self.n_features {
            self.m[j] = self.decay * self.m[j] + x[j];
        }

        for i in 0..self.n_features {
            let row_offset = i * self.n_features;
            let x_i = x[i];
            for j in 0..self.n_features {
                self.s[row_offset + j] = self.decay * self.s[row_offset + j] + x_i * x[j];
            }
        }

        for i in 0..self.n_features {
            self.z[i] = self.decay * self.z[i] + x[i] * y;
        }

        for _ in 0..self.max_iter {
            let mut max_delta = T::zero();

            for j in 0..self.n_features {
                let row_offset = j * self.n_features;
                let mut rho = self.z[j];
                for k in 0..self.n_features {
                    if k != j {
                        rho = rho - self.s[row_offset + k] * self.coef[k];
                    }
                }

                let beta_old = self.coef[j];
                let var_j = (self.s[row_offset + j] / self.wsum)
                    - (self.m[j] / self.wsum) * (self.m[j] / self.wsum);
                let sigma_j = if var_j > T::zero() {
                    var_j.sqrt()
                } else {
                    T::zero()
                };
                let thr = self.alpha * sigma_j * self.wsum;

                if rho > thr {
                    self.coef[j] = (rho - thr) / self.s[row_offset + j];
                } else if rho < -thr {
                    self.coef[j] = (rho + thr) / self.s[row_offset + j];
                } else {
                    self.coef[j] = T::zero();
                }

                let delta = (self.coef[j] - beta_old).abs();
                if delta > max_delta {
                    max_delta = delta;
                }
            }

            if max_delta < self.tol {
                break;
            }
        }
    }
}

pub fn ewm_lasso(
    x: &ListArray<i64>,
    y: &PrimitiveArray<f64>,
    options: EwmLassoOptions,
) -> PolarsResult<ListArray<i64>> {
    if x.len() != y.len() {
        polars_bail!(
            ComputeError: "x and y must have the same length (got {} and {})",
            x.len(),
            y.len()
        );
    }

    let values = x
        .values()
        .as_any()
        .downcast_ref::<PrimitiveArray<f64>>()
        .ok_or_else(|| polars_err!(ComputeError: "expected list values to be Float64"))?;

    if values.null_count() > 0 {
        polars_bail!(ComputeError: "list values must not contain nulls");
    }

    let offsets = x.offsets();
    let mut n_features = None;
    for i in 0..x.len() {
        if x.is_valid(i) {
            let len = (offsets[i + 1] - offsets[i]) as usize;
            n_features = Some(len);
            break;
        }
    }

    let Some(n_features) = n_features else {
        let empty_values = PrimitiveArray::<f64>::from_vec(Vec::<f64>::new());
        return Ok(ListArray::new(
            x.dtype().clone(),
            OffsetsBuffer::from(Offsets::new_zeroed(x.len())),
            empty_values.boxed(),
            Some(x.validity().cloned().unwrap_or_default()),
        ));
    };

    let mut state = EwmLassoState::<f64>::new(n_features, options);
    ewm_lasso_with_state(&mut state, x, y)
}

pub fn ewm_lasso_with_state(
    state: &mut EwmLassoState<f64>,
    x: &ListArray<i64>,
    y: &PrimitiveArray<f64>,
) -> PolarsResult<ListArray<i64>> {
    if x.len() != y.len() {
        polars_bail!(
            ComputeError: "x and y must have the same length (got {} and {})",
            x.len(),
            y.len()
        );
    }

    let n_features = state.n_features;
    let offsets = x.offsets();
    let mut offsets_out = Vec::with_capacity(x.len() + 1);
    offsets_out.push(0i64);
    let mut values_out = Vec::with_capacity(x.len() * n_features);
    let mut validity = Vec::with_capacity(x.len());
    let values = x
        .values()
        .as_any()
        .downcast_ref::<PrimitiveArray<f64>>()
        .ok_or_else(|| polars_err!(ComputeError: "expected list values to be Float64"))?;

    if values.null_count() > 0 {
        polars_bail!(ComputeError: "list values must not contain nulls");
    }

    let values_slice = values.values();

    for i in 0..x.len() {
        let valid = x.is_valid(i) && y.is_valid(i);
        validity.push(valid);

        if !valid {
            offsets_out.push(offsets_out.last().copied().unwrap());
            continue;
        }

        let start = offsets[i] as usize;
        let end = offsets[i + 1] as usize;
        if end - start != n_features {
            polars_bail!(
                ComputeError: "list entries must all have the same length (expected {}, got {})",
                n_features,
                end - start
            );
        }

        state.update_row(&values_slice[start..end], y.value(i));
        values_out.extend_from_slice(state.coefficients());
        offsets_out.push(offsets_out.last().copied().unwrap() + n_features as i64);
    }

    let validity = if validity.iter().all(|v| *v) {
        None
    } else {
        Some(validity.into())
    };

    Ok(ListArray::new(
        x.dtype().clone(),
        OffsetsBuffer::try_from(offsets_out)?,
        PrimitiveArray::<f64>::from_vec(values_out).boxed(),
        validity,
    ))
}
