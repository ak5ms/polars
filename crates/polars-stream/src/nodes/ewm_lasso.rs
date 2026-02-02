use arrow::array::{Array, ListArray};
use arrow::bitmap::Bitmap;
use arrow::offset::{Offsets, OffsetsBuffer};
use polars_compute::ewm::{ewm_lasso_with_state, EwmLassoOptions, EwmLassoState};
use polars_core::prelude::{DataType, IntoColumn};
use polars_core::series::Series;
use polars_error::PolarsResult;
use polars_utils::pl_str::PlSmallStr;

use super::ComputeNode;
use crate::async_executor::{JoinHandle, TaskPriority, TaskScope};
use crate::execute::StreamingExecutionState;
use crate::graph::PortState;
use crate::pipe::{RecvPort, SendPort};

pub struct EwmLassoNode {
    name: &'static str,
    output_name: PlSmallStr,
    options: EwmLassoOptions,
    state: Option<EwmLassoState<f64>>,
}

impl EwmLassoNode {
    pub fn new(
        name: &'static str,
        output_name: PlSmallStr,
        options: EwmLassoOptions,
    ) -> Self {
        Self {
            name,
            output_name,
            options,
            state: None,
        }
    }
}

impl ComputeNode for EwmLassoNode {
    fn name(&self) -> &str {
        self.name
    }

    fn update_state(
        &mut self,
        recv: &mut [PortState],
        send: &mut [PortState],
        _state: &StreamingExecutionState,
    ) -> PolarsResult<()> {
        assert!(recv.len() == 1 && send.len() == 1);
        recv.swap_with_slice(send);
        Ok(())
    }

    fn spawn<'env, 's>(
        &'env mut self,
        scope: &'s TaskScope<'s, 'env>,
        recv_ports: &mut [Option<RecvPort<'_>>],
        send_ports: &mut [Option<SendPort<'_>>],
        _state: &'s StreamingExecutionState,
        join_handles: &mut Vec<JoinHandle<PolarsResult<()>>>,
    ) {
        assert_eq!(recv_ports.len(), 1);
        assert_eq!(send_ports.len(), 1);

        let mut recv = recv_ports[0].take().unwrap().serial();
        let mut send = send_ports[0].take().unwrap().serial();

        join_handles.push(scope.spawn_task(TaskPriority::High, async move {
            while let Ok(mut morsel) = recv.recv().await {
                let df = morsel.df_mut();

                debug_assert_eq!(df.width(), 2);

                let columns = df.columns();
                let x_series = columns[0].as_materialized_series().clone();
                let y_series = columns[1].as_materialized_series().clone();

                let x_series = match x_series.dtype() {
                    DataType::List(inner) if inner.as_ref() == &DataType::Float64 => x_series,
                    _ => x_series.cast(&DataType::List(Box::new(DataType::Float64)))?,
                };
                let y_series = match y_series.dtype() {
                    DataType::Float64 => y_series,
                    _ => y_series.cast(&DataType::Float64)?,
                };

                let x = x_series.list()?.rechunk();
                let y = y_series.f64()?.rechunk();

                let x_arr = x.downcast_iter().next().unwrap();
                let y_arr = y.downcast_iter().next().unwrap();

                if self.state.is_none() {
                    let offsets = x_arr.offsets();
                    let mut n_features = None;
                    for i in 0..x_arr.len() {
                        if x_arr.is_valid(i) && y_arr.is_valid(i) {
                            let len = (offsets[i + 1] - offsets[i]) as usize;
                            n_features = Some(len);
                            break;
                        }
                    }
                    if let Some(n_features) = n_features {
                        self.state = Some(EwmLassoState::new(n_features, self.options));
                    }
                }

                let result = if let Some(state) = self.state.as_mut() {
                    ewm_lasso_with_state(state, x_arr, y_arr)?
                } else {
                    let empty_values = arrow::array::PrimitiveArray::<f64>::from_vec(Vec::<f64>::new());
                    let offsets = OffsetsBuffer::from(Offsets::new_zeroed(x_arr.len()));
                    let validity = Bitmap::from_iter(std::iter::repeat(false).take(x_arr.len()));
                    ListArray::new(
                        x_arr.dtype().clone(),
                        offsets,
                        empty_values.boxed(),
                        Some(validity),
                    )
                };

                let dtype = DataType::List(Box::new(DataType::Float64));
                let mut out_series = unsafe {
                    Series::from_chunks_and_dtype_unchecked(
                        self.output_name.clone(),
                        vec![result.boxed()],
                        &dtype,
                    )
                };
                out_series.rename(self.output_name.clone());
                *df = polars_core::frame::DataFrame::new(
                    out_series.len(),
                    vec![out_series.into_column()],
                )?;

                if send.send(morsel).await.is_err() {
                    break;
                }
            }

            Ok(())
        }));
    }
}
