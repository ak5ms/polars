use polars_ops::frame::join_many_sorted;
use polars_utils::pl_str::PlSmallStr;

use super::*;

pub struct JoinManyExec {
    inputs: Vec<Box<dyn Executor>>,
    on: Vec<PlSmallStr>,
}

impl JoinManyExec {
    pub fn new(inputs: Vec<Box<dyn Executor>>, on: Vec<PlSmallStr>) -> Self {
        Self { inputs, on }
    }
}

impl Executor for JoinManyExec {
    fn execute<'a>(&'a mut self, state: &'a mut ExecutionState) -> PolarsResult<DataFrame> {
        state.should_stop()?;
        #[cfg(debug_assertions)]
        if state.verbose() {
            eprintln!("run JoinManyExec")
        }

        let mut dfs = Vec::with_capacity(self.inputs.len());
        for input in &mut self.inputs {
            dfs.push(input.execute(state)?);
        }

        join_many_sorted(&dfs, &self.on)
    }
}
