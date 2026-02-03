use std::sync::Arc;

use polars_core::frame::DataFrame;
use polars_core::schema::Schema;
use polars_ops::frame::join_many_sorted;
use polars_utils::pl_str::PlSmallStr;

use crate::nodes::compute_node_prelude::*;
use crate::nodes::in_memory_sink::InMemorySinkNode;
use crate::nodes::in_memory_source::InMemorySourceNode;

enum JoinManyState {
    Sink { inputs: Vec<InMemorySinkNode> },
    Source(InMemorySourceNode),
    Done,
}

pub struct JoinManyNode {
    state: JoinManyState,
    on: Vec<PlSmallStr>,
}

impl JoinManyNode {
    pub fn new(schemas: Vec<Arc<Schema>>, on: Vec<PlSmallStr>) -> Self {
        let inputs = schemas
            .into_iter()
            .map(InMemorySinkNode::new)
            .collect();
        Self {
            state: JoinManyState::Sink { inputs },
            on,
        }
    }
}

impl ComputeNode for JoinManyNode {
    fn name(&self) -> &str {
        "join-many"
    }

    fn update_state(
        &mut self,
        recv: &mut [PortState],
        send: &mut [PortState],
        state: &StreamingExecutionState,
    ) -> PolarsResult<()> {
        assert_eq!(send.len(), 1);

        if send[0] == PortState::Done && !matches!(self.state, JoinManyState::Done) {
            self.state = JoinManyState::Done;
        }

        if let JoinManyState::Sink { inputs } = &mut self.state {
            if recv.iter().all(|state| *state == PortState::Done) {
                let mut dfs = Vec::with_capacity(inputs.len());
                for input in inputs.iter_mut() {
                    let df = input.get_output()?.unwrap_or_else(DataFrame::empty);
                    dfs.push(df);
                }
                let joined = join_many_sorted(&dfs, &self.on)?;
                let source =
                    InMemorySourceNode::new(Arc::new(joined), MorselSeq::default());
                self.state = JoinManyState::Source(source);
            }
        }

        match &mut self.state {
            JoinManyState::Sink { inputs } => {
                for (idx, input) in inputs.iter_mut().enumerate() {
                    input.update_state(&mut recv[idx..idx + 1], &mut [], state)?;
                }
                send[0] = PortState::Blocked;
            },
            JoinManyState::Source(source) => {
                recv.fill(PortState::Done);
                source.update_state(&mut [], send, state)?;
            },
            JoinManyState::Done => {
                recv.fill(PortState::Done);
                send[0] = PortState::Done;
            },
        }
        Ok(())
    }

    fn is_memory_intensive_pipeline_blocker(&self) -> bool {
        matches!(self.state, JoinManyState::Sink { .. })
    }

    fn spawn<'env, 's>(
        &'env mut self,
        scope: &'s TaskScope<'s, 'env>,
        recv_ports: &mut [Option<RecvPort<'_>>],
        send_ports: &mut [Option<SendPort<'_>>],
        state: &'s StreamingExecutionState,
        join_handles: &mut Vec<JoinHandle<PolarsResult<()>>>,
    ) {
        assert_eq!(send_ports.len(), 1);
        match &mut self.state {
            JoinManyState::Sink { inputs } => {
                for (idx, input) in inputs.iter_mut().enumerate() {
                    if recv_ports[idx].is_some() {
                        input.spawn(
                            scope,
                            &mut recv_ports[idx..idx + 1],
                            &mut [],
                            state,
                            join_handles,
                        );
                    }
                }
            },
            JoinManyState::Source(source) => {
                source.spawn(scope, &mut [], send_ports, state, join_handles)
            },
            JoinManyState::Done => unreachable!(),
        }
    }
}
