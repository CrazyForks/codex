use std::sync::Arc;

use super::SessionTask;
use super::SessionTaskContext;
use crate::codex::TurnContext;
use crate::protocol::EventMsg;
use crate::state::TaskKind;
use async_trait::async_trait;
use codex_protocol::user_input::UserInput;
use tokio_util::sync::CancellationToken;

#[derive(Clone, Copy, Default)]
pub(crate) struct CompactTask;

#[async_trait]
impl SessionTask for CompactTask {
    fn kind(&self) -> TaskKind {
        TaskKind::Compact
    }

    async fn run(
        self: Arc<Self>,
        session: Arc<SessionTaskContext>,
        ctx: Arc<TurnContext>,
        input: Vec<UserInput>,
        _cancellation_token: CancellationToken,
    ) -> Option<String> {
        let session = session.clone_session();
        if crate::compact::should_use_remote_compact_task(&ctx.provider) {
            let _ = session.services.otel_manager.counter(
                "codex.task.compact",
                1,
                &[("type", "remote")],
            );
            if let Err(err) =
                crate::compact_remote::run_remote_compact_task(session.clone(), ctx.clone()).await
            {
                let event = EventMsg::Error(
                    err.to_error_event(Some("Error running remote compact task".to_string())),
                );
                session.send_event(&ctx, event).await;
            }
        } else {
            let _ = session.services.otel_manager.counter(
                "codex.task.compact",
                1,
                &[("type", "local")],
            );
            if let Err(err) =
                crate::compact::run_compact_task(session.clone(), ctx.clone(), input).await
            {
                let event = EventMsg::Error(
                    err.to_error_event(Some("Error running local compact task".to_string())),
                );
                session.send_event(&ctx, event).await;
            }
        }

        None
    }
}
