mod default_scheduler;
mod token_scheduler;

use std::sync::Arc;

pub use default_scheduler::{DefaultScheduler, DefaultSchedulerMethod, DefaultSchedulerOutput};
pub use token_scheduler::{
    IterationBatch, PrefillChunk, SchedulerSequence, SequenceId, TokenScheduler, UserStats,
    PREFILL_CHUNK_SIZE, MAX_DECODE_BATCH,
};
use tokio::sync::Mutex;

use crate::{
    engine::IntervalLogger,
    paged_attention::{
        CacheConfig, KVCacheManager, PagedAttentionScheduler, PagedAttentionSchedulerConfig,
        PagedAttentionSchedulerOutput,
    },
    sequence::Sequence,
};

#[derive(Clone)]
pub enum SchedulerConfig {
    DefaultScheduler {
        method: DefaultSchedulerMethod,
    },
    PagedAttentionMeta {
        max_num_seqs: usize,
        config: CacheConfig,
    },
    /// Token-level continuous batching scheduler with fairness.
    TokenScheduler,
}

impl SchedulerConfig {
    pub fn into_scheduler(self) -> Arc<Mutex<dyn Scheduler>> {
        match self {
            Self::DefaultScheduler { method } => {
                Arc::new(Mutex::new(DefaultScheduler::new(method)))
            }
            Self::PagedAttentionMeta {
                max_num_seqs,
                config,
            } => Arc::new(Mutex::new(PagedAttentionScheduler::new(
                PagedAttentionSchedulerConfig { max_num_seqs },
                config,
            ))),
            Self::TokenScheduler => {
                Arc::new(Mutex::new(TokenScheduler::new()))
            }
        }
    }
}

pub enum SchedulerOutput<'a> {
    DefaultScheduler {
        output: DefaultSchedulerOutput<'a>,
    },
    PagedAttention {
        output: PagedAttentionSchedulerOutput,
    },
}

pub trait Scheduler: Send + Sync {
    fn schedule(&mut self, logger: &IntervalLogger) -> SchedulerOutput<'_>;
    fn waiting_len(&self) -> usize;
    fn running_len(&self) -> usize;
    fn add_seq(&mut self, seq: Sequence);
    /// This may do nothing. It depends on the implementation
    fn free_finished_sequence_groups(&mut self);
    /// Get recurrent state pool indices of finished sequences for freeing.
    /// Called before free_finished_sequence_groups to allow cleanup of hybrid cache slots.
    fn get_finished_recurrent_indices(&self) -> Vec<usize>;

    // PagedAttention metadata
    fn block_size(&self) -> Option<usize>;
    fn kv_cache_manager(&self) -> Option<Arc<Mutex<KVCacheManager>>>;

    /// Set whether prefix caching is enabled. Called by Engine after creation
    /// to synchronize with the global no_prefix_cache setting.
    fn set_prefix_caching_enabled(&mut self, enabled: bool);

    // =========================================================================
    // Continuous batching methods (TokenScheduler)
    // Default implementations return None/false for non-TokenScheduler.
    // =========================================================================

    /// Returns true if this scheduler uses continuous batching (iteration-level).
    fn is_continuous_batching(&self) -> bool {
        false
    }

    /// Schedule one iteration of work. Returns (prefill_chunks, decode_ids).
    /// Only meaningful for continuous batching schedulers.
    fn schedule_iteration(&mut self) -> Option<IterationBatch> {
        None
    }

    /// Get mutable reference to a sequence by ID.
    fn get_sequence_mut(&mut self, _id: SequenceId) -> Option<&mut Sequence> {
        None
    }

    /// Record that prefill progress was made on a sequence.
    fn record_prefill_progress(&mut self, _id: SequenceId, _tokens: usize) {}

    /// Record that a decode token was generated for a sequence.
    fn record_decode_token(&mut self, _id: SequenceId) {}

    /// Add a sequence with user ID (for fairness tracking).
    fn add_seq_with_user(&mut self, seq: Sequence, _user_id: String) {
        // Default: ignore user_id, just add normally
        self.add_seq(seq);
    }
}
