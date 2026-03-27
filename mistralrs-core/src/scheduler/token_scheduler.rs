//! Token-level continuous batching scheduler.
//!
//! This scheduler operates at iteration granularity:
//! - Each `schedule_iteration()` call returns work for ONE forward pass
//! - Prefill is chunked (256 tokens) to avoid blocking decode
//! - Fairness: users who have received fewer tokens get priority
//!
//! Design principles:
//! - Sequences stored in HashMap by ID (no &mut borrow issues)
//! - Queues are index sets, not containers
//! - State is explicit in SequenceState enum

use std::{
    collections::{HashMap, HashSet},
    sync::Arc,
    time::Instant,
};

use tokio::sync::Mutex;

use crate::{
    engine::IntervalLogger,
    paged_attention::KVCacheManager,
    sequence::{Sequence, SequenceState, StopReason},
};

use super::{Scheduler, SchedulerOutput};

// ============================================================================
// Constants
// ============================================================================

/// Max tokens per prefill chunk. Keeps prefill from blocking decode.
pub const PREFILL_CHUNK_SIZE: usize = 256;

/// Max sequences in decode batch.
pub const MAX_DECODE_BATCH: usize = 32;

/// Max total active sequences (prefilling + decoding).
pub const MAX_ACTIVE_SEQUENCES: usize = 64;

/// Max sequences in prefilling state simultaneously.
pub const MAX_PREFILLING: usize = 4;

/// Reset fairness counters after this many total tokens.
const FAIRNESS_DECAY_THRESHOLD: usize = 1000;

/// Decode-to-prefill ratio: run decode 3 out of 4 iterations when both have work.
const DECODE_PRIORITY_RATIO: usize = 4;

// ============================================================================
// Types
// ============================================================================

/// Unique sequence identifier.
pub type SequenceId = usize;

/// A chunk of prefill work for one sequence.
#[derive(Debug, Clone)]
pub struct PrefillChunk {
    pub seq_id: SequenceId,
    pub start_pos: usize,
    pub end_pos: usize,
}

/// Work batch for a single iteration.
#[derive(Debug, Default)]
pub struct IterationBatch {
    /// Prefill chunks to run (currently max 1, designed for future expansion).
    pub prefill: Vec<PrefillChunk>,
    /// Sequence IDs for decode (1 token each).
    pub decode: Vec<SequenceId>,
}

impl IterationBatch {
    pub fn is_empty(&self) -> bool {
        self.prefill.is_empty() && self.decode.is_empty()
    }
}

/// Per-user statistics for fairness tracking.
#[derive(Debug, Default, Clone)]
pub struct UserStats {
    pub tokens_generated: usize,
    pub active_sequences: usize,
    pub total_requests: usize,
}

/// Internal state for scheduler-managed sequences.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SchedulerSeqState {
    Waiting,
    Prefilling,
    Decoding,
    Finished,
}

/// Scheduler-side sequence metadata.
pub struct SchedulerSequence {
    pub sequence: Sequence,
    pub user_id: String,
    pub arrival_time: Instant,
    pub prefill_progress: usize,
    pub scheduler_state: SchedulerSeqState,
}

impl SchedulerSequence {
    pub fn new(sequence: Sequence, user_id: String) -> Self {
        Self {
            sequence,
            user_id,
            arrival_time: Instant::now(),
            prefill_progress: 0,
            scheduler_state: SchedulerSeqState::Waiting,
        }
    }

    /// Total prompt length (tokens to prefill).
    pub fn prompt_len(&self) -> usize {
        self.sequence.prompt_tokens()
    }

    /// How many prefill tokens remain.
    pub fn prefill_remaining(&self) -> usize {
        self.prompt_len().saturating_sub(self.prefill_progress)
    }

    /// Is prefill complete?
    pub fn prefill_done(&self) -> bool {
        self.prefill_progress >= self.prompt_len()
    }
}

// ============================================================================
// TokenScheduler
// ============================================================================

pub struct TokenScheduler {
    /// All sequences by ID.
    sequences: HashMap<SequenceId, SchedulerSequence>,

    /// Waiting queue (FIFO order preserved by arrival_time).
    waiting: HashSet<SequenceId>,

    /// Currently prefilling.
    prefilling: HashSet<SequenceId>,

    /// Currently decoding.
    decoding: HashSet<SequenceId>,

    /// Per-user token counters for fairness.
    user_tokens: HashMap<String, usize>,

    /// Per-user stats.
    user_stats: HashMap<String, UserStats>,

    /// Total tokens generated (for decay).
    total_tokens: usize,

    /// Iteration counter (for decode/prefill interleaving).
    iteration: usize,
}

impl TokenScheduler {
    pub fn new() -> Self {
        Self {
            sequences: HashMap::new(),
            waiting: HashSet::new(),
            prefilling: HashSet::new(),
            decoding: HashSet::new(),
            user_tokens: HashMap::new(),
            user_stats: HashMap::new(),
            total_tokens: 0,
            iteration: 0,
        }
    }

    /// Add a new sequence with user ID.
    pub fn add_sequence(&mut self, seq: Sequence, user_id: String) {
        let id = *seq.id();

        // Update user stats
        let stats = self.user_stats.entry(user_id.clone()).or_default();
        stats.total_requests += 1;
        stats.active_sequences += 1;

        // Ensure user has a token counter
        self.user_tokens.entry(user_id.clone()).or_insert(0);

        // Create scheduler sequence and add to waiting
        let sched_seq = SchedulerSequence::new(seq, user_id);
        self.sequences.insert(id, sched_seq);
        self.waiting.insert(id);
    }

    /// Schedule one iteration of work.
    pub fn schedule_iteration(&mut self) -> IterationBatch {
        self.iteration += 1;

        // 1. Remove finished sequences
        self.cleanup_finished();

        // 2. Admit waiting → prefilling (respecting caps)
        self.admit_waiting();

        // 3. Advance completed prefills → decoding
        self.advance_prefill_to_decode();

        // 4. Apply fairness decay if needed
        self.maybe_decay_fairness();

        // 5. Build iteration batch
        self.build_batch()
    }

    /// Mark sequence as complete.
    pub fn mark_complete(&mut self, seq_id: SequenceId, reason: StopReason) {
        if let Some(sched_seq) = self.sequences.get_mut(&seq_id) {
            sched_seq.scheduler_state = SchedulerSeqState::Finished;
            sched_seq.sequence.set_state(SequenceState::Done(reason));

            // Update user stats
            if let Some(stats) = self.user_stats.get_mut(&sched_seq.user_id) {
                stats.active_sequences = stats.active_sequences.saturating_sub(1);
            }
        }

        // Remove from active sets
        self.prefilling.remove(&seq_id);
        self.decoding.remove(&seq_id);
    }

    /// Record that a decode token was generated for a sequence.
    pub fn record_decode_token(&mut self, seq_id: SequenceId) {
        if let Some(sched_seq) = self.sequences.get(&seq_id) {
            let user_id = sched_seq.user_id.clone();
            *self.user_tokens.entry(user_id.clone()).or_insert(0) += 1;
            if let Some(stats) = self.user_stats.get_mut(&user_id) {
                stats.tokens_generated += 1;
            }
            self.total_tokens += 1;
        }
    }

    /// Record prefill progress for a sequence.
    pub fn record_prefill_progress(&mut self, seq_id: SequenceId, tokens_processed: usize) {
        if let Some(sched_seq) = self.sequences.get_mut(&seq_id) {
            sched_seq.prefill_progress += tokens_processed;
        }
    }

    /// Get mutable reference to underlying sequence.
    pub fn get_sequence_mut(&mut self, seq_id: SequenceId) -> Option<&mut Sequence> {
        self.sequences.get_mut(&seq_id).map(|s| &mut s.sequence)
    }

    /// Get reference to underlying sequence.
    pub fn get_sequence(&self, seq_id: SequenceId) -> Option<&Sequence> {
        self.sequences.get(&seq_id).map(|s| &s.sequence)
    }

    /// Get scheduler sequence (for prefill_progress etc).
    pub fn get_scheduler_sequence(&self, seq_id: SequenceId) -> Option<&SchedulerSequence> {
        self.sequences.get(&seq_id)
    }

    /// Get user stats.
    pub fn user_stats(&self) -> &HashMap<String, UserStats> {
        &self.user_stats
    }

    // ------------------------------------------------------------------------
    // Internal methods
    // ------------------------------------------------------------------------

    fn cleanup_finished(&mut self) {
        let finished: Vec<_> = self
            .sequences
            .iter()
            .filter(|(_, s)| {
                s.scheduler_state == SchedulerSeqState::Finished
                    || s.sequence.is_finished_paged_attn()
            })
            .map(|(id, _)| *id)
            .collect();

        for id in finished {
            self.waiting.remove(&id);
            self.prefilling.remove(&id);
            self.decoding.remove(&id);
            self.sequences.remove(&id);
        }
    }

    fn admit_waiting(&mut self) {
        let active_count = self.prefilling.len() + self.decoding.len();
        if active_count >= MAX_ACTIVE_SEQUENCES {
            return;
        }
        if self.prefilling.len() >= MAX_PREFILLING {
            return;
        }

        // Sort waiting by arrival time
        let mut waiting_sorted: Vec<_> = self.waiting.iter().copied().collect();
        waiting_sorted.sort_by_key(|id| {
            self.sequences
                .get(id)
                .map(|s| s.arrival_time)
                .unwrap_or_else(Instant::now)
        });

        let slots_available = (MAX_ACTIVE_SEQUENCES - active_count)
            .min(MAX_PREFILLING - self.prefilling.len());

        for id in waiting_sorted.into_iter().take(slots_available) {
            self.waiting.remove(&id);
            self.prefilling.insert(id);
            if let Some(sched_seq) = self.sequences.get_mut(&id) {
                sched_seq.scheduler_state = SchedulerSeqState::Prefilling;
                sched_seq.sequence.set_state(SequenceState::RunningPrefillPrompt);
            }
        }
    }

    fn advance_prefill_to_decode(&mut self) {
        let done: Vec<_> = self
            .prefilling
            .iter()
            .copied()
            .filter(|id| {
                self.sequences
                    .get(id)
                    .map(|s| s.prefill_done())
                    .unwrap_or(false)
            })
            .collect();

        for id in done {
            self.prefilling.remove(&id);
            self.decoding.insert(id);
            if let Some(sched_seq) = self.sequences.get_mut(&id) {
                sched_seq.scheduler_state = SchedulerSeqState::Decoding;
                sched_seq.sequence.set_state(SequenceState::RunningCompletion);
            }
        }
    }

    fn maybe_decay_fairness(&mut self) {
        if self.total_tokens > FAIRNESS_DECAY_THRESHOLD {
            for val in self.user_tokens.values_mut() {
                *val /= 2;
            }
            self.total_tokens = 0;
        }
    }

    fn build_batch(&mut self) -> IterationBatch {
        let mut batch = IterationBatch::default();

        let has_prefill = !self.prefilling.is_empty();
        let has_decode = !self.decoding.is_empty();

        // Decide whether to run prefill or decode this iteration
        // Decode gets priority 3/4 iterations when both have work
        let run_prefill = if has_prefill && has_decode {
            self.iteration % DECODE_PRIORITY_RATIO == 0
        } else {
            has_prefill
        };

        if run_prefill {
            // Pick one prefilling sequence (oldest by arrival)
            let prefill_id = self
                .prefilling
                .iter()
                .copied()
                .min_by_key(|id| {
                    self.sequences
                        .get(id)
                        .map(|s| s.arrival_time)
                        .unwrap_or_else(Instant::now)
                });

            if let Some(id) = prefill_id {
                if let Some(sched_seq) = self.sequences.get(&id) {
                    let start = sched_seq.prefill_progress;
                    let end = (start + PREFILL_CHUNK_SIZE).min(sched_seq.prompt_len());

                    batch.prefill.push(PrefillChunk {
                        seq_id: id,
                        start_pos: start,
                        end_pos: end,
                    });
                }
            }
        }

        if has_decode && !run_prefill {
            // Sort decoding sequences by fairness: (user_tokens, arrival_time)
            let mut decode_sorted: Vec<_> = self.decoding.iter().copied().collect();
            decode_sorted.sort_by_key(|id| {
                self.sequences.get(id).map(|s| {
                    let user_tokens = self.user_tokens.get(&s.user_id).copied().unwrap_or(0);
                    (user_tokens, s.arrival_time)
                })
            });

            batch.decode = decode_sorted.into_iter().take(MAX_DECODE_BATCH).collect();
        }

        batch
    }
}

impl Default for TokenScheduler {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// Scheduler trait implementation (for compatibility with existing engine)
// ============================================================================

impl Scheduler for TokenScheduler {
    fn schedule(&mut self, logger: &IntervalLogger) -> SchedulerOutput<'_> {
        // This is called by the existing engine loop.
        // We return an empty DefaultScheduler output and let the engine
        // call our schedule_iteration() method separately.
        //
        // NOTE: Full integration requires engine loop changes.
        // For now, this allows the TokenScheduler to coexist.

        logger.set_num_running(self.prefilling.len() + self.decoding.len());
        logger.set_num_waiting(self.waiting.len());

        // Return empty batch - actual work done via schedule_iteration()
        SchedulerOutput::DefaultScheduler {
            output: super::DefaultSchedulerOutput {
                completion: vec![].into(),
                prompt: vec![].into(),
            },
        }
    }

    fn waiting_len(&self) -> usize {
        self.waiting.len()
    }

    fn running_len(&self) -> usize {
        self.prefilling.len() + self.decoding.len()
    }

    fn add_seq(&mut self, seq: Sequence) {
        // Default user ID when not provided
        self.add_sequence(seq, "default".to_string());
    }

    fn free_finished_sequence_groups(&mut self) {
        self.cleanup_finished();
    }

    fn get_finished_recurrent_indices(&self) -> Vec<usize> {
        self.sequences
            .values()
            .filter(|s| {
                s.scheduler_state == SchedulerSeqState::Finished
                    || s.sequence.is_finished_paged_attn()
            })
            .filter_map(|s| s.sequence.recurrent_state_idx())
            .collect()
    }

    fn block_size(&self) -> Option<usize> {
        None
    }

    fn kv_cache_manager(&self) -> Option<Arc<Mutex<KVCacheManager>>> {
        None
    }

    fn set_prefix_caching_enabled(&mut self, _enabled: bool) {
        // Not using PagedAttention prefix caching
    }

    // =========================================================================
    // Continuous batching methods
    // =========================================================================

    fn is_continuous_batching(&self) -> bool {
        true
    }

    fn schedule_iteration(&mut self) -> Option<super::IterationBatch> {
        Some(self.schedule_iteration())
    }

    fn get_sequence(&self, id: super::SequenceId) -> Option<&Sequence> {
        self.get_sequence(id)
    }

    fn get_sequence_mut(&mut self, id: super::SequenceId) -> Option<&mut Sequence> {
        self.get_sequence_mut(id)
    }

    fn record_prefill_progress(&mut self, id: super::SequenceId, tokens: usize) {
        self.record_prefill_progress(id, tokens);
    }

    fn record_decode_token(&mut self, id: super::SequenceId) {
        self.record_decode_token(id);
    }

    fn add_seq_with_user(&mut self, seq: Sequence, user_id: String) {
        self.add_sequence(seq, user_id);
    }
}
