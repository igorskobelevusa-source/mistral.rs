//! Experimental low-level decode session API.
//!
//! This module defines the core engine-facing surface needed to support
//! external token-level schedulers without forcing scheduling policy into the
//! engine thread itself.

use crate::{
    paged_attention::block_hash::compute_block_hashes,
    paged_attention::KVCacheManager, prefix_cacher::MatchingCache,
    prefix_cacher::PrefixCacheManagerV2, response::Response, sampler::Sampler,
    sequence::SeqStepType, sequence::Sequence, sequence::SequenceGroup,
    sequence::SequenceRecognizer, sequence::SequenceState, EngineConfig, MistralRsError,
    MistralRsConfig, Pipeline, SamplingParams, SchedulerConfig, StopTokens,
};
use crate::pipeline::{
    text_models_inputs_processor::PagedAttentionMeta, CacheBackendMetadata, CacheInstruction,
};
use candle_core::Tensor;
use futures::Future;
use rand::SeedableRng;
use rand_isaac::Isaac64Rng;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use tokio::sync::Mutex;

const STATEFUL_SEED: u64 = 0;

/// Capability flags for lower-level execution APIs.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct BackendFeatures {
    /// True when the backend can create opaque per-sequence decode sessions.
    pub supports_stateful_decode: bool,
    /// True when the backend can batch externally managed sessions together.
    pub supports_external_scheduler: bool,
}

/// Opaque handle for backend-owned per-sequence decode state.
#[allow(dead_code)]
#[derive(Default)]
pub struct DecodeSession {
    pub(crate) config: DecodeSessionConfig,
    pub(crate) sequence: Option<Sequence>,
    pub(crate) response_rx: Option<tokio::sync::mpsc::Receiver<Response>>,
    runtime: Option<Arc<StatefulRuntime>>,
    block_hashes: Vec<crate::paged_attention::block_hash::BlockHash>,
}

/// Result of a prompt prefill pass.
#[derive(Debug, Clone, Default)]
pub struct PrefillOutput {
    /// Number of prompt tokens consumed by the backend.
    pub prompt_tokens: usize,
    /// Number of prompt tokens satisfied from prefix/paged cache.
    pub cached_prompt_tokens: usize,
    /// True when the backend only staged prompt/cache state and did not execute
    /// a forward pass yet.
    pub staged_only: bool,
}

/// Result of advancing a single sequence by one decode step.
#[derive(Debug, Clone, Default)]
pub struct DecodeStepOutput {
    /// Token produced by the backend during this step, if any.
    pub token: Option<u32>,
    /// Decoded text delta made visible by this step, if any.
    pub text_delta: Option<String>,
    /// Whether the sequence has reached a terminal state.
    pub is_done: bool,
}

/// Result of advancing multiple externally managed sessions together.
#[derive(Debug, Clone, Default)]
pub struct BatchDecodeOutput;

/// Configuration for a stateful decode session.
#[derive(Debug, Clone)]
pub struct DecodeSessionConfig {
    /// Sampling parameters used for generation.
    pub sampling_params: SamplingParams,
    /// Whether to return logprobs during sampling.
    pub return_logprobs: bool,
}

impl Default for DecodeSessionConfig {
    fn default() -> Self {
        Self {
            sampling_params: SamplingParams::deterministic(),
            return_logprobs: false,
        }
    }
}

/// A model loaded for future external-scheduler execution.
///
/// Unlike [`crate::MistralRs`], this type does not immediately wrap the model in
/// an engine thread and request queue. It preserves direct access to the
/// underlying pipeline so a future stateful stepping implementation can own the
/// execution policy explicitly.
#[derive(Clone)]
#[allow(dead_code)]
pub struct StatefulModel {
    scheduler_config: SchedulerConfig,
    engine_config: EngineConfig,
    model_id: String,
    config: MistralRsConfig,
    next_session_id: Arc<AtomicUsize>,
    runtime: Arc<StatefulRuntime>,
}

#[allow(dead_code)]
struct StatefulRuntime {
    pipeline: Arc<Mutex<dyn Pipeline>>,
    prefix_cacher: Arc<Mutex<PrefixCacheManagerV2>>,
    rng: Arc<std::sync::Mutex<Isaac64Rng>>,
    paged_kv_cache_manager: Option<Arc<tokio::sync::Mutex<KVCacheManager>>>,
    has_paged_attention: bool,
}

impl StatefulModel {
    pub async fn new(
        pipeline: Arc<Mutex<dyn Pipeline>>,
        scheduler_config: SchedulerConfig,
        engine_config: EngineConfig,
        model_id: String,
        config: MistralRsConfig,
    ) -> Self {
        let (has_paged_attention, cache_config, disable_prefix_cache) = {
            let pipeline_guard = pipeline.lock().await;
            (
                pipeline_guard.get_metadata().cache_config.is_some(),
                pipeline_guard.get_metadata().cache_config.clone(),
                engine_config.no_prefix_cache
                    || engine_config.no_kv_cache
                    || config.max_seq_len.is_none(),
            )
        };
        let prefix_cacher = Arc::new(Mutex::new(PrefixCacheManagerV2::new(
            engine_config.prefix_cache_n,
            disable_prefix_cache,
            has_paged_attention,
        )));
        let paged_kv_cache_manager = cache_config.map(|cache_config| {
            Arc::new(tokio::sync::Mutex::new(KVCacheManager::new(
                cache_config.num_gpu_blocks,
                cache_config.block_size,
                !engine_config.no_prefix_cache,
                vec![0],
            )))
        });
        let runtime = Arc::new(StatefulRuntime {
            pipeline,
            prefix_cacher,
            rng: Arc::new(std::sync::Mutex::new(Isaac64Rng::seed_from_u64(STATEFUL_SEED))),
            paged_kv_cache_manager,
            has_paged_attention,
        });
        Self {
            scheduler_config,
            engine_config,
            model_id,
            config,
            next_session_id: Arc::new(AtomicUsize::new(1)),
            runtime,
        }
    }

    pub fn model_id(&self) -> &str {
        &self.model_id
    }

    pub fn config(&self) -> &MistralRsConfig {
        &self.config
    }

    pub fn scheduler_config(&self) -> &SchedulerConfig {
        &self.scheduler_config
    }

    pub fn engine_config(&self) -> &EngineConfig {
        &self.engine_config
    }

    pub fn pipeline(&self) -> Arc<Mutex<dyn Pipeline>> {
        self.runtime.pipeline.clone()
    }

    pub fn backend_features(&self) -> BackendFeatures {
        BackendFeatures {
            supports_stateful_decode: true,
            supports_external_scheduler: false,
        }
    }

    pub fn new_decode_session(
        &self,
        config: DecodeSessionConfig,
    ) -> Result<DecodeSession, MistralRsError> {
        let pipeline = self.runtime.pipeline.try_lock().map_err(|_| {
            MistralRsError::Unsupported(
                "stateful runtime pipeline is currently busy; try again".into(),
            )
        })?;
        match self.config.category {
            crate::ModelCategory::Text | crate::ModelCategory::Vision { .. } => (),
            _ => {
                return Err(MistralRsError::Unsupported(
                    "stateful decode sessions are currently only prepared for text and vision models".into(),
                ))
            }
        }

        let top_k = config.sampling_params.top_k.map(|x| x as i64).unwrap_or(-1);
        let top_p = config.sampling_params.top_p.unwrap_or(1.0);
        let min_p = config.sampling_params.min_p.unwrap_or(0.0);
        let tokenizer = pipeline.tokenizer();
        let _sampler = Sampler::new(
            config.sampling_params.temperature,
            config.sampling_params.top_n_logprobs,
            tokenizer.clone(),
            config.sampling_params.frequency_penalty,
            config.sampling_params.presence_penalty,
            config.sampling_params.repetition_penalty,
            config.sampling_params.dry_params.clone(),
            top_k,
            top_p,
            min_p,
            vec![],
        )
        .map_err(|e| MistralRsError::Unsupported(format!("failed to prepare sampler: {e}")))?;

        let (stop_tokens, stop_strings) = split_stop_tokens(
            &config.sampling_params,
            pipeline.tokenizer(),
            pipeline.get_metadata().tok_env(),
        )?;

        let dummy_group = Arc::new(tokio::sync::Mutex::new(SequenceGroup::new(
            config.sampling_params.n_choices.max(1),
            false,
            true,
            None,
        )));
        let session_id = self.next_session_id.fetch_add(1, Ordering::Relaxed);
        let num_hidden_layers = pipeline.get_metadata().num_hidden_layers;
        let eos_tokens = pipeline.get_metadata().eos_tok.clone();
        let is_xlora = pipeline.get_metadata().is_xlora;
        let block_size = pipeline.get_metadata().cache_config.clone().map(|c| c.block_size);
        let preallocated_tokens = config
            .sampling_params
            .max_len
            .or(self.config.max_seq_len)
            .unwrap_or(crate::kv_cache::NormalCache::CACHE_GROW_SIZE)
            .max(crate::kv_cache::NormalCache::CACHE_GROW_SIZE);
        let preallocated_cache =
            build_preallocated_cache(&*pipeline, num_hidden_layers, preallocated_tokens)?;
        drop(pipeline);

        let (tx, rx) = tokio::sync::mpsc::channel::<Response>(1);
        let mut sequence = Sequence::new_waiting(
            Vec::new(),
            String::new(),
            session_id,
            0,
            num_hidden_layers,
            tx,
            _sampler,
            stop_tokens,
            stop_strings,
            config.sampling_params.max_len,
            config.return_logprobs,
            is_xlora,
            dummy_group,
            0,
            0,
            SequenceRecognizer::None,
            None,
            None,
            None,
            None,
            block_size,
            None,
            None,
            SeqStepType::PromptAndDecode,
            None,
            None,
            preallocated_cache,
            false,
            eos_tokens,
        );

        {
            let pipeline = self.runtime.pipeline.try_lock().map_err(|_| {
                MistralRsError::Unsupported(
                    "stateful runtime pipeline is currently busy; try again".into(),
                )
            })?;
            if let Some(chat_template) = pipeline.get_chat_template() {
                if chat_template.is_harmony_format() {
                    if !crate::harmony::is_harmony_encoding_ready() {
                        crate::harmony::prewarm_harmony_encoding();
                    }
                    let _ = sequence.enable_harmony_mode();
                } else if chat_template.uses_think_tags() {
                    sequence.enable_think_tag_mode();
                }
            }

            if !pipeline.get_metadata().no_kv_cache && pipeline.cache().is_hybrid() {
                let mut hybrid_cache = pipeline.cache().hybrid();
                if let Some(slot_idx) = hybrid_cache.allocate_seq() {
                    sequence.set_recurrent_state_idx(Some(slot_idx));
                }
            }
        }

        Ok(DecodeSession {
            config,
            sequence: Some(sequence),
            response_rx: Some(rx),
            runtime: Some(self.runtime.clone()),
            block_hashes: Vec::new(),
        })
    }

    pub fn prefill(
        &self,
        session: &mut DecodeSession,
        input_ids: &[u32],
    ) -> Result<PrefillOutput, MistralRsError> {
        let runtime = session.runtime.as_ref().ok_or_else(|| {
            MistralRsError::Unsupported("decode session is missing runtime state".into())
        })?;
        let mut sequence = session.sequence.take().ok_or_else(|| {
            MistralRsError::Unsupported("decode session is missing sequence state".into())
        })?;

        sequence.set_toks_and_reallocate(input_ids.to_vec(), None);

        let cached_prompt_tokens = if runtime.has_paged_attention {
            let mut kv_mgr = runtime
                .paged_kv_cache_manager
                .as_ref()
                .ok_or_else(|| {
                    MistralRsError::Unsupported(
                        "paged attention is enabled but no KV cache manager is present".into(),
                    )
                })?
                .try_lock()
                .map_err(|_| {
                    MistralRsError::Unsupported(
                        "stateful paged KV manager is currently busy; try again".into(),
                    )
                })?;

            session.block_hashes =
                compute_block_hashes(sequence.get_toks(), kv_mgr.block_size(), sequence.mm_features(), &[]);
            let computed = if self.engine_config.no_prefix_cache {
                crate::paged_attention::kv_cache_manager::ComputedBlocks {
                    block_ids: Vec::new(),
                    num_computed_tokens: 0,
                }
            } else {
                kv_mgr.get_computed_blocks(&session.block_hashes, sequence.get_toks().len())
            };
            kv_mgr
                .allocate_slots(*sequence.id(), sequence.get_toks().len(), &computed.block_ids)
                .ok_or_else(|| {
                    MistralRsError::Unsupported(
                        "failed to allocate paged KV slots for stateful prefill".into(),
                    )
                })?;
            sequence.set_prefix_cache_len(computed.num_computed_tokens);
            computed.num_computed_tokens
        } else {
            let prefill_cache = runtime
                .prefix_cacher
                .try_lock()
                .map_err(|_| {
                    MistralRsError::Unsupported(
                        "stateful prefix cacher is currently busy; try again".into(),
                    )
                })?
                .search_for_matching_cache(
                    sequence.get_toks(),
                    sequence.image_hashes(),
                    sequence.audio_hashes(),
                )
                .map_err(|e| MistralRsError::Unsupported(format!("prefix cache lookup failed: {e}")))?;

            match prefill_cache {
                Some(MatchingCache::Normal {
                    normal,
                    recurrent_snapshots,
                    images_to_keep,
                    audios_to_keep,
                    toks,
                    offset,
                }) => {
                    if let Some(snapshots) = recurrent_snapshots {
                        if let Some(slot_idx) = sequence.recurrent_state_idx() {
                            let pipeline = runtime.pipeline.try_lock().map_err(|_| {
                                MistralRsError::Unsupported(
                                    "stateful runtime pipeline is currently busy; try again".into(),
                                )
                            })?;
                            if pipeline.cache().is_hybrid() {
                                let mut hybrid_cache = pipeline.cache().hybrid();
                                let _ = hybrid_cache.restore_recurrent_state(slot_idx, &snapshots);
                            }
                        }
                    }
                    sequence.keep_num_images(images_to_keep);
                    sequence.keep_num_audios(audios_to_keep);
                    let prompt_tokens = toks.len();
                    sequence = sequence.prefill_v2_normal(normal, toks, offset);
                    prompt_tokens
                }
                None => 0,
            }
        };

        let prompt_tokens = sequence.get_toks().len();
        session.sequence = Some(sequence);
        Ok(PrefillOutput {
            prompt_tokens,
            cached_prompt_tokens,
            staged_only: true,
        })
    }

    pub fn decode_step(
        &self,
        session: &mut DecodeSession,
        token_id: u32,
    ) -> Result<DecodeStepOutput, MistralRsError> {
        let runtime = session.runtime.as_ref().ok_or_else(|| {
            MistralRsError::Unsupported("decode session is missing runtime state".into())
        })?;
        let mut sequence = session.sequence.take().ok_or_else(|| {
            MistralRsError::Unsupported("decode session is missing sequence state".into())
        })?;

        let is_prompt = sequence.is_prompt() || sequence.is_waiting();
        if !is_prompt {
            let expected = sequence.get_toks().last().copied().ok_or_else(|| {
                MistralRsError::Unsupported(
                    "decode session has no prior token to continue from".into(),
                )
            })?;
            if token_id != expected {
                session.sequence = Some(sequence);
                return Err(MistralRsError::Unsupported(format!(
                    "stateful decode_step currently uses backend-managed sampling; expected token {expected}, got {token_id}"
                )));
            }
        }

        if is_prompt {
            sequence.set_state(SequenceState::RunningPrompt);
            sequence.set_step_start_instant();
        } else {
            sequence.set_state(SequenceState::RunningCompletion);
            if runtime.has_paged_attention {
                let mut kv_mgr = runtime
                    .paged_kv_cache_manager
                    .as_ref()
                    .ok_or_else(|| {
                        MistralRsError::Unsupported(
                            "paged attention is enabled but no KV cache manager is present".into(),
                        )
                    })?
                    .try_lock()
                    .map_err(|_| {
                        MistralRsError::Unsupported(
                            "stateful paged KV manager is currently busy; try again".into(),
                        )
                    })?;
                kv_mgr
                    .allocate_slots(*sequence.id(), sequence.len() + 1, &[])
                    .ok_or_else(|| {
                        MistralRsError::Unsupported(
                            "failed to allocate paged KV slots for stateful decode".into(),
                        )
                    })?;
            }
        }

        let step_duration = match run_stateful_future(async {
            let mut pipeline = runtime.pipeline.lock().await;
            let mut prefix_cacher = runtime.prefix_cacher.lock().await;
            let return_raw_logits = sequence.return_raw_logits;

            if runtime.has_paged_attention {
                let block_size = {
                    let kv_mgr = runtime
                        .paged_kv_cache_manager
                        .as_ref()
                        .expect("paged KV manager exists when paged attention is enabled")
                        .lock()
                        .await;
                    kv_mgr.block_size()
                };
                let metadata = PagedAttentionMeta {
                    block_size,
                    sliding_window: pipeline.get_metadata().sliding_window,
                    kv_cache_manager: runtime
                        .paged_kv_cache_manager
                        .as_ref()
                        .expect("paged KV manager exists when paged attention is enabled")
                        .clone(),
                };
                let mut seqs = vec![&mut sequence];
                pipeline
                    .step(
                        &mut seqs,
                        is_prompt,
                        return_raw_logits,
                        &mut prefix_cacher,
                        self.engine_config.disable_eos_stop,
                        runtime.rng.clone(),
                        CacheBackendMetadata::PagedAttention { metadata },
                    )
                    .await
            } else {
                let pre_op = if is_prompt {
                    if sequence.token_offset() != 0 {
                        CacheInstruction::In
                    } else {
                        CacheInstruction::Reset {
                            load_preallocated_cache: true,
                            reset_non_granular: false,
                        }
                    }
                } else if !self.engine_config.no_kv_cache {
                    CacheInstruction::In
                } else {
                    CacheInstruction::Nothing
                };
                let post_op = if !self.engine_config.no_kv_cache {
                    CacheInstruction::Out
                } else {
                    CacheInstruction::Reset {
                        load_preallocated_cache: false,
                        reset_non_granular: false,
                    }
                };
                let mut seqs = vec![&mut sequence];
                pipeline
                    .step(
                        &mut seqs,
                        is_prompt,
                        return_raw_logits,
                        &mut prefix_cacher,
                        self.engine_config.disable_eos_stop,
                        runtime.rng.clone(),
                        CacheBackendMetadata::DefaultInstructions { pre_op, post_op },
                    )
                    .await
            }
        }) {
            Ok(duration) => duration,
            Err(e) => {
                session.sequence = Some(sequence);
                return Err(MistralRsError::Unsupported(format!(
                    "stateful decode step failed: {e}"
                )));
            }
        };

        let output = DecodeStepOutput {
            token: sequence.logprobs().last().map(|logprob| logprob.token),
            text_delta: sequence.peek_delta().ok().flatten(),
            is_done: matches!(sequence.getstate(), SequenceState::Done(_)),
        };

        if is_prompt && !output.is_done {
            match sequence.sequence_stepping_type() {
                SeqStepType::OneShot => (),
                SeqStepType::PromptAndDecode => sequence.set_state(SequenceState::RunningCompletion),
            }
            #[allow(clippy::cast_precision_loss)]
            let prompt_tok_per_sec = sequence.len() as f32 / step_duration.as_secs_f32();
            sequence.prompt_tok_per_sec = prompt_tok_per_sec;
            sequence.prompt_timestamp = Some(
                std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .expect("Time travel has occurred!")
                    .as_millis(),
            );
            sequence.total_prompt_time = Some(step_duration.as_millis());
            sequence.step_start_instant = None;
        }

        session.sequence = Some(sequence);
        Ok(output)
    }

    pub fn decode_batch(
        &self,
        _sessions: &mut [&mut DecodeSession],
        _token_ids: &[u32],
    ) -> Result<BatchDecodeOutput, MistralRsError> {
        Err(MistralRsError::Unsupported(
            "external batched decode is not exposed by this backend yet".into(),
        ))
    }
}

fn run_stateful_future<F, T>(future: F) -> Result<T, MistralRsError>
where
    F: Future<Output = Result<T, candle_core::Error>>,
{
    match tokio::runtime::Handle::try_current() {
        Ok(handle) => {
            if handle.runtime_flavor() == tokio::runtime::RuntimeFlavor::CurrentThread {
                Err(MistralRsError::Unsupported(
                    "stateful blocking step cannot run inside a single-threaded Tokio runtime"
                        .into(),
                ))
            } else {
                tokio::task::block_in_place(|| {
                    handle
                        .block_on(future)
                        .map_err(|e| MistralRsError::Unsupported(e.to_string()))
                })
            }
        }
        Err(_) => {
            let rt = tokio::runtime::Runtime::new().map_err(|e| {
                MistralRsError::Unsupported(format!(
                    "failed to create runtime for stateful decode: {e}"
                ))
            })?;
            rt.block_on(future)
                .map_err(|e| MistralRsError::Unsupported(e.to_string()))
        }
    }
}

impl Drop for DecodeSession {
    fn drop(&mut self) {
        let Some(runtime) = &self.runtime else {
            return;
        };
        let Some(sequence) = &self.sequence else {
            return;
        };

        let seq_id = *sequence.id();
        if let Some(kv_mgr) = &runtime.paged_kv_cache_manager {
            if let Ok(mut kv_mgr) = kv_mgr.try_lock() {
                kv_mgr.free(seq_id);
            }
        }

        if let Some(slot_idx) = sequence.recurrent_state_idx() {
            if let Ok(pipeline) = runtime.pipeline.try_lock() {
                if pipeline.cache().is_hybrid() {
                    pipeline.cache().hybrid().free_seq(slot_idx);
                }
            }
        }
    }
}

fn split_stop_tokens(
    sampling_params: &SamplingParams,
    tokenizer: Option<Arc<tokenizers::Tokenizer>>,
    tok_env: Option<llguidance::toktrie::TokEnv>,
) -> Result<(Vec<u32>, Vec<String>), MistralRsError> {
    match &sampling_params.stop_toks {
        None => Ok((vec![], vec![])),
        Some(StopTokens::Ids(ids)) => {
            if let Some(tok_env) = tok_env.as_ref() {
                let tok_trie = tok_env.tok_trie();
                for id in ids {
                    if tok_trie.has_extensions(tok_trie.token(*id)) {
                        return Err(MistralRsError::Unsupported(format!(
                            "stop token {} is also a prefix of other tokens and cannot be used directly",
                            id
                        )));
                    }
                }
            }
            Ok((ids.clone(), vec![]))
        }
        Some(StopTokens::Seqs(seqs)) => {
            let Some(tokenizer) = tokenizer else {
                return Err(MistralRsError::Unsupported(
                    "stop sequences require a tokenizer".into(),
                ));
            };
            let mut stop_toks = Vec::new();
            let mut stop_strings = Vec::new();
            for stop_txt in seqs {
                let encoded = tokenizer
                    .encode_fast(stop_txt.to_string(), true)
                    .map_err(|e| MistralRsError::Unsupported(format!("encode stop sequence: {e}")))?;
                let toks = encoded.get_ids().to_vec();
                if toks.len() == 1 {
                    if tok_env.as_ref().is_some_and(|tok_env| {
                        let tok_trie = tok_env.tok_trie();
                        tok_trie.has_extensions(tok_trie.token(toks[0]))
                    }) {
                        stop_strings.push(stop_txt.clone());
                    } else {
                        stop_toks.push(toks[0]);
                    }
                } else {
                    stop_strings.push(stop_txt.clone());
                }
            }
            Ok((stop_toks, stop_strings))
        }
    }
}

fn build_preallocated_cache(
    pipeline: &dyn Pipeline,
    _num_hidden_layers: usize,
    n_tokens: usize,
) -> Result<Option<(Tensor, Tensor)>, MistralRsError> {
    if !matches!(
        pipeline.category(),
        crate::ModelCategory::Text | crate::ModelCategory::Vision { .. }
    ) || !pipeline.do_preallocated_cache()
    {
        return Ok(None);
    }

    let metadata = pipeline.get_metadata();
    let model_metadata = metadata.model_metadata.as_ref().ok_or_else(|| {
        MistralRsError::Unsupported(
            "normal-cache models require model metadata for stateful decode".into(),
        )
    })?;
    let required_blocks = n_tokens.div_ceil(crate::kv_cache::NormalCache::CACHE_GROW_SIZE);
    let max_seq_len = required_blocks * crate::kv_cache::NormalCache::CACHE_GROW_SIZE;
    let k_shape = (
        1usize,
        model_metadata.num_kv_heads(),
        max_seq_len,
        model_metadata.k_head_dim(),
    );
    let v_shape = (
        1usize,
        model_metadata.num_kv_heads(),
        max_seq_len,
        model_metadata.v_head_dim(),
    );
    let dtype = metadata.activation_dtype;
    let device = pipeline.device();
    let k_seq_cache = Tensor::zeros(k_shape, dtype, &device)
        .map_err(|e| MistralRsError::Unsupported(format!("allocate preallocated K cache: {e}")))?;
    let v_seq_cache = if k_shape == v_shape {
        k_seq_cache.clone()
    } else {
        Tensor::zeros(v_shape, dtype, &device).map_err(|e| {
            MistralRsError::Unsupported(format!("allocate preallocated V cache: {e}"))
        })?
    };
    Ok(Some((k_seq_cache, v_seq_cache)))
}
