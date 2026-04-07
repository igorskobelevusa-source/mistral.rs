use mistralrs::{DecodeSessionConfig, ModelDType, TextModelBuilder};

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let model = TextModelBuilder::new("hf-internal-testing/tiny-random-LlamaForCausalLM")
        .with_force_cpu()
        .with_dtype(ModelDType::F32)
        .build_stateful()
        .await?;

    let mut session = model.new_decode_session(DecodeSessionConfig::default())?;
    let out = model.prefill(&mut session, &[1, 2, 3, 4])?;

    println!(
        "stateful_smoke_ok prompt_tokens={} cached_prompt_tokens={} staged_only={}",
        out.prompt_tokens, out.cached_prompt_tokens, out.staged_only
    );
    Ok(())
}
