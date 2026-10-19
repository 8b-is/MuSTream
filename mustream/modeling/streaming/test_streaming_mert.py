"""
Tests for streaming_mert.py: causality, chunked streaming vs. a full pass,
KV-cache wraparound, rope translation invariance, and the small student.

    python -m modeling.streaming.test_streaming_mert      # from mustream/
"""
import torch

from ..init_streaming import build_small_streaming_mert, load_full_streaming_mert



def check_causal(model, forward_fn, seq_len_samples, hop_length, label):
    """Check that perturbing future audio leaves earlier output frames unchanged."""
    torch.manual_seed(0)
    x = torch.randn(1, seq_len_samples)
    perturb_from_sample = seq_len_samples // 2
    safe_frames = perturb_from_sample // hop_length - 1

    x_perturbed = x.clone()
    x_perturbed[:, perturb_from_sample:] += torch.randn_like(x_perturbed[:, perturb_from_sample:]) * 10.0

    with torch.no_grad():
        out_a = forward_fn(model, x)
        out_b = forward_fn(model, x_perturbed)

    early_a = out_a[:, :safe_frames]
    early_b = out_b[:, :safe_frames]
    max_diff = (early_a - early_b).abs().max().item()
    late_diff = (out_a[:, safe_frames:] - out_b[:, safe_frames:]).abs().max().item()
    print(f"[{label}] causality check: max diff in early (past) frames = {max_diff:.3e} "
          f"(should be ~0), max diff in later frames = {late_diff:.3e} (should be > 0)")
    assert max_diff < 1e-4, f"{label}: future perturbation leaked into past frames! diff={max_diff}"
    assert late_diff > 1e-3, f"{label}: perturbation had no effect at all -- check test is exercising the model"


if __name__ == "__main__":
    model = load_full_streaming_mert()
    n_params = sum(p.numel() for p in model.parameters())
    print(f"loaded full-streaming MERT: {n_params / 1e6:.1f}M params")

    dummy_input = torch.randn(1, 16000)
    with torch.no_grad():
        out, _ = model(dummy_input)
        _, hidden_states = model(dummy_input, output_hidden_states=True)
    print(f"output shape: {out.shape}")
    print(f"num hidden_states layers: {len(hidden_states)}")

    def forward_fn(m, x):
        with torch.no_grad():
            return m(x)[0]

    check_causal(model, forward_fn, seq_len_samples=16000, hop_length=320, label="full-streaming-full-mask-pass")

    # Chunked streaming must match a single full pass.
    torch.manual_seed(1)
    hop_length = 320
    x = torch.randn(1, 16000)
    with torch.no_grad():
        full_out, _ = model(x)

        chunk_frames = 5
        chunk_samples = chunk_frames * hop_length
        ys = []
        with model.feature_extractor.streaming(1), model.transformer.streaming(1):
            for offset in range(0, x.shape[-1], chunk_samples):
                chunk = x[:, offset:offset + chunk_samples]
                feat = model.feature_extractor(chunk).transpose(1, 2)
                proj = model.feature_projection(feat)
                if not model.config.do_stable_layer_norm:
                    proj = model.final_layer_norm(proj)  # post-norm: same placement as forward()
                ys.append(model.transformer(proj))
        streamed_out = torch.cat(ys, dim=1)
        if model.config.do_stable_layer_norm:
            streamed_out = model.final_layer_norm(streamed_out)  # pre-norm: same placement as forward()

    n = min(full_out.shape[1], streamed_out.shape[1])
    max_diff = (full_out[:, :n] - streamed_out[:, :n]).abs().max().item()
    print(f"[chunked-streaming] max diff vs. single full pass = {max_diff:.3e} (should be ~0)")
    assert max_diff < 1e-3, f"chunked streaming diverged from full pass! diff={max_diff}"

    # KV-cache wraparound: stream well past a small context with several chunk sizes.
    # TF32 is disabled so the tolerance reflects the algorithm, not matmul rounding.
    tf32_matmul_was = torch.backends.cuda.matmul.allow_tf32
    tf32_cudnn_was = torch.backends.cudnn.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False   # conv feature extractor uses cudnn
    try:
        context = 200
        model_ctx = load_full_streaming_mert(context=context)
        model_ctx.eval()
        torch.manual_seed(2)
        x = torch.randn(1, 600 * hop_length)  # 600 frames -> wraps a 200-frame ring ~3x
        with torch.no_grad():
            full_ctx, _ = model_ctx(x)
            for chunk_frames in [1, 2, 5, 25, 200]:
                chunk_samples = chunk_frames * hop_length
                ys = []
                with model_ctx.feature_extractor.streaming(1), model_ctx.transformer.streaming(1):
                    for offset in range(0, x.shape[-1], chunk_samples):
                        ys.append(model_ctx(x[:, offset:offset + chunk_samples])[0])
                streamed = torch.cat(ys, dim=1)
                m = min(full_ctx.shape[1], streamed.shape[1])
                d = (full_ctx[:, :m] - streamed[:, :m]).abs().max().item()
                print(f"[wraparound] context={context}, chunk={chunk_frames:3d} frames: "
                      f"max diff vs. full pass = {d:.3e} (should be ~0)")
                assert d < 1e-3, (
                    f"streaming with chunk={chunk_frames} diverged from full pass past the "
                    f"ring capacity (diff={d}) -- RingKVCache eviction ordering is wrong."
                )
    finally:
        torch.backends.cuda.matmul.allow_tf32 = tf32_matmul_was
        torch.backends.cudnn.allow_tf32 = tf32_cudnn_was

    # Translation invariance: the same audio, with and without a prefix, must give
    # the same tail features under rope. The tail sits beyond the receptive field
    # (depth * context frames) from the seam.
    def translation_shift_diff(m, context, n_prefix=100, n_tail=64, hop_length=320):
        depth = len(m.transformer.layers)
        reach = depth * (context - 1) + 4
        n_core = reach + n_tail
        torch.manual_seed(3)
        core = torch.randn(1, n_core * hop_length)
        prefix = torch.randn(1, n_prefix * hop_length)
        with torch.no_grad():
            alone, _ = m(core)
            shifted, _ = m(torch.cat([prefix, core], dim=-1))
        return (alone[:, -n_tail:] - shifted[:, -n_tail:]).abs().max().item()

    context = 8
    diffs = {}
    for scheme in ("sin", "rope"):
        m = load_full_streaming_mert(context=context, positional_embedding=scheme)
        m.eval()
        diffs[scheme] = translation_shift_diff(m, context)
        print(f"[translation-invariance] positional_embedding={scheme:>4}: "
              f"max diff when the same audio is shifted later in the stream = {diffs[scheme]:.3e}")
    assert diffs["rope"] < 1e-4, (
        f"rope should be translation-invariant but shifted by {diffs['rope']:.3e} -- "
        "is rope actually reaching the attention layers?"
    )
    assert diffs["sin"] > 100 * diffs["rope"], (
        f"sin is expected to be strongly position-dependent (got {diffs['sin']:.3e}); "
        "if it is not, this test is not measuring what it claims to."
    )
    print("[translation-invariance] rope is position-invariant, sin is not -- as expected.")

    # Small streaming student (random init): causality and chunked streaming.
    small = build_small_streaming_mert(
        "m-a-p/MERT-v1-95M", context=200, positional_embedding="rope",
        num_hidden_layers=4,   # shallow to keep the test cheap
    )
    small.eval()
    print(f"\nsmall streaming student: {sum(p.numel() for p in small.parameters()) / 1e6:.1f}M params, "
          f"post-norm={not small.config.do_stable_layer_norm}, conv_dim={small.config.conv_dim}")
    check_causal(small, forward_fn, seq_len_samples=300 * hop_length, hop_length=hop_length,
                 label="small-streaming")
    torch.manual_seed(4)
    x = torch.randn(1, 300 * hop_length)
    with torch.no_grad():
        small_full, small_hidden = small(x, output_hidden_states=True)
        assert small_hidden is not None and len(small_hidden) == small.config.num_hidden_layers + 1
        assert small_full.shape[1] == 300, (
            f"the causal conv stack should emit exactly one frame per hop, got {small_full.shape[1]}"
        )
        for chunk_frames in [1, 5, 50]:
            chunk_samples = chunk_frames * hop_length
            ys = []
            with small.feature_extractor.streaming(1), small.transformer.streaming(1):
                for offset in range(0, x.shape[-1], chunk_samples):
                    ys.append(small(x[:, offset:offset + chunk_samples])[0])
            streamed = torch.cat(ys, dim=1)
            n = min(small_full.shape[1], streamed.shape[1])
            d = (small_full[:, :n] - streamed[:, :n]).abs().max().item()
            print(f"[small-streaming] chunk={chunk_frames:2d} frames: max diff vs. full pass = {d:.3e}")
            assert d < 1e-3, f"small streaming student diverged from its full pass (diff={d})"
