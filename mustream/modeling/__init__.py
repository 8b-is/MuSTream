"""
Everything needed to build, initialize and reload the MERT backbones.

    mert/                 the original (Hugging Face) MERT, plus the small
                          bidirectional student built on its MERTModel
    moshi/                vendored moshi streaming modules (conv, KV-cache
                          transformer, rope), with local additions such as
                          the block-causal attention lookahead
    streaming/            the causal streaming backbone, FullStreamingMERTModel
    load_mert.py          the pretrained MERT teacher and its raw weights
    init_streaming.py     pretrained MERT weights -> streaming backbone
    load_checkpoint.py    train_mert_causal_distill.py checkpoints -> trained students

Import from the submodules directly (e.g.
`from modeling.init_streaming import load_full_streaming_mert`), with the
mustream/ folder on sys.path.
"""
