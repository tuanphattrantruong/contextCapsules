# GGUF models (not shipped)

The engine experiments (E10, E12, E13, E14) need two quantised GGUF checkpoints in
this directory. They are ~5.6 GB together and are therefore not included; download
them here before running those experiments.

`experiments/run_e10.sh` fetches both automatically if they are missing. To fetch
them by hand:

```sh
curl -L -o qwen2.5-0.5b-instruct-q8_0.gguf \
  https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct-GGUF/resolve/main/qwen2.5-0.5b-instruct-q8_0.gguf

curl -L -o llama-3.1-8b-instruct-q4km.gguf \
  https://huggingface.co/bartowski/Meta-Llama-3.1-8B-Instruct-GGUF/resolve/main/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf
```

| File | Size | SHA-256 |
|---|---|---|
| `qwen2.5-0.5b-instruct-q8_0.gguf` | 675,710,816 B | `ca59ca7f13d0e15a8cfa77bd17e65d24f6844b554a7b6c12e07a5f89ff76844e` |
| `llama-3.1-8b-instruct-q4km.gguf` | 4,920,739,232 B | `7b064f5842bf9532c91456deda288a1b672397a54fa729aa665952863033557c` |

The digests above are the files used for the reference results. E12 binds the full
GGUF SHA-256 into a capsule's compatibility descriptor, so a different quantisation
of the same model is correctly refused rather than silently attached.

Everything except E10/E12/E13/E14 runs without these files.
