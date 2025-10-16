# InferSim: A Lightweight LLM Inference Performance Simulator

InferSim is a lightweight simulator for LLM inference, written in pure Python without any 3rd-party depenencies. It calculates the TTFT, TPOT and throughput TGS (tokens/GPU/second) based on computation complexity FLOPs (Floating-Point Operations), GPU computing power FLOPS (Floating-Point Operations per Second), GPU memory bandwidth and MFU (Model FLOPs Utilization) obtained by benchmarking the state-of-the-art LLM kernels. For multi-GPU, multi-node deployment, InferSim also estimates the communication latency according to data volume and bandwidth.

The main use cases of InferSim include:
- **Model-Sys co-design**: predicting inference performance given the hyperparameters of a model.
- **Inference performance analysis**: quantifying performance bottlenecks, such as compute-bound or IO-bound, and supporting optimization efforts.

For more details, please check [InferSim Technical Report](https://github.com/user-attachments/files/22580184/infersim_tech_report.pdf).

## Simulation Result

| Model | GPU | Prefill TGS(Actual) | Prefill TGS(Sim) | Decode TGS(Actual) | Decode TGS(Sim) | Notes |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| DeepSeek-V3 | H800 | 7839 | 9034 | 2324 | 2675 | Actual data from [deepseek/profile-data](https://github.com/deepseek-ai/profile-data/). Simulated with same setup: [example/deepseek-v3/](./example/deepseek-v3/). |
| Qwen3-30B-A3B-BF16 | H20 | 16594 | 17350 | 2749 | 2632 | Actual data tested with SGLang, simulation example: [example/qwen3-30B-A3B/](./example/qwen3-30B-A3B/). |
| Qwen3-8B-FP8 | H20 | 15061 | 16328 | 2682 | 2581 | Actual data tested with SGLang, simulation example: [example/qwen3-8B/](./example/qwen3-8B/). |

## Supported Features

- **Attention**: MHA/GQA, MLA. Benchmarked on FlashInfer, FlashAttention-3, FlashMLA.
- **MoE**: GroupedGEMM. Benchmarked on DeepGEMM.
- **Linear**: GEMM. Benchmarked on DeepGEMM.
- **Parallelization**: DP Attn, EP MoE.
- **Large EP**: DeepEP dispatch and combine, with normal and low_latency mode.

## Help

```
$ python3 main.py --help
usage: main.py [-h] --config-path CONFIG_PATH [--device-type {H20,H800}] [--world-size WORLD_SIZE] [--num-nodes NUM_NODES]
               [--max-prefill-tokens MAX_PREFILL_TOKENS] [--decode-bs DECODE_BS] [--target-tgs TARGET_TGS]
               [--target-tpot TARGET_TPOT] [--target-isl TARGET_ISL] [--target-osl TARGET_OSL] [--use-fp8-gemm]
               [--use-fp8-kv] [--enable-deepep] [--enable-tbo] [--sm-ratio SM_RATIO] [--prefill-only] [--decode-only]

optional arguments:
  -h, --help            show this help message and exit
  --config-path CONFIG_PATH
                        The path of the hf model config.json
  --device-type {H20,H800}
                        Device type
  --world-size WORLD_SIZE
                        Num of GPUs
  --num-nodes NUM_NODES
                        Num of nodes
  --max-prefill-tokens MAX_PREFILL_TOKENS
                        Max prefill tokens
  --decode-bs DECODE_BS
                        Decoding batchsize. If not specified, bs = tgs * tpot.
  --target-tgs TARGET_TGS
                        Target tokens/s per GPU
  --target-tpot TARGET_TPOT
                        TPOT in ms
  --target-isl TARGET_ISL
                        Input sequence length, in tokens
  --target-osl TARGET_OSL
                        Output sequence length, in tokens
  --use-fp8-gemm        Use fp8 gemm
  --use-fp8-kv          Use fp8 kvcache
  --enable-deepep       Enable DeepEP
  --enable-tbo          Enable two batch overlap
  --sm-ratio SM_RATIO   In TBO DeepEP normal mode, the SM ratio used for computation
  --prefill-only        Only simulate prefill
  --decode-only         Only simulate decoding
```

## Example

```
$ bash example/qwen3-30B-A3B/decode.sh

================ Simulator Result ================
Device type:                             H20
World size:                              4
Attn type:                               MHA/GQA
Use FP8 GEMM:                            0
Use FP8 KV:                              0
------------------Model Weights-------------------
One attn params size (MB):               36.00
One expert params size (MB):             9.00
Per GPU params size (GB):                15.19
---------------------KV Cache---------------------
KV cache space (GB):                     60.81
Input seq len:                           4096
Output seq len:                          2048
Target decode batchsize:                 100
Target per-token KV cache size (KB):     103.79
Current per-token KV cache size (KB):    96.00
----------------------FLOPs-----------------------
Num hidden layers:                       48
Per-token per-layer attn core (GFLOPs):  0.08
Per-token per-layer MoE/FFN (GFLOPs):    0.08
Per-token per-layer others (GFLOPs):     0.04
Per-token attn core (GFLOPs):            4.03
Per-token MoE (GFLOPs):                  3.62
Per-token others (GFLOPs):               1.81
Per-token total (GFLOPs):                9.46
---------------------Decoding---------------------
Attn core MFU:                           0.15
Attn core latency (us):                  361.77
KV loading latency (us):                 298.02
QKV_proj latency (us):                   31.03
O_proj latency (us):                     16.95
Routed experts/FFN MFU:                  0.18
Routed experts/FFN latency (us):         269.28
Experts loading latency (us):            85.83
Comm before MoE/FFN (us):                4.24
Comm after MoE/FFN (us):                 4.24
TPOT (ms):                               38.00
Throughput (TGS):                        2632
```

## Acknowledgement

This work is developed and maintained by Alimama AI Infra Team & Future Living Lab, Alibaba Group.
