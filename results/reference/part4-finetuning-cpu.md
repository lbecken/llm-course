# Reference result: Part IV labs (fine-tuning, LoRA, REINFORCE, DPO) on CPU

| | |
|---|---|
| Date | 2026-11-12 |
| Hardware | 4-vCPU x86-64 sandbox, 15 GiB RAM, no GPU |
| Software | Python 3.13, torch 2.15.0+cpu |
| Status | **Measured** by the course author. One seed per cell unless stated. 300 evaluation questions per accuracy (200 in the DPO lab): standard error up to 0.03 |

All four labs start from one cached base model: MiniGPT, 2 layers × 96, 231 744
parameters, 1 500 pretraining steps on Pocket sentences and 3-digit addition facts
(`mlle.finetune.base.get_base_model`; 45 s to train).

## Lab 14a — `python labs/finetuning/14_chat_sft.py` (about 3 min)

1 000 fine-tuning steps, learning rate 2e-3, 3 000 conversations.

| Variant | chat (seen) | chat (held-out) | raw (held-out) | Pocket loss |
|---|---:|---:|---:|---:|
| base model, no fine-tuning | 0.000 | 0.000 | 0.770 | 0.4034 |
| SFT on conversations only | 0.877 | 0.880 | 0.000 | 11.1749 |
| SFT + replay of pretraining batches | 0.870 | 0.880 | 0.967 | 0.4028 |
| control: more pretraining, no conversations | 0.000 | 0.000 | 0.903 | 0.4496 |
| control: SFT from random initialisation | 0.083 | 0.033 | 0.000 | 18.5142 |

With `--n-convs 100`: chat (held-out) 0.093 (SFT), 0.080 (SFT + replay), 0.000 (random
initialisation). With `--n-convs 20`: 0.003, 0.010, 0.003.

## Lab 14b — `python labs/finetuning/14_lora.py` (about 3–4 min)

1 000 steps, learning rate 1e-3, seed 0.

| Variant | trainable | chat (held-out) | Pocket loss, adapter on | adapter off |
|---|---:|---:|---:|---:|
| full fine-tuning (no replay) | 231 744 | 0.880 | 11.1749 | n/a |
| LoRA r=8 on q, v; token rows frozen | 6 144 | 0.017 | 1.2073 | 0.4034 |
| LoRA r=8 on q, v + chat-token rows | 7 680 | 0.040 | 0.7508 | 0.4034 |
| LoRA r=2 on all matrices + chat-token rows | 8 448 | 0.330 | 1.0235 | 0.4034 |
| LoRA r=8 on all matrices + chat-token rows | 29 184 | 0.740 | 0.6374 | 0.4034 |
| LoRA r=32 on all matrices + chat-token rows | 112 128 | 0.823 | 2.4786 | 0.4034 |

Seeds 0, 1, 2 (all matrices + token rows): r=8 at lr 1e-3: 0.740, 0.727, 0.743; at 3e-3:
0.727, 0.843, 0.790. r=32 at 1e-3: 0.823, 0.813, 0.827; at 3e-3: 0.790, 0.863, 0.887.
Rank 8 on other subsets (lr 1e-3, seed 0): `c_fc` only 0.357; q, v + `c_fc` 0.430;
`c_fc` + `c_proj` 0.490.

## Lab 15a — `python labs/finetuning/15_reinforce.py` (about 3 min)

Start: chat model after 300 SFT steps with replay. 300 training prompts, 16 prompts × 8
samples per step, 300 steps, learning rate 3e-4.

| Run | metric | step 0 | 75 | 150 | 225 | 300 |
|---|---|---:|---:|---:|---:|---:|
| A exact reward | accuracy, training prompts | 0.533 | 0.577 | 0.597 | 0.620 | 0.637 |
| | accuracy, held-out | 0.523 | 0.537 | 0.537 | 0.510 | 0.517 |
| | lazy checker | 0.943 | 0.927 | 0.880 | 0.890 | 0.900 |
| B lazy reward | accuracy, training prompts | 0.533 | 0.497 | 0.500 | 0.480 | 0.477 |
| | accuracy, held-out | 0.523 | 0.483 | 0.450 | 0.457 | 0.460 |
| | lazy checker | 0.943 | 0.957 | 0.970 | 0.980 | 0.987 |
| C labels (SFT) | accuracy, training prompts | 0.533 | 0.687 | 0.760 | 0.823 | 0.843 |
| | accuracy, held-out | 0.523 | 0.537 | 0.543 | 0.540 | 0.557 |

Pocket loss after A: 0.4073; after B: 0.4180 (start 0.4051). Fraction of informative
prompts in logged steps: A 0.31–0.62, B 0.50–0.62. With `--lr 1e-3`, run A: training
prompts 0.533, 0.453, 0.497, 0.570, 0.593; held-out 0.523, 0.373, 0.420, 0.483, 0.493;
Pocket loss 2.58. With `--num-samples 2 --prompts-per-step 64`, run A: training prompts
0.630, held-out 0.570 at step 300; informative fraction 0.12–0.31.

## Lab 15b — `python labs/finetuning/15_dpo.py` (about 3 min)

100 DPO steps, 300 pairs.

| Part 1: style preference | long style | accuracy | log p preferred | log p rejected |
|---|---:|---:|---:|---:|
| start (SFT on a 50/50 mixture of styles) | 0.685 | 0.735 | −1.29 | −2.46 |
| DPO β=1, lr 3e-4 | 0.990 | 0.325 | −3.04 | −24.24 |
| DPO β=5, lr 1e-4 | 0.981 | 0.685 | −0.87 | −7.31 |
| DPO β=1, lr 3e-4 + 0.3 × SFT on preferred | 0.998 | 0.775 | −0.19 | −12.16 |

| Part 2: correctness, near-miss negatives | accuracy | log p preferred | log p rejected |
|---|---:|---:|---:|
| start (under-trained chat model) | 0.505 | −1.98 | −8.97 |
| DPO β=1, lr 3e-4 | 0.380 | −2.86 | −12.20 |
| DPO β=1, lr 3e-4 + 0.3 × SFT on preferred | 0.495 | −1.82 | −10.81 |

Other step counts (style / accuracy): 20 steps: 0.990 / 0.445, 0.944 / 0.700,
0.995 / 0.650; 300 steps: 0.986 / 0.345, 0.988 / 0.655, 1.000 / 0.725.

## Notes

- Timings are for a quiet machine; running anything else in parallel multiplied them by
  three in the author's session.
- The numbers characterise a 232 k-parameter model on a synthetic task. The *patterns*
  (forgetting, the value of controls, reward hacking, likelihood displacement) are the
  point; do not transfer magnitudes to large models.
- GPU: not measured. The labs run unchanged on a GPU but are too small to benefit.
