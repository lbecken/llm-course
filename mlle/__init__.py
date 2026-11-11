"""
mlle — course code for "Modern LLM Engineering".

The package is organised to follow the course parts:

    mlle.data         tiny built-in datasets (bitmap digits, a synthetic language, arithmetic)
    mlle.foundations  Part I   — MLP, n-grams, BPE, positions, attention, MiniGPT, optimizers
    mlle.inference    Part III — sampling, KV cache, attention variants, quantization
    mlle.bench        Part III — timing / memory measurement harness
    mlle.finetune     Part IV  — chat rendering, SFT with loss masks, LoRA, REINFORCE, DPO
    mlle.agents       Part V   — tools, reasoning strategies, agent loop, multi-agent, experiment loop

Everything here is written to be read. Production concerns (speed, distributed training,
dtype management) are deliberately left to nanochat, which the lessons walk through
side by side with this code.
"""

__version__ = "0.1.0"
