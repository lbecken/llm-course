"""
Supervised fine-tuning with a loss mask (Module 14).

The model, the loss and the optimizer are those of pretraining. Three things change:

    data      curated conversations rendered into the chat format
    targets   positions the assistant does not have to produce get target -1 (ignored)
    packing   conversations are packed with padding, never cropped (Module 08)

nanochat's version is scripts/chat_sft.py; the masking is its line
`targets[mask_targets == 0] = -1`.
"""

from __future__ import annotations

import random

import torch

from mlle.data.packing import pack_rows
from mlle.finetune.chat import ChatTokenizer
from mlle.foundations.optim import lr_multiplier
from mlle.inference.kvcache import KVCache


def build_sft_rows(tok: ChatTokenizer, convs: list[dict], row_capacity: int):
    """Render and pack conversations. Returns (inputs, targets), each (N_rows, row_capacity - 1).
    targets is -1 wherever the loss must be ignored (context tokens and padding)."""
    rendered = [tok.render_conversation(c) for c in convs]
    # pack ids; carry the masks along by packing index lists the same way
    order = list(range(len(rendered)))
    packed_idx, _ = pack_rows([[i] * len(rendered[i][0]) for i in order], row_capacity, pad_id=-1, strategy="bestfit_pad")
    bos = tok.get_bos_token_id()
    rows, masks = [], []
    for row in packed_idx:
        ids, mask, seen = [], [], []
        for i in row:
            if i != -1 and i not in seen:
                seen.append(i)
                ids.extend(rendered[i][0])
                mask.extend(rendered[i][1])
        pad = row_capacity - len(ids)
        rows.append(ids + [bos] * pad)  # pad with BOS, as nanochat does; those positions are masked
        masks.append(mask + [0] * pad)
    ids_t = torch.tensor(rows, dtype=torch.long)
    mask_t = torch.tensor(masks, dtype=torch.long)
    inputs, targets = ids_t[:, :-1], ids_t[:, 1:].clone()
    targets[mask_t[:, 1:] == 0] = -1  # the mask of the TARGET token decides
    return inputs, targets


def sft_train(model, inputs, targets, steps: int = 300, batch_size: int = 32, lr: float = 3e-3, seed: int = 0,
              params=None, extra_batch_fn=None, verbose: bool = False) -> list[float]:
    """Fine-tune on packed rows. `params` restricts which parameters are trained (LoRA).
    `extra_batch_fn()` may return an additional (x, y) batch mixed into every step
    (e.g. pretraining data, to counter forgetting). Returns the loss curve."""
    params = [p for p in (params if params is not None else model.parameters()) if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=lr, betas=(0.9, 0.95), weight_decay=0.0)
    g = torch.Generator().manual_seed(seed)
    model.train()
    losses = []
    for step in range(steps):
        for group in opt.param_groups:
            group["lr"] = lr * lr_multiplier(step, steps, warmup_steps=10, warmdown_ratio=0.5, final_lr_frac=0.0)
        idx = torch.randint(0, inputs.shape[0], (batch_size,), generator=g)
        loss = model(inputs[idx], targets[idx])
        if extra_batch_fn is not None:
            x2, y2 = extra_batch_fn()
            loss = 0.5 * loss + 0.5 * model(x2, y2)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        losses.append(loss.item())
        if verbose and step % 50 == 0:
            print(f"  sft step {step:4d} | loss {loss.item():.4f}")
    model.eval()
    return losses


@torch.no_grad()
def complete(model, prompt_ids: list[int], stop_ids: set[int], max_new_tokens: int = 12, temperature: float = 0.0,
             num_samples: int = 1, generator=None) -> list[list[int]]:
    """Generate up to `max_new_tokens` after the prompt, for `num_samples` parallel samples,
    stopping each row at a stop token (not included in the result). Uses the KV cache:
    the prompt is processed once (batch 1) and the cache is replicated, as in nanochat's Engine."""
    model.eval()
    device = next(model.parameters()).device
    idx = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    cache = KVCache.for_model(model, 1, max_len=len(prompt_ids) + max_new_tokens)
    logits = model(idx, cache=cache)[:, -1, :]
    if num_samples > 1:
        cache = cache.expand_batch(num_samples)
        logits = logits.expand(num_samples, -1)
    outputs: list[list[int]] = [[] for _ in range(num_samples)]
    done = [False] * num_samples
    for _ in range(max_new_tokens):
        if temperature == 0.0:
            nxt = logits.argmax(dim=-1, keepdim=True)
        else:
            nxt = torch.multinomial(torch.softmax(logits / temperature, dim=-1), 1, generator=generator)
        for i, t in enumerate(nxt[:, 0].tolist()):
            if not done[i]:
                if t in stop_ids:
                    done[i] = True
                else:
                    outputs[i].append(t)
        if all(done) or cache.pos >= cache.max_len:
            break
        logits = model(nxt, cache=cache)[:, -1, :]
    return outputs


def chat_accuracy(model, tok: ChatTokenizer, convs: list[dict], max_new_tokens: int = 12) -> float:
    """Exact-match accuracy of greedy chat completions against the reference answers."""
    stop = {tok.encode_special("<|assistant_end|>"), tok.get_bos_token_id()}
    correct = 0
    for conv in convs:
        out = complete(model, tok.render_for_completion(conv), stop, max_new_tokens)[0]
        correct += tok.decode(out) == conv["problem"].answer
    return correct / len(convs)


def raw_completion_accuracy(model, tok: ChatTokenizer, problems, max_new_tokens: int = 6) -> float:
    """Accuracy when the model is prompted the way it was PRETRAINED: '<|bos|>47+38=' -> '85'."""
    bos = tok.get_bos_token_id()
    correct = 0
    for p in problems:
        out = complete(model, [bos] + tok.encode(p.question), {bos}, max_new_tokens)[0]
        correct += tok.decode(out) == p.answer
    return correct / len(problems)
