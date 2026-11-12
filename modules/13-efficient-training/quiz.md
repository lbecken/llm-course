# Module 13 — Knowledge check

Closed book, about 30 minutes. Solutions: [`solutions/13-efficient-training.md`](../../solutions/13-efficient-training.md).

1. **(Concept)** List the four contributions to the memory of a training step. Which one grows with the batch, and what is it proportional to?

2. **(Concept)** fp16 and bf16 both use 16 bits. What does each give up relative to fp32, and which failure does each risk?

3. **(Calculation)** In bf16 (8 significant bits), what is the spacing between representable values near 4.0? A weight of 4.0 receives 500 updates of +0.01 each. What is its final stored value if weights are kept in bf16?

4. **(Concept)** State the rule of mixed-precision training in one sentence, and name the two parts of the computation that are conventionally kept in fp32 besides the weights.

5. **(Concept)** Why does fp16 training need a gradient scaler while bf16 training does not?

6. **(Calculation)** Total batch 262 144 tokens, sequence length 1 024, device batch size 16 rows, 2 GPUs. How many accumulation steps per optimizer step? What happens to that number, and to the training result, on 1 GPU?

7. **(Code reading)** In an accumulation loop the loss is divided by the number of micro-batches before `backward()`. What goes wrong if the division is omitted? What goes wrong if `zero_grad()` is called inside the micro-batch loop?

8. **(Concept)** Compare gradient accumulation and activation checkpointing: what does each trade, and when do you need the second?

9. **(Calculation)** A model needs 8.0e8 FLOPs per token and trains at 50 000 tokens/s on a device with a peak of 2.0e14 FLOPs/s. What is the MFU? Give two legitimate reasons it is below 100 %.

10. **(Concept)** What does `torch.compile` do, and why does nanochat keep an uncompiled copy of the model?

11. **(Method)** You time 20 training steps with `time.time()` around `loss.backward()` on a GPU and compute tokens/s. Name three errors in this measurement.

12. **(Concept)** In data-parallel training on $N$ GPUs, what is computed locally, what is communicated, and what must be identical on all ranks after a step? How does nanochat's approach differ from wrapping the model in `DistributedDataParallel`?

## Mastery criteria

- [ ] Account for training memory and predict the effect of batch size on it.
- [ ] Choose a number format and justify where fp32 must be kept.
- [ ] Implement gradient accumulation and verify its equivalence.
- [ ] Compute tokens/s and MFU from a log and time a step correctly.
- [ ] Translate a multi-GPU recipe into a single-device run with the same optimisation.
- [ ] Complete **Milestone 3**: a benchmark report with a correctness column for every claim.
- [ ] At least 10 of 12 quiz questions correct.
