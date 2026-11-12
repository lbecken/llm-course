# Module 08 — Knowledge check

Closed book, about 30 minutes. Solutions: [`solutions/08-data-and-pretraining.md`](../../solutions/08-data-and-pretraining.md).

1. **(Concept)** Give an operational definition of "data quality" for pretraining, and explain why it cannot be decided by inspection alone.

2. **(Concept)** What is the difference between a validation set and a test set? Which does nanochat's last data shard play the role of?

3. **(Diagnosis)** A model reports 0.21 bits per byte on a "held-out" set of English web text, where comparable models reach about 0.9. List three explanations in order of likelihood.

4. **(Calculation)** A depth-12 run needs 1.32 B training tokens. With best-fit cropping discarding 35 % of tokens and 47 M tokens per shard, how many shards must be on disk to avoid a second epoch?

5. **(Concept)** Why does nanochat crop documents in pretraining but pad in fine-tuning?

6. **(Concept)** Why is a row that starts in the middle of a document a problem, and why is the resulting extra loss partly "unfair" to the model?

7. **(Code reading)** In the pretraining loader, `row_capacity = T + 1`, and after a row is built: `cpu_inputs.copy_(row_buffer[:, :-1])`, `cpu_targets.copy_(row_buffer[:, 1:])`. If a row holds three short documents, each beginning with BOS, what is the model asked to predict at the last token of the first document? Is that a problem?

8. **(Log reading)** `step 02000/02520 (79.37%) | loss: 3.412 | lrm: 0.35 | dt: 9800.00ms | tok/sec: 53,498 | bf16_mfu: 0.00 | epoch: 2 pq: 3 rg: 17`. State four things you can conclude.

9. **(Reproducibility)** Which three files make up a nanochat checkpoint, and which one would you open to find out with what learning rates a model was trained?

10. **(Experimental reasoning)** Two validation-bpb numbers: 0.83 from a run in January with the streaming loader and 0.79 from a run in February with the BOS-aligned loader. Can you conclude the February model is better? What would make the comparison valid?

## Mastery criteria

- [ ] Explain train/validation/test discipline and three ways it is violated.
- [ ] Explain the three packing strategies and compute their waste for a simple length distribution.
- [ ] Trace a document from a parquet row to a `(B, T)` batch in nanochat.
- [ ] Run the miniature nanochat pipeline and locate every sizing decision in its log.
- [ ] Diagnose a run from its loss values and throughput.
- [ ] At least 8 of 10 quiz questions correct.
