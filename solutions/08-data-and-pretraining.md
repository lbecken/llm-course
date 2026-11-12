# Solutions — Module 08

## Lab exercises

### Exercise 1: design the splits

1. Split by **time** and by **customer**: train on the earliest period, validate on a later period, test on the latest, and ensure no customer appears in more than one split. A random split leaks in two ways. *Group leakage:* the same customer's other conversations (same products, phrasing, recurring problem) are in training, so the model is partly tested on memorised specifics. *Temporal leakage:* training on conversations from after the test conversations gives the model knowledge of products, incidents and policies that it will not have when deployed on next year's customers.
2. The boilerplate is trivially predictable, so it lowers the average loss without reflecting any ability to help customers; a large part of the "improvement" between two models may be boilerplate. Whole-conversation deduplication does not remove it, because the conversations differ elsewhere. Remedies: strip or mask the boilerplate, or report loss on the remaining tokens only (a loss mask, as in Module 14).
3. The validation set has been used for hundreds of decisions; the system is adapted to it, and its loss is optimistically biased. A test set should have been set aside at the start and evaluated once, at the end.

### Exercise 2: packing

Reference (author): median-400 documents: crop 63.5 % / 27.5 % / 1.6 % at $T$ = 128 / 512 / 2048;
long documents (median 1 500, heavy tail) at 2048: crop 30.7 %, pad 13.3 % padding plus 10.1 % truncated.
Short uniform documents (median 60): both best-fit variants waste almost nothing at any $T$ ≥ 512.

2. `bestfit_pad` wastes about 13 % of the *compute* (padding positions do forward and backward work for no training signal); `bestfit_crop` wastes about 31 % of the *data*. In pretraining compute is scarce and data abundant, so cropping is right. In fine-tuning the curated data is scarce (and the documents are short, so padding is small), so padding is right.
3. Cropping the first (arbitrary) document instead of the shortest discards more: an arbitrary document is on average much longer than the shortest of 1 000, and everything beyond the remaining space is thrown away. Choosing the shortest minimises the discarded tail. (Expect the cropped fraction to rise by several percentage points.)

### Exercise 3: contamination

1. The memorising model's loss on the leaked half, 0.107 nats/char, is below the entropy of the language (0.314). That is impossible for a model that has not seen the text.
2. With 5 % leakage the average validation loss of the memoriser improves only slightly (about 5 % of the way from 1.02 to 0.11) and is easily mistaken for noise; the loss sanity check fails as a detector. The shingle detector still flags the leaked documents individually, because it compares each evaluation document with the training set.
3. One changed word: exact-duplicate hashing no longer matches. Shingle overlap drops only slightly (a single word change destroys at most $n$ of the document's shingles), so near-duplicate and contamination detection still fire. The memorising model's loss on those documents rises a little at the changed word and stays far below the floor elsewhere. Paraphrases that change most words defeat n-gram methods; that needs semantic similarity and is an open practical problem.

Summary: the loss sanity check needs no access to training data but only catches gross
leakage; overlap detection catches partial and near-verbatim leakage but needs the
training data and misses paraphrase.

### Exercise 4: the miniature pipeline

1. Vocabulary 353 (padded to 384 internally). Scaling lines: `Scaling LRs by 0.0625 for batch size 2,048 (reference: 524,288)`, `Scaling weight decay from 0.280000 to 0.425775 for depth 4`, `Scaling the LR for the AdamW parameters ∝1/√(64/768) = 3.464102`. Gradient accumulation: 1. Validation bpb: 1.8437, 0.5652, 0.4763, 0.4664.
2. Pocket text is ASCII, so one character is one byte, and the entropy rate in bits per character equals bits per byte. `evaluate_bpb` gives special tokens a byte length of 0 and excludes their loss, so the BOS that separates documents (whose prediction would involve the document-length distribution, which the entropy rate ignores) does not enter. Both numbers therefore measure the same thing: bits per byte of the text itself.
3. `model_config` (architecture), `user_config` (every command-line argument, including learning rates, batch sizes, steps, schedule parameters), plus the tokenizer and dataset in the same base directory, the nanochat commit, and the PyTorch version (not in the file: record them yourself).
4. Reference: 80 s vs 24 s total; 40.6 k vs 31.6 k tokens/s. For 300 steps compilation costs about a minute and saves about 3 seconds: not worth it. Break-even: the compiled run saves $2048\,(1/31.6\text{k} - 1/40.6\text{k}) \approx 14$ ms per step; 56 s of overhead is recovered after about 4 000 steps. For real runs of thousands of much longer steps, compilation pays off quickly.

### Exercise 5: a controlled data experiment

Expected pattern (your exact values will differ; see the lab table for the author's when available):

- **A** (ample data): train and validation bpb nearly equal, `epoch: 1` or 2.
- **B** (600 documents, 300 steps): slightly worse validation than A at equal steps; the loader passes through the data many times (`epoch` in the tens), but 300 steps are too few to memorise.
- **C** (600 documents, 3 000 steps): training bpb falls *below the entropy floor* (0.4528), which is the signature of memorisation, while validation bpb is no better than, or worse than, B. The `epoch` counter in the hundreds is the early warning.

The log field that warns of overfitting before any validation number: `epoch`.

### Exercise 6

There is no reference: this run has not been performed by the author on a GPU. A
complete record contains the probe JSON, the training log, parsed throughput and memory,
final validation bpb, and samples. Sanity anchors: the first logged loss should be near
$\ln 32768 = 10.4$; validation bpb should fall monotonically apart from noise; `epoch`
should remain 1.

### Challenge

Under `stream`, token positions within documents are represented in proportion to their
occurrence (every token is used). Under `bestfit_crop` with small $T$, long documents are
either skipped until a row has room or cropped to their beginnings, so late positions are
under-represented: the mean within-document position of training tokens drops sharply at
$T = 128$ and moderately at $T = 512$. The model therefore sees more beginnings than
middles and endings of long documents.

## Quiz

1. Data is higher quality if training on it yields better downstream evaluation scores per token (or per unit of compute). Inspection can remove obvious junk, but the effect of a source or filter on the trained model is not predictable by reading samples: nanochat's log lists five reasonable-looking alternatives that were worse than its baseline.
2. The validation set guides decisions during development (monitoring, comparing runs, tuning); the test set is touched once, to report final quality, and therefore remains unbiased. nanochat's last shard is a validation set.
3. (i) Contamination: the held-out text, or near-copies, is in the training data. (ii) A pipeline difference: the evaluation counts bytes or tokens differently (e.g. masks most positions, or includes long runs of trivially predictable padding/boilerplate). (iii) The held-out text is not comparable (templated or highly repetitive). A genuine fourfold improvement is the least likely explanation.
4. Tokens to read $= 1.32\text{B}/(1-0.35) = 2.03$ B; $2.03\text{B}/47\text{M} \approx 43.2$, so 44 training shards (plus the validation shard).
5. Pretraining: data is abundant and compute is the constraint, so every position should carry a real token (crop, 100 % utilisation). Fine-tuning: data is scarce and curated, conversations are short, and every one of them matters, so nothing is discarded and the small remainder of each row is padded and masked.
6. Its first tokens must be predicted without the context that determines them, which adds unavoidable loss and gradient noise, and the situation never occurs at inference (prompts start with BOS). The extra loss measures the data pipeline, not the model's ability.
7. The BOS of the second document. The model is trained to predict "a new document starts" after the first document's last token. It is not a problem: it is how the model learns document boundaries (and, later, to end its turn). In evaluation, BOS has byte length 0 and is excluded from bits per byte.
8. (a) The run is 79 % done and in its warm-down phase (multiplier 0.35). (b) Throughput is about 53 k tokens/s, i.e. $53{,}498 \times 9.8 \approx 524$ k tokens per step: the default batch at depth 12, probably with gradient accumulation given the 9.8 s step. (c) MFU 0.00 means the device is not in nanochat's table (or is a CPU): use tokens/s. (d) `epoch: 2`: the data has been exhausted once and is being repeated; too few shards were downloaded.
9. `model_<step>.pt`, `optim_<step>_rank<r>.pt`, `meta_<step>.json`. The learning rates are in the JSON under `user_config`.
10. No. The loaders produce different validation text (BOS-aligned rows contain fewer context-free positions), so the absolute numbers are not comparable; the 0.04 difference could be entirely the pipeline. Evaluate both checkpoints with the same loader and tokenizer (e.g. re-run `base_eval` on the January checkpoint), or compare on an external benchmark.
