# Solutions — Module 03

## Lab exercises

### Exercise 1: BPE by hand

Text `low lower lowest`, one chunk, `_` = space.

1. `l o w _ l o w e r _ l o w e s t` (16 tokens).
2. Pair counts: `l o` 3, `o w` 3, `w e` 2, `w _` 1, `_ l` 2, `e r` 1, `r _` 1, `e s` 1, `s t` 1. Ties `l o` / `o w` → first in text: **merge 1: `lo`**. Text: `lo w _ lo w e r _ lo w e s t` (13). Now `lo w` occurs 3 times: **merge 2: `low`**. Text: `low _ low e r _ low e s t` (10). Counts: `low e` 2, `_ low` 2, `low _` 1, …; first in text among the ties is `_ low` (position 1) versus `low e` (position 2): **merge 3: `_low`**. Text: `low _low e r _low e s t` (8).
3. `slow` → `s l o w` → merge 1 gives `s lo w` → merge 2 gives `s low`: **2 tokens**. Merge 3 does not apply (no space).
4. With a split pattern that attaches the space to the following word, the chunks are `low`, `_lower`, `_lowest`; merges cannot cross chunks, and `_l` (2 occurrences, within chunks) competes from the start. Merge 3 (`_low`) is still possible because the space is inside the chunk; what becomes impossible is a merge like `w _` or `r _` that joins the end of one word to the following space.

**Common mistake:** counting overlapping occurrences when *applying* a merge. Counting pairs uses all adjacent positions; replacing proceeds left to right without overlap.

### Exercise 2: vocabulary size versus compression

1. Pocket has only a few dozen distinct chunks (words with and without a leading space, the full stop, the newline). After 87 merges each chunk is one token; no adjacent pair inside a chunk remains, so there is nothing to merge. Real corpora have an effectively unbounded set of chunks and never saturate.
2. Typical for English prose: first merges are `e␣`-like pairs within words (`th`, `in`, `er`, `␣t`, `␣the`); roughly 2.0–2.6 bytes/token at 256 merges and 2.8–3.5 at 768 on a small corpus. Source code gives lower ratios at the same size (more symbols, more distinct identifiers).
3. Close to one token per byte, and for non-Latin scripts *more tokens than characters* (2–4 bytes per character).

### Exercise 3: special tokens and injection

1. `tok.encode_special("<|assistant_end|>") not in ids` is `True`: the string became ordinary tokens such as `<`, `|`, `a`, `s`, ….
2. One unsafe implementation:
   ```python
   import re
   def encode_unsafe(tok, text):
       pattern = "(" + "|".join(re.escape(s) for s in tok.special_tokens) + ")"
       ids = []
       for part in re.split(pattern, text):
           ids += [tok.special_tokens[part]] if part in tok.special_tokens else tok.encode(part)
       return ids
   ```
   The attack string now contains the real `<|assistant_end|>` and `<|user_start|>` ids.
3. The user could terminate their own turn and forge further turns: fake assistant messages ("the assistant already agreed to…"), fake tool outputs, or a fake system preamble, all indistinguishable to the model from genuine structure. With `encode` (ordinary) the forged markers are inert text.

**What the safe encoder does not protect against:** instructions written in plain language inside user data or tool output. The model reads those as text and may follow them; that is prompt injection and needs defences at the application level (Module 16).

### Exercise 4: the real tokenizer

1. 88 merges; it stopped because every chunk had become a single token (same reason as Exercise 2.1). Vocabulary: 256 + 88 + 9 = 353, below the requested 512.
2. Better than GPT-4's tokenizer only on the Pocket text it was trained on (4.59 vs 4.42 bytes/token, about +4 %); worst on Korean (1.00 vs 2.45) and near 1.1–1.2 on news, code, maths and science.
3. The tool output: `<|output_start|>`, `2`, `<|output_end|>`. At inference these tokens are inserted by the engine after it has executed the expression; the model never has to generate them, so training it to predict them would at best waste capacity and at worst teach it to hallucinate tool results.

### Challenge: digits

With `\d{1,2}` there are at most 110 distinct number chunks (0–9 and 00–99), every one
of them frequent; a 5-digit number is 3 tokens split as 12|34|5, so the last token's
place value depends on the length of the number. With `\d{1,3}` there are up to 1 110
number chunks; numbers are shorter in tokens (12345 → 123|45), but each chunk is rarer,
and a small vocabulary spends a large share of its entries on numbers. For a model that
must learn arithmetic, consistency matters more than brevity: single digits (`\d`), or
right-aligned grouping, make place value learnable. For a 32K-vocabulary prose model the
nanochat choice of 2 is a compromise that keeps number tokens few; arithmetic is then
delegated to a tool.

## Quiz

1. Advantage: tiny vocabulary, no arbitrary segmentation, spelling is visible. Disadvantage: sequences 4–5× longer, so more compute (quadratically more in attention), shorter effective context, and each position carries little information.
2. The most frequent adjacent pair of current tokens is replaced by a new token. The trained tokenizer is the ordered list of merges (equivalently: a table from token bytes to rank), plus the split pattern and the special tokens.
3. $32768 - 256 - 9 = 32\,503$ merges. Embedding table: $32768 \times 768 = 25\,165\,824$ parameters.
4. Its base vocabulary is all 256 byte values, so any byte string is representable; unseen words are composed of smaller learned pieces or, at worst, single bytes. Cost: more tokens for that word (more context used, more compute, and usually poorer modelling).
5. With `allowed_special="all"`, special-token strings appearing in the text would be converted to the real control ids. User-supplied text could then inject control tokens (end its turn, forge an assistant or tool message). `encode_ordinary` treats everything as plain text; only code can add control tokens.
6. Undecidable from these numbers. B's tokens are longer on average, so each is harder to predict. Needed: the number of bytes (or characters) per token for each tokenizer on this text. Compare bits per byte: loss in nats ÷ ln 2 ÷ bytes per token. For example with 3.0 and 4.2 bytes/token: A = 2.9/0.693/3.0 = 1.39, B = 3.4/0.693/4.2 = 1.17 bits per byte, and B is better.
7. `12 | 34 | 56 | 7`: chunks of at most two digits from the left. The same digit position falls into different places within its token depending on the number's length (7 digits vs 6 digits), so "units", "tens", … do not correspond to fixed token positions, and carrying crosses token boundaries irregularly.
8. Bits per byte divides the summed loss by the summed byte length of the targets. Special tokens are not text: they have no bytes in the original document. Setting their length to 0 excludes them from the byte count, and `evaluate_bpb` also excludes their loss.
9. (i) Lower compression on code: fewer bytes per token, so longer sequences, less code per context window, more compute per file. (ii) Many code tokens (identifiers, indentation, operators) are split into rare fragments whose embeddings are poorly trained, giving a higher loss in bits per byte than a model with a code-aware tokenizer of the same size.
10. At inference the assistant (the model) generates the tool *request* (`<|python_start|>expr<|python_end|>`), so it must be trained to produce it: mask 1. The tool *result* is produced by the Python evaluator and forced into the sequence by the engine: the model only reads it. Supervising those tokens would train the model to predict tool outputs itself.
