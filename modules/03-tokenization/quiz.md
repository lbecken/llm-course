# Module 03 — Knowledge check

Closed book, about 20 minutes. Solutions: [`solutions/03-tokenization.md`](../../solutions/03-tokenization.md).

1. **(Concept)** Give one advantage and one disadvantage of character-level tokens compared with subword tokens.

2. **(Algorithm)** In BPE training, what is merged at each step, and what data structure constitutes the trained tokenizer?

3. **(Calculation)** A tokenizer has 32 768 entries including 9 special tokens. How many merges were learned? If the model width is 768, how many parameters does the token embedding table have?

4. **(Concept)** Why can a byte-level BPE tokenizer encode a word it has never seen? What does it cost?

5. **(Code reading)** nanochat calls `self.enc.encode_ordinary(text)` rather than `self.enc.encode(text, allowed_special="all")`. What would change with the second form, and why does it matter?

6. **(Concept)** Model A (vocabulary 8 000) has validation loss 2.9 nats/token; model B (vocabulary 64 000) has 3.4 nats/token on the same text. Which is better? What additional information do you need, and what metric resolves it?

7. **(Applied)** With nanochat's split pattern, how is `1234567` pre-split? Why can this make arithmetic harder for a model?

8. **(Concept)** Why is `token_bytes` for a special token set to 0 in `scripts/tok_train.py`?

9. **(Experimental reasoning)** A tokenizer trained on English news is used for a model that will mostly process Python code. Predict two measurable consequences.

10. **(Synthesis)** In the rendered conversation, the tokens between `<|output_start|>` and `<|output_end|>` have mask 0 while `<|python_start|>…<|python_end|>` have mask 1. Explain the difference in terms of who produces those tokens at inference time.

## Mastery criteria

- [ ] Run three BPE merges by hand and encode a new string with them.
- [ ] Explain leading-space tokens and digit grouping and one behavioural consequence of each.
- [ ] Explain why special tokens cannot be forged by user text, and what that does and does not protect against.
- [ ] Compute a compression ratio and convert a per-token loss into bits per byte.
- [ ] At least 8 of 10 quiz questions correct.
