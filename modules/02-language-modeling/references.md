# Module 02 — Further study

| Resource | Type | Level | What it contributes |
|---|---|---|---|
| Andrej Karpathy, [*The spelled-out intro to language modeling: building makemore*](https://www.youtube.com/watch?v=PaCmpygFfXo) (2022) | video, 2 h | introductory | Bigram character model by counting, then the same model as a one-layer neural network trained by gradient descent. Shows that the two reach the same loss, a perfect companion to this module. |
| Jurafsky & Martin, [*Speech and Language Processing*](https://web.stanford.edu/~jurafsky/slp3/) (3rd ed. draft), ch. 3 "N-gram Language Models" | book (free online) | introductory–intermediate | The standard textbook treatment: n-grams, smoothing, back-off, perplexity, and their limits. |
| Claude Shannon, [*A Mathematical Theory of Communication*](https://people.math.harvard.edu/~ctm/home/text/others/shannon/entropy/entropy.pdf) (1948), sections 1–7 | paper | intermediate | Defines entropy and, in section 3, generates English with character and word n-grams of increasing order: the original version of this module's sample table. |
| Claude Shannon, *Prediction and Entropy of Printed English*, Bell System Technical Journal 30 (1951) | paper | intermediate | Estimates the entropy of English by having humans predict the next letter (roughly 0.6–1.3 bits per character). The human experiment behind "the loss has a floor". |
| David MacKay, [*Information Theory, Inference, and Learning Algorithms*](https://www.inference.org.uk/itila/book.html), ch. 2, 4–6 (2003) | book (free online) | intermediate | Entropy, the source-coding theorem and arithmetic coding: why a predictor is a compressor. |
| Linzen, Dupoux, Goldberg, [*Assessing the Ability of LSTMs to Learn Syntax-Sensitive Dependencies*](https://arxiv.org/abs/1611.01368), TACL 2016 | paper | intermediate | Origin of the subject–verb agreement test with attractors used throughout this course, applied to recurrent networks on real text. |
| Bengio, Vinyals, Jaitly, Shazeer, [*Scheduled Sampling for Sequence Prediction with Recurrent Neural Networks*](https://arxiv.org/abs/1506.03099), NeurIPS 2015 | paper | intermediate | A clear statement of the train/generate mismatch (exposure bias) and one early attempt to reduce it. Historical; not used in modern LLM pretraining. |

Historical versus current: n-gram models are no longer competitive as language models,
but they remain the right mental baseline and are still used inside evaluation metrics
and data-deduplication tools (Module 08).
