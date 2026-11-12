# Module 03 — Further study

| Resource | Type | Level | What it contributes |
|---|---|---|---|
| Andrej Karpathy, [*Let's build the GPT Tokenizer*](https://www.youtube.com/watch?v=zduSFxRajkE) (2024) and the companion repository [minbpe](https://github.com/karpathy/minbpe) | video 2 h + code | introductory | Builds byte-level BPE from scratch, including the regex split and special tokens, and explains many tokenizer-induced oddities of LLMs. The closest companion to this module. |
| Sennrich, Haddow, Birch, [*Neural Machine Translation of Rare Words with Subword Units*](https://arxiv.org/abs/1508.07909), ACL 2016 | paper | intermediate | Introduced BPE for neural text models. Section 3 is the algorithm in a dozen lines of Python. |
| OpenAI, [tiktoken](https://github.com/openai/tiktoken) | code + docs | reference | The encoder nanochat uses at inference. The README explains `encode_ordinary` versus `encode` with `allowed_special`. |
| Karpathy, [rustbpe](https://github.com/karpathy/rustbpe) | code | reference | The Rust trainer nanochat uses; small enough to read. |
| Kudo & Richardson, [*SentencePiece: A simple and language independent subword tokenizer and detokenizer for Neural Text Processing*](https://arxiv.org/abs/1808.06226), EMNLP 2018 | paper | intermediate | The main alternative family (unigram LM tokenization, whitespace as an ordinary symbol), used by many non-OpenAI models. Useful contrast with the regex-split approach. |
| Hugging Face, [*Summary of the tokenizers*](https://huggingface.co/docs/transformers/tokenizer_summary) | documentation | introductory | Side-by-side description of BPE, WordPiece and Unigram with examples. |
| nanochat, [Guide: counting r in strawberry (and how to add abilities generally)](https://github.com/karpathy/nanochat/discussions/164) | discussion | intermediate | A worked case of a tokenization-induced weakness and how to address it with synthetic training data. Written against an older revision of nanochat; the idea carries over, file names may not. |

Historical versus current: the 2016 BPE paper used characters and an end-of-word marker;
modern LLM tokenizers are byte-level with a regex pre-split, which is what this module
teaches. Vocabulary sizes have grown from ~30–50K (2019) to 100–250K in recent large
models; nanochat's 32K is a deliberate choice for small models.
