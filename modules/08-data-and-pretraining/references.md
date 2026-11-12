# Module 08 — Further study

| Resource | Type | Level | What it contributes |
|---|---|---|---|
| nanochat, [`dev/LOG.md`](https://github.com/karpathy/nanochat/blob/92d63d4e8bb4df75c3b71618f31ddde2378b2bcd/dev/LOG.md): entries 2026-03-04 (dataset upgrade), 2026-02-17 (data experiments, negative), 2026-01-13 (BOS-aligned loader and bin packing) | engineering log | intermediate | Primary source for this module: real measurements of how data choice and packing changed results, including five failed dataset attempts. |
| nanochat, [`dev/repackage_data_reference.py`](https://github.com/karpathy/nanochat/blob/92d63d4e8bb4df75c3b71618f31ddde2378b2bcd/dev/repackage_data_reference.py) | code | introductory | How the hosted shards were made: shuffle, 100 MB zstd-compressed parquet files, row groups of 1 000. |
| Li et al., [*DataComp-LM: In search of the next generation of training sets for language models*](https://arxiv.org/abs/2406.11794) (DCLM, 2024) | paper | intermediate | A controlled benchmark for *data curation*: fixed model and training recipe, varying only the data. Shows how much filtering matters and defines the CORE metric used by nanochat. |
| Penedo et al., [*The FineWeb Datasets: Decanting the Web for the Finest Text Data at Scale*](https://arxiv.org/abs/2406.17557) (2024) | paper | intermediate | A transparent account of building a web pretraining set: extraction, filtering, deduplication, and ablations of each step. FineWeb-EDU was nanochat's dataset before ClimbMix. |
| NVIDIA, [Nemotron-ClimbMix](https://huggingface.co/datasets/nvidia/Nemotron-ClimbMix) dataset card; Diao et al., [*CLIMB: CLustering-based Iterative Data Mixture Bootstrapping for Language Model Pre-training*](https://arxiv.org/abs/2504.13161) (2025) | dataset + paper | advanced | The source of nanochat's current data and the method used to choose its mixture. |
| Broder, [*On the resemblance and containment of documents*](https://www.cs.princeton.edu/courses/archive/spring13/cos598C/broder97resemblance.pdf) (1997) | paper | intermediate | Shingling and MinHash: the classic near-duplicate detection method behind the toy version in this module. |
| Carlini et al., [*Extracting Training Data from Large Language Models*](https://arxiv.org/abs/2012.07805), USENIX Security 2021 | paper | intermediate | Demonstrates that language models memorise and can emit verbatim training text: why duplicates and private data in a corpus matter. |
| Andrej Karpathy, [*A Recipe for Training Neural Networks*](https://karpathy.github.io/2019/04/25/recipe/) (2019), sections 1–2 | blog post | introductory | "Become one with the data" and "set up the end-to-end skeleton": the habits behind this module's diagnosis table. |

Historical versus current: the specific datasets named here (FineWeb-EDU, ClimbMix) are
the current choices of one project and will be superseded. The methods (filter,
deduplicate, hold out, verify against a fixed evaluation) are stable.
