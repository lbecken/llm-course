# Module 21 — Further study

| Resource | Type | Level | What it contributes |
|---|---|---|---|
| nanochat, [`runs/speedrun.sh`](https://github.com/karpathy/nanochat/blob/92d63d4e8bb4df75c3b71618f31ddde2378b2bcd/runs/speedrun.sh), [`runs/runcpu.sh`](https://github.com/karpathy/nanochat/blob/92d63d4e8bb4df75c3b71618f31ddde2378b2bcd/runs/runcpu.sh) and [`nanochat/checkpoint_manager.py`](https://github.com/karpathy/nanochat/blob/92d63d4e8bb4df75c3b71618f31ddde2378b2bcd/nanochat/checkpoint_manager.py) | source code | intermediate | The primary source for section F: a complete pipeline as one script, and the checkpoint metadata that connects its stages. |
| Sculley et al., [*Hidden Technical Debt in Machine Learning Systems*](https://papers.nips.cc/paper_files/paper/2015/hash/86df7dcfd896fcaf2674f757a2463eba-Abstract.html), NeurIPS 2015 | paper | introductory | The classic account of why the model is a small part of an ML system and where the system-level failures come from (entanglement, hidden feedback loops, undeclared consumers). D.1's "silent failure" column in general form. |
| Pineau et al., [*Improving Reproducibility in Machine Learning Research*](https://arxiv.org/abs/2003.12206), JMLR 22, 2021 | paper | introductory | The reproducibility checklist used by major conferences and the experience of introducing it. A good template for Exercise 4's run card. |
| Mitchell et al., [*Model Cards for Model Reporting*](https://arxiv.org/abs/1810.03993), FAT* 2019 | paper | introductory | What to document about a trained model for those who will use it: intended use, evaluation data, metrics, limitations. The outward-facing counterpart of a manifest. |
| Gebru et al., [*Datasheets for Datasets*](https://arxiv.org/abs/1803.09010), Communications of the ACM 64(12), 2021 | paper | introductory | The same idea for data: provenance, composition, collection, recommended uses. Relevant to the first artefact of D.1. |
| Beyer, Jones, Petoff, Murphy (eds.), [*Site Reliability Engineering*](https://sre.google/sre-book/table-of-contents/) (2016), chapters 6 and 12 | book (free online) | intermediate | Monitoring and effective troubleshooting for running systems: symptoms versus causes, and working from the symptom upstream. D.4 and D.5 apply the same method to model systems. |
| Bouthillier et al., [*Accounting for Variance in Machine Learning Benchmarks*](https://arxiv.org/abs/2103.03098), MLSys 2021 | paper | advanced | Which sources of randomness matter for a reported result and how many repetitions are needed: the quantitative basis for "replicable within noise". |

Historical versus current: these are systems and methodology references and age slowly.
Tooling for experiment tracking and serving changes quickly; the fields worth recording
do not.
