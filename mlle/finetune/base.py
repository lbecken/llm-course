"""
The shared base model of Parts IV-V.

`get_base_model()` pretrains a small MiniGPT on the two-skill world of
`mlle.finetune.data` (Pocket sentences + addition facts as plain text) and caches the
weights under results/runs/part4/, so that the fine-tuning, RL, preference and tool labs
all start from the SAME checkpoint, as they would in a real pipeline.

The default size is chosen so that the base model is good but imperfect at addition
(about 75-80 % exact answers): later stages need room to improve it, and a tool needs
something to fix.
"""

from __future__ import annotations

import random
from dataclasses import asdict, dataclass

import torch

from mlle.data import pocket
from mlle.data.tasks import AdditionTask
from mlle.finetune.chat import ChatTokenizer
from mlle.finetune.data import base_documents, documents_to_stream, make_tokenizer
from mlle.foundations.minigpt import MiniGPT, MiniGPTConfig
from mlle.foundations.train import TrainConfig, estimate_loss, get_batch, split_tokens, train
from mlle.utils import REPO_ROOT


@dataclass
class BaseSpec:
    n_layer: int = 2
    n_embd: int = 96
    n_head: int = 4
    block_size: int = 64
    steps: int = 1500
    max_digits: int = 3
    n_sentences: int = 6000
    n_facts: int = 30000
    seed: int = 0

    def tag(self) -> str:
        return f"base-L{self.n_layer}-C{self.n_embd}-s{self.steps}-d{self.max_digits}-seed{self.seed}"


@dataclass
class World:
    """Everything a Part IV-V lab needs: tokenizer, task, and held-out text of both kinds."""
    tok: ChatTokenizer
    task: AdditionTask
    train_tokens: torch.Tensor  # pretraining stream (mixed)
    val_tokens: torch.Tensor  # held-out part of the mixed stream
    pocket_val: torch.Tensor  # held-out Pocket sentences only: the "language" skill
    spec: BaseSpec


def build_world(spec: BaseSpec | None = None) -> World:
    spec = spec or BaseSpec()
    tok = make_tokenizer()
    task = AdditionTask(max_digits=spec.max_digits, test_fraction=0.1)
    stream = documents_to_stream(tok, base_documents(spec.n_sentences, spec.n_facts, task, seed=spec.seed))
    train_tokens, val_tokens = split_tokens(stream)
    rng = random.Random(10_000 + spec.seed)  # sentences sampled independently of the training documents
    pocket_val = documents_to_stream(tok, [pocket.sample_sentence(rng).text.strip() for _ in range(1500)])
    return World(tok, task, train_tokens, val_tokens, pocket_val, spec)


def new_model(world: World, seed: int | None = None) -> MiniGPT:
    s = world.spec
    torch.manual_seed(s.seed if seed is None else seed)
    return MiniGPT(MiniGPTConfig.nanochat_style(world.tok.vocab_size, block_size=s.block_size, n_layer=s.n_layer,
                                                n_head=s.n_head, n_embd=s.n_embd))


def get_base_model(spec: BaseSpec | None = None, cache_dir=None, verbose: bool = True) -> tuple[MiniGPT, World]:
    """Return (pretrained model, world). Trains on first use (about 45 s on a 4-core CPU
    for the default spec) and loads the cached weights afterwards."""
    spec = spec or BaseSpec()
    world = build_world(spec)
    model = new_model(world)
    cache_dir = REPO_ROOT / "results" / "runs" / "part4" if cache_dir is None else cache_dir
    path = cache_dir / f"{spec.tag()}.pt"
    if path.exists():
        saved = torch.load(path, map_location="cpu")
        if saved.get("spec") == asdict(spec):
            model.load_state_dict(saved["state_dict"])
            if verbose:
                print(f"loaded cached base model {path.relative_to(REPO_ROOT) if path.is_relative_to(REPO_ROOT) else path}")
            return model.eval(), world
    if verbose:
        print(f"pretraining base model {spec.tag()} ({model.num_params():,} parameters, {spec.steps} steps) ...")
    train(model, world.train_tokens, world.val_tokens,
          TrainConfig(steps=spec.steps, batch_size=32, lr=3e-3, eval_every=max(1, spec.steps // 3), seed=spec.seed), verbose=verbose)
    cache_dir.mkdir(parents=True, exist_ok=True)
    torch.save({"spec": asdict(spec), "state_dict": model.state_dict()}, path)
    return model.eval(), world


def pocket_loss(model: MiniGPT, world: World) -> float:
    """Mean loss (nats per character) on held-out Pocket sentences: how well the model
    still does what pretraining taught it. Used to measure forgetting."""
    return estimate_loss(model, world.pocket_val, 32, 10)


def pretraining_batch_fn(world: World, batch_size: int = 32, seed: int = 0):
    """A function returning fresh batches of pretraining data (for replay during fine-tuning)."""
    g = torch.Generator().manual_seed(seed)
    return lambda: get_batch(world.train_tokens, batch_size, world.spec.block_size, g)


def get_chat_model(base: MiniGPT, world: World, steps: int = 300, lr: float = 1e-3, n_convs: int = 3000,
                   long_fraction: float = 0.0, cache_dir=None, verbose: bool = True) -> MiniGPT:
    """The base model after supervised fine-tuning on chat-formatted addition (with replay
    of pretraining data, see Module 14). Cached like the base model. The defaults give a
    deliberately UNDER-trained assistant (about 50 % exact answers), so that the methods of
    Module 15 have room to show an effect."""
    import copy

    from mlle.finetune.data import conversations
    from mlle.finetune.sft import build_sft_rows, sft_train

    cache_dir = REPO_ROOT / "results" / "runs" / "part4" if cache_dir is None else cache_dir
    path = cache_dir / f"chat-{world.spec.tag()}-s{steps}-lr{lr:g}-n{n_convs}-long{long_fraction:g}.pt"
    model = copy.deepcopy(base)
    if path.exists():
        model.load_state_dict(torch.load(path, map_location="cpu"))
        return model.eval()
    if verbose:
        print(f"fine-tuning the chat model ({steps} SFT steps with replay) ...")
    x, y = build_sft_rows(world.tok, conversations(world.task, n_convs, "train", 7, long_fraction=long_fraction), world.spec.block_size)
    sft_train(model, x, y, steps=steps, lr=lr, extra_batch_fn=pretraining_batch_fn(world))
    cache_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), path)
    return model.eval()
