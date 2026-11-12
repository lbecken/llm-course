"""Part IV: chat rendering, loss masks, SFT, LoRA, REINFORCE and DPO."""

import copy

import pytest
import torch

from mlle.data.tasks import AdditionProblem, AdditionTask
from mlle.finetune.base import BaseSpec, build_world, get_base_model, new_model, pocket_loss
from mlle.finetune.chat import ChatTokenizer
from mlle.finetune.data import ALPHABET, LONG_PREFIX, addition_conversation, conversations, extract_answer, make_tokenizer
from mlle.finetune.dpo import dpo_loss, dpo_train, make_preference_pairs, make_style_pairs, mean_logprobs, sequence_logprob
from mlle.finetune.lora import LoRALinear, add_lora, count_trainable, set_lora_enabled
from mlle.finetune.rl import reinforce_step, reward_exact, reward_last_digit, rollout_batch
from mlle.finetune.sft import build_sft_rows, chat_accuracy, complete, sft_train
from mlle.foundations.bpe import SPECIAL_TOKENS

TINY = BaseSpec(n_layer=1, n_embd=32, n_head=2, steps=30, max_digits=2, n_sentences=200, n_facts=400)


@pytest.fixture(scope="module")
def world():
    return build_world(TINY)


@pytest.fixture()
def model(world):
    m = new_model(world)
    with torch.no_grad():
        for p in m.parameters():  # zero-initialised projections would make every test trivial
            if p.ndim == 2:
                torch.nn.init.normal_(p, std=0.1)
    return m


# ----------------------------------------------------------------------------- chat format
def test_tokenizer_has_nanochats_special_tokens_after_the_alphabet():
    tok = make_tokenizer()
    assert tok.vocab_size == len(set(ALPHABET)) + len(SPECIAL_TOKENS) == len(set(ALPHABET)) + 9
    assert tok.decode(tok.encode("47+38=85")) == "47+38=85"
    assert tok.encode_special("<|bos|>") == tok.get_bos_token_id() >= len(set(ALPHABET))


def test_render_conversation_structure_and_mask():
    tok = make_tokenizer()
    conv = addition_conversation(AdditionProblem(47, 38))
    ids, mask = tok.render_conversation(conv)
    names = [tok.itos[i] for i in ids]
    assert names == ["<|bos|>", "<|user_start|>", "4", "7", "+", "3", "8", "<|user_end|>", "<|assistant_start|>", "8", "5", "<|assistant_end|>"]
    assert mask == [0] * 9 + [1, 1, 1]  # only the assistant's text and its end token are supervised
    assert tok.render_for_completion(conv) == ids[:9]


def test_tool_call_parts_supervise_the_request_but_not_the_tool_output():
    tok = make_tokenizer()
    ids, mask = tok.render_conversation(addition_conversation(AdditionProblem(2, 3), use_tool=True))
    by_name = [(tok.itos[i], m) for i, m in zip(ids, mask)]
    assert ("<|python_start|>", 1) in by_name and ("<|python_end|>", 1) in by_name
    assert ("<|output_start|>", 0) in by_name and ("<|output_end|>", 0) in by_name
    out_start = [n for n, _ in by_name].index("<|output_start|>")
    assert by_name[out_start + 1] == ("5", 0)  # the calculator's result is read, not produced
    assert by_name[-2:] == [("5", 1), ("<|assistant_end|>", 1)]  # the final answer is produced


def test_roles_must_alternate():
    tok = make_tokenizer()
    with pytest.raises(AssertionError):
        tok.render_conversation({"messages": [{"role": "assistant", "content": "hi"}]})


def test_answer_styles_and_extraction():
    task = AdditionTask(max_digits=2)
    convs = conversations(task, 200, "train", 0, long_fraction=0.5)
    long = [c for c in convs if c["messages"][1]["content"].startswith(LONG_PREFIX)]
    assert 60 < len(long) < 140
    assert all(extract_answer(c["messages"][1]["content"]) == c["problem"].answer for c in convs)


# ----------------------------------------------------------------------------- SFT
def test_sft_rows_mask_everything_but_assistant_tokens(world):
    tok = world.tok
    convs = conversations(world.task, 40, "train", 0)
    x, y = build_sft_rows(tok, convs, 64)
    assert x.shape == y.shape and x.shape[1] == 63
    supervised = y[y >= 0]
    allowed = set(tok.encode("0123456789")) | {tok.encode_special("<|assistant_end|>")}
    assert set(supervised.tolist()) <= allowed  # never a user token, a structure token or padding
    n_answer_tokens = sum(len(c["problem"].answer) + 1 for c in convs)
    assert len(supervised) == n_answer_tokens  # every conversation was packed whole, none cropped


def test_sft_teaches_the_format_and_complete_stops_at_the_end_token(world, model):
    tok = world.tok
    convs = conversations(world.task, 4, "train", 0)  # memorise 4 conversations
    x, y = build_sft_rows(tok, convs, 64)
    assert chat_accuracy(model, tok, convs) == 0.0  # an untrained model does not answer in the format
    losses = sft_train(model, x, y, steps=300, batch_size=4, lr=3e-3)
    assert losses[-1] < 0.1 < losses[0]
    assert chat_accuracy(model, tok, convs) >= 0.75
    stop = {tok.encode_special("<|assistant_end|>")}
    outs = [tok.decode(complete(model, tok.render_for_completion(c), stop, max_new_tokens=12)[0]) for c in convs]
    assert any(o == c["problem"].answer for o, c in zip(outs, convs))  # stopped at the end token: nothing after the answer
    assert all(len(o) <= 4 for o in outs)


def test_complete_batches_samples_from_one_prefill(world, model):
    tok = world.tok
    prompt = tok.render_for_completion(conversations(world.task, 1, "train", 0)[0])
    g = torch.Generator().manual_seed(0)
    outs = complete(model, prompt, set(), max_new_tokens=6, temperature=1.0, num_samples=16, generator=g)
    assert len(outs) == 16 and all(len(o) == 6 for o in outs)
    assert len({tuple(o) for o in outs}) > 1  # rows are sampled independently
    greedy = complete(model, prompt, set(), max_new_tokens=6)
    assert greedy == complete(model, prompt, set(), max_new_tokens=6)


# ----------------------------------------------------------------------------- LoRA
def test_lora_starts_as_identity_freezes_base_and_can_be_switched_off(world, model):
    tok = world.tok
    idx = torch.randint(0, tok.vocab_size, (2, 16))
    before = model(idx)
    base_state = {k: v.clone() for k, v in model.state_dict().items()}
    chat_tokens = sorted(i for n, i in tok.special.items() if n != "<|bos|>")
    params = add_lora(model, r=4, alpha=8, targets=("c_q", "c_v", "c_fc"), new_token_ids=chat_tokens)
    assert torch.allclose(model(idx), before, atol=1e-6)  # B = 0 and row corrections = 0 at the start
    assert count_trainable(model) == sum(p.numel() for p in params) < 0.5 * sum(v.numel() for v in base_state.values())

    x, y = build_sft_rows(tok, conversations(world.task, 16, "train", 0), 64)
    sft_train(model, x, y, steps=30, batch_size=8, lr=3e-3, params=params)
    assert not torch.allclose(model(idx), before, atol=1e-4)  # the adapter learned something
    set_lora_enabled(model, False)
    assert torch.allclose(model(idx), before, atol=1e-6)  # ... and the base model is untouched underneath
    set_lora_enabled(model, True)
    layer = next(m for m in model.modules() if isinstance(m, LoRALinear))
    probe = torch.randn(3, layer.base.in_features)
    assert torch.allclose(layer(probe), probe @ layer.merged_weight().T, atol=1e-5)


# ----------------------------------------------------------------------------- RL
def test_rewards():
    p = AdditionProblem(47, 38)  # 85
    assert reward_exact(p, "85") == 1.0 and reward_exact(p, "86") == 0.0
    assert reward_last_digit(p, "15") == 1.0  # wrong, but the lazy checker is satisfied
    assert reward_last_digit(p, "5") == 0.0 and reward_last_digit(p, "86") == 0.0


def test_rollout_targets_cover_exactly_the_sampled_tokens(world, model):
    tok = world.tok
    conv = conversations(world.task, 1, "train", 0)[0]
    g = torch.Generator().manual_seed(0)
    inputs, targets, rewards = rollout_batch(model, tok, conv, 6, reward_exact, g)
    prompt_len = len(tok.render_for_completion(conv))
    assert inputs.shape == targets.shape and rewards.shape == (6,)
    assert (targets[:, : prompt_len - 1] == -1).all()  # nothing in the prompt is an action
    assert (targets[:, prompt_len - 1] >= 0).all()  # the first sampled token always is


def test_reinforce_raises_the_probability_of_rewarded_samples(world, model):
    """One prompt, reward 1 for answers starting with '7'. The probability of '7' as first token must rise."""
    tok = world.tok
    conv = conversations(world.task, 1, "train", 0)[0]
    prompt = torch.tensor([tok.render_for_completion(conv)])
    seven = tok.encode("7")[0]
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3)
    g = torch.Generator().manual_seed(0)

    def p_seven():
        with torch.no_grad():
            return torch.softmax(model(prompt)[0, -1], -1)[seven].item()

    before = p_seven()
    for _ in range(30):
        batch = rollout_batch(model, tok, conv, 16, lambda problem, text: float(text.startswith("7")), g)
        stats = reinforce_step(model, opt, [batch])
    assert p_seven() > before + 0.2
    assert 0.0 <= stats["informative_prompts"] <= 1.0


def test_no_gradient_when_all_samples_get_the_same_reward(world, model):
    tok = world.tok
    conv = conversations(world.task, 1, "train", 0)[0]
    batch = rollout_batch(model, tok, conv, 8, lambda problem, text: 1.0, torch.Generator().manual_seed(0))
    before = [p.clone() for p in model.parameters()]
    stats = reinforce_step(model, torch.optim.SGD(model.parameters(), lr=1.0), [batch])
    assert stats["informative_prompts"] == 0.0
    assert all(torch.equal(a, b) for a, b in zip(before, model.parameters()))  # advantage = 0 everywhere


# ----------------------------------------------------------------------------- DPO
def test_sequence_logprob_matches_a_manual_computation(world, model):
    tok = world.tok
    prompt, answer = tok.encode("12+3="), tok.encode("15")
    with torch.no_grad():
        logits = model(torch.tensor([prompt + answer]))[0]
        logp = torch.log_softmax(logits, -1)
        manual = logp[len(prompt) - 1, answer[0]] + logp[len(prompt), answer[1]]
        assert torch.allclose(sequence_logprob(model, prompt, answer), manual, atol=1e-5)


def test_dpo_loss_is_log2_at_the_reference_and_training_widens_the_margin(world, model):
    tok = world.tok
    convs = conversations(world.task, 12, "train", 0)
    pairs = make_preference_pairs(tok, convs)
    assert all(w != l and w[-1] == l[-1] == tok.encode_special("<|assistant_end|>") for _, w, l in pairs)
    reference = copy.deepcopy(model)
    assert abs(dpo_loss(model, reference, *pairs[0]).item() - 0.6931) < 1e-3  # margin 0 => -log sigmoid(0)
    w0, l0 = mean_logprobs(model, pairs)
    losses = dpo_train(model, pairs, steps=40, batch_size=8, lr=3e-3)
    w1, l1 = mean_logprobs(model, pairs)
    assert losses[-1] < 0.5 < losses[0] + 0.2
    assert (w1 - l1) > (w0 - l0) + 1.0  # the DIFFERENCE grew; nothing is promised about w1 itself


def test_style_pairs_differ_only_in_style(world):
    tok = world.tok
    convs = conversations(world.task, 5, "train", 0)
    for (prompt, preferred, rejected), c in zip(make_style_pairs(tok, convs, prefer="long"), convs):
        assert tok.decode(preferred[:-1]) == LONG_PREFIX + c["problem"].answer
        assert tok.decode(rejected[:-1]) == c["problem"].answer


# ----------------------------------------------------------------------------- the shared base model
@pytest.mark.slow
def test_base_model_is_cached_and_learns_something(tmp_path):
    spec = BaseSpec(n_layer=1, n_embd=32, n_head=2, steps=120, max_digits=2, n_sentences=500, n_facts=1500)
    model, world = get_base_model(spec, cache_dir=tmp_path, verbose=False)
    assert pocket_loss(model, world) < 2.0  # far below the uniform loss of ln(56) = 4.0
    again, _ = get_base_model(spec, cache_dir=tmp_path, verbose=False)
    assert all(torch.equal(a, b) for a, b in zip(model.parameters(), again.parameters()))
