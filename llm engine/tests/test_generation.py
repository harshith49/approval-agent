"""Boundary checks for uncached decoding and the user-facing CLI."""
import sys

import pytest
import torch

from engine.config import ModelConfig
from engine.model import GPT2Model
from engine.generate import generate
from engine.sampler import greedy
from engine.weights import resolve_device


@pytest.fixture
def model():
    torch.manual_seed(21)
    return GPT2Model(ModelConfig(vocab_size=37, max_positions=16, hidden_size=24,
                               num_layers=2, num_heads=4, intermediate_size=96)).eval()


def test_greedy_picks_largest_logit():
    assert greedy(torch.tensor([[1., 7., 2.], [8., -1., 3.]])).tolist() == [1, 0]


def test_zero_tokens_returns_prompt(model):
    ids = torch.tensor([[1, 2]])
    assert torch.equal(generate(model, ids, 0), ids)


@pytest.mark.parametrize('ids,count', [(torch.tensor([[1]]), -1),
    (torch.tensor([[1]]), 16), (torch.tensor([[1]]), 1.5),
    (torch.tensor([[1], [2]]), 1), (torch.tensor([1]), 0),
    (torch.tensor([[37]]), 0), (torch.tensor([[1.0]]), 0),
    (torch.empty((1, 0), dtype=torch.long), 0)])
def test_invalid_generation_request(model, ids, count):
    with pytest.raises(ValueError):
        generate(model, ids, count)


def test_exact_context_budget(model):
    assert generate(model, torch.tensor([[1]]), 15).shape == (1, 16)


def test_eos_is_included_and_stops_generation(model):
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()
    # Zero logits deterministically choose ID 0; no fake decoding loop needed.
    assert generate(model, torch.tensor([[2]]), 5, eos_token_id=0).tolist() == [[2, 0]]
    assert generate(model, torch.tensor([[2]]), 5).tolist() == [[2, 0, 0, 0, 0, 0]]


def test_device_selection(monkeypatch):
    monkeypatch.setattr(torch.cuda, 'is_available', lambda: False)
    assert resolve_device('auto') == torch.device('cpu')
    assert resolve_device('cpu') == torch.device('cpu')
    with pytest.raises(ValueError, match='CUDA'):
        resolve_device('cuda')
    with pytest.raises(ValueError, match='device'):
        resolve_device('mps')


def test_cli_empty_prompt_uses_eos_seed(model, monkeypatch, capsys):
    from engine import generate as cli

    class Tokenizer:
        eos_token_id = 2

        def decode(self, ids, *, skip_special_tokens):
            assert skip_special_tokens is True
            return 'seed=' + ','.join(str(i) for i in ids)

    monkeypatch.setattr(cli, 'load_model', lambda config: (model, Tokenizer()))
    monkeypatch.setattr(sys, 'argv', ['mini-infer', '--prompt', '', '--max-new-tokens', '0',
                                     '--device', 'cpu'])
    cli.main()
    assert capsys.readouterr().out == '\n'


def test_cli_zero_tokens_preserves_literal_special_token(monkeypatch, capsys):
    from engine import generate as cli
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained('gpt2', cache_dir='model_cache')
    model = GPT2Model(ModelConfig(vocab_size=50257, max_positions=16, hidden_size=24,
                                 num_layers=1, num_heads=4, intermediate_size=96)).eval()
    prompt = 'Hello <|endoftext|> world'
    monkeypatch.setattr(cli, 'load_model', lambda config: (model, tokenizer))
    monkeypatch.setattr(sys, 'argv', ['mini-infer', '--prompt', prompt,
                                     '--max-new-tokens', '0', '--device', 'cpu'])
    cli.main()
    assert capsys.readouterr().out == prompt + '\n'


def test_cli_empty_prompt_first_forward_uses_eos(model, monkeypatch, capsys):
    from engine import generate as cli
    seen = []
    hook = model.register_forward_pre_hook(lambda module, args: seen.append(args[0].tolist()))

    class Tokenizer:
        eos_token_id = 2

        def decode(self, ids, *, skip_special_tokens):
            return ','.join(str(i) for i in ids)

    monkeypatch.setattr(cli, 'load_model', lambda config: (model, Tokenizer()))
    monkeypatch.setattr(sys, 'argv', ['mini-infer', '--prompt', '', '--max-new-tokens', '1'])
    try:
        cli.main()
    finally:
        hook.remove()
    assert seen == [[[2]]]
    capsys.readouterr()
