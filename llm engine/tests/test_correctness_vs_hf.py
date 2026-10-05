"""Public GPT-2 is the mandatory FP32 oracle, never a synthetic substitute."""
import pytest
import torch
from transformers import GPT2Config, GPT2LMHeadModel, AutoTokenizer

from engine.config import EngineConfig, ModelConfig
from engine.model import GPT2Model
from engine.weights import copy_hf_weights, load_model

PROMPTS = ['Hello, world!', 'The quick brown fox jumps over the lazy dog.',
           'Café — hello!\n  Spaces matter.']


@pytest.fixture(scope='session')
def public_models():
    reference = GPT2LMHeadModel.from_pretrained('gpt2', cache_dir='model_cache',
                                               attn_implementation='eager').float().eval()
    model, tokenizer = load_model(EngineConfig(device='cpu'))
    return model, reference, tokenizer


@pytest.mark.parametrize('prompt', PROMPTS)
def test_public_gpt2_logits(public_models, prompt):
    model, reference, tokenizer = public_models
    ids = tokenizer(prompt, return_tensors='pt')['input_ids']
    with torch.inference_mode():
        actual = model(ids)
        expected = reference(ids, use_cache=False).logits
    error = (actual - expected).abs().max().item()
    print(f'max absolute logit error: {error:.9g}')
    torch.testing.assert_close(actual, expected, atol=1e-4, rtol=1e-4,
                               msg=lambda msg: f'max error {error}: {msg}')


def test_tiny_weight_mapping_and_logits():
    torch.manual_seed(11)
    reference = GPT2LMHeadModel(GPT2Config(vocab_size=37, n_positions=16, n_embd=24,
        n_layer=2, n_head=4, attn_implementation='eager')).eval()
    model = GPT2Model(ModelConfig(vocab_size=37, max_positions=16, hidden_size=24,
        num_layers=2, num_heads=4, intermediate_size=96)).eval()
    copy_hf_weights(model, reference.state_dict())
    ids = torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8]])
    with torch.inference_mode():
        torch.testing.assert_close(model(ids), reference(ids).logits, atol=1e-6, rtol=1e-5)
    # Every parameter participates in reference behavior; square projections
    # especially must transpose even though their shape alone cannot reveal it.
    torch.testing.assert_close(model.blocks[0].attention.projection.weight,
                                reference.transformer.h[0].attn.c_proj.weight.T)
    assert model.lm_head.weight.data_ptr() == model.token_embedding.weight.data_ptr()


@pytest.mark.parametrize('failure', ['missing', 'shape'])
def test_bad_checkpoint_does_not_partially_modify_model(failure):
    cfg = ModelConfig(vocab_size=37, max_positions=16, hidden_size=24,
                      num_layers=2, num_heads=4, intermediate_size=96)
    model = GPT2Model(cfg)
    state = GPT2LMHeadModel(GPT2Config(vocab_size=37, n_positions=16, n_embd=24,
                           n_layer=2, n_head=4)).state_dict()
    key = 'transformer.h.1.mlp.c_proj.weight'
    if failure == 'missing':
        del state[key]
    else:
        state[key] = torch.zeros(1, 1)
    before = model.token_embedding.weight.detach().clone()
    with pytest.raises(ValueError, match=key.replace('.', r'\.')):
        copy_hf_weights(model, state)
    torch.testing.assert_close(model.token_embedding.weight, before, atol=0, rtol=0)


def test_loading_failure_preserves_context(monkeypatch):
    from engine import weights

    def offline(*args, **kwargs):
        raise OSError('not cached and network unavailable')

    monkeypatch.setattr(weights.GPT2Config, 'from_pretrained', offline)
    with pytest.raises(OSError, match='network unavailable'):
        load_model(EngineConfig(device='cpu'))


@pytest.mark.parametrize('prompt', PROMPTS)
def test_public_gpt2_greedy_50_tokens(public_models, prompt):
    from engine.generate import generate

    model, reference, tokenizer = public_models
    ids = tokenizer(prompt, return_tensors='pt')['input_ids']
    actual = generate(model, ids, 50)
    with torch.inference_mode():
        expected = reference.generate(ids, attention_mask=torch.ones_like(ids),
            max_new_tokens=50, do_sample=False, use_cache=False, eos_token_id=None,
            pad_token_id=tokenizer.eos_token_id)
    assert expected.shape[1] == ids.shape[1] + 50
    assert torch.equal(actual, expected)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA hardware unavailable')
def test_public_gpt2_cuda_parity(public_models):
    from engine.generate import generate
    model, reference, tokenizer = public_models
    ids = tokenizer(PROMPTS[0], return_tensors='pt')['input_ids'].to('cuda')
    try:
        model.to('cuda')
        reference.to('cuda')
        with torch.inference_mode():
            torch.testing.assert_close(model(ids), reference(ids, use_cache=False).logits,
                                       atol=1e-4, rtol=1e-4)
            expected = reference.generate(ids, attention_mask=torch.ones_like(ids),
                max_new_tokens=50, do_sample=False, use_cache=False, eos_token_id=None,
                pad_token_id=tokenizer.eos_token_id)
        assert torch.equal(generate(model, ids, 50), expected)
    finally:
        model.cpu()
        reference.cpu()
