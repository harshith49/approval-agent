"""Baseline generation: recompute the full sequence on every decoding step."""
import argparse

import torch

from engine.config import EngineConfig
from engine.model import GPT2Model
from engine.sampler import greedy
from engine.weights import load_model


@torch.inference_mode()
def generate(model: GPT2Model, input_ids: torch.Tensor, max_new_tokens: int,
             *, eos_token_id: int | None = None) -> torch.Tensor:
    """Return prompt plus generated IDs for one unpadded request.

    With eos_token_id=None, decode exactly max_new_tokens. Otherwise include
    EOS and stop. Reserve the entire requested output budget before decoding.
    """
    model.validate_input_ids(input_ids)
    if input_ids.shape[0] != 1:
        raise ValueError("Baseline generation accepts exactly one request")
    if not isinstance(max_new_tokens, int) or max_new_tokens < 0:
        raise ValueError("max_new_tokens must be a nonnegative integer")
    if input_ids.shape[1] + max_new_tokens > model.config.max_positions:
        raise ValueError("Prompt plus output budget exceeds the model context limit")
    if eos_token_id is not None and not 0 <= eos_token_id < model.config.vocab_size:
        raise ValueError("eos_token_id must be inside the vocabulary")
    output = input_ids
    for _ in range(max_new_tokens):
        token = greedy(model(output)[:, -1, :])
        output = torch.cat((output, token[:, None]), dim=1)
        if eos_token_id is not None and token.item() == eos_token_id:
            break
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description="mini-infer: uncached GPT-2 generation")
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--max-new-tokens", type=int, default=50)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    args = parser.parse_args()
    try:
        if args.max_new_tokens < 0:
            raise ValueError("max_new_tokens must be a nonnegative integer")
        model, tokenizer = load_model(EngineConfig(device=args.device))
        device = model.token_embedding.weight.device
        # GPT-2 cannot forward an empty sequence; EOS is also its BOS seed.
        ids = (tokenizer(args.prompt, return_tensors="pt")["input_ids"] if args.prompt
               else torch.tensor([[tokenizer.eos_token_id]], dtype=torch.long))
        output = generate(model, ids.to(device), args.max_new_tokens,
                          eos_token_id=tokenizer.eos_token_id)
    except ValueError as error:
        parser.error(str(error))
    # Preserve user text verbatim, including literal special-token spellings.
    # Only newly generated special tokens (including the terminal EOS) are hidden.
    new_ids = output[0, ids.shape[1]:].tolist()
    continuation = tokenizer.decode(new_ids, skip_special_tokens=True) if new_ids else ""
    print(args.prompt + continuation)


if __name__ == "__main__":
    main()
