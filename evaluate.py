"""
توليد إجابة النموذج على سؤال معيّن عبر sampling تلقائي (autoregressive).
"""

import torch

from model.tokenizer import ByteTokenizer


@torch.no_grad()
def generate_answer(model, tokenizer: ByteTokenizer, prompt: str, config,
                     max_new_tokens: int = 80, temperature: float = 0.9,
                     device: str = "cpu") -> str:
    model.eval()
    ids = tokenizer.encode(prompt, add_bos=True, add_eos=False)
    ids = ids[-config.block_size:]
    x = torch.tensor([ids], dtype=torch.long, device=device)

    for _ in range(max_new_tokens):
        x_cond = x[:, -config.block_size:]
        logits, _ = model(x_cond)
        logits = logits[:, -1, :] / max(temperature, 1e-5)
        probs = torch.softmax(logits, dim=-1)
        next_id = torch.multinomial(probs, num_samples=1)

        if next_id.item() == tokenizer.EOS_ID:
            break
        x = torch.cat([x, next_id], dim=1)

    generated_ids = x[0].tolist()
    return tokenizer.decode(generated_ids)
