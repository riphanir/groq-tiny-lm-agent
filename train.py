"""
منطق تدريب النموذج على بيانات نصية متراكمة (byte-level next-token prediction).
مُحسَّن ليعمل على CPU ضمن الوقت المحدود لـ GitHub Actions.
"""

import random

import torch

from model.tokenizer import ByteTokenizer


def build_training_ids(examples: list, tokenizer: ByteTokenizer | None = None):
    """يحوّل قائمة أمثلة {'prompt', 'completion'} إلى تسلسل واحد طويل من الـ token ids."""
    tokenizer = tokenizer or ByteTokenizer()
    all_ids = []
    for ex in examples:
        prompt = str(ex.get("prompt", "")).strip()
        completion = str(ex.get("completion", "")).strip()
        text = f"{prompt}\n{completion}"
        all_ids.extend(tokenizer.encode(text, add_bos=True, add_eos=True))
    return all_ids


def get_batch(data_ids: list, block_size: int, batch_size: int, device: str):
    max_start = len(data_ids) - block_size - 1
    if max_start < 1:
        # بيانات قليلة جدًا: كرّرها حتى تغطي نافذة سياق واحدة على الأقل
        reps = (block_size + 2) // max(len(data_ids), 1) + 1
        data_ids = data_ids * reps
        max_start = len(data_ids) - block_size - 1

    ix = [random.randint(0, max_start) for _ in range(batch_size)]
    x = torch.tensor([data_ids[i:i + block_size] for i in ix], dtype=torch.long)
    y = torch.tensor([data_ids[i + 1:i + 1 + block_size] for i in ix], dtype=torch.long)
    return x.to(device), y.to(device)


def train_model(model, data_ids: list, config, steps: int = 300,
                 batch_size: int = 16, lr: float = 3e-4, device: str = "cpu",
                 log_every: int = 50):
    model.to(device)
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    for step in range(steps):
        x, y = get_batch(data_ids, config.block_size, batch_size, device)
        _, loss = model(x, y)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        if step % log_every == 0 or step == steps - 1:
            print(f"  خطوة {step:4d}/{steps} | loss = {loss.item():.4f}")

    model.eval()
    return model
