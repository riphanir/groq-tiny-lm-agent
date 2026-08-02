"""
نقطة الدخول الرئيسية للوكيل.

أول تشغيل (لا يوجد checkpoint):
  1) إنشاء نموذج بأوزان عشوائية (~10M باراميتر).
  2) إرسال حالة النموذج إلى Groq -> يولّد 100 مثال تدريب تأسيسي.
  3) تدريب النموذج على هذه البيانات.

التشغيلات اللاحقة:
  1) تحميل النموذج المُدرَّب سابقًا.
  2) طلب 10 أسئلة اختبار من Groq.
  3) توليد إجابات النموذج الحالي على هذه الأسئلة.
  4) إرسال كامل سجل الاختبارات (كل التشغيلات السابقة) إلى Groq ليحلّل
     مستوى النموذج وتقدمه، ويولّد 100 مثال تدريب جديد يستهدف نقاط الضعف.
  5) تدريب النموذج على كامل البيانات المتراكمة (القديمة + الجديدة).

في كل الحالات: حفظ الـ checkpoint + الحالة + البيانات، ليتم لاحقًا commit تلقائي
عبر GitHub Actions.
"""

import json
import os
from pathlib import Path

import torch

from model.architecture import GPT, GPTConfig
from model.tokenizer import ByteTokenizer
from train import build_training_ids, train_model
from evaluate import generate_answer
from groq_agent.client import GroqAgent

ROOT = Path(__file__).parent
CKPT_DIR = ROOT / "checkpoints"
DATA_DIR = ROOT / "data"
STATE_DIR = ROOT / "state"

CKPT_PATH = CKPT_DIR / "model.pt"
CONFIG_PATH = CKPT_DIR / "config.json"
STATE_PATH = STATE_DIR / "state.json"
TRAIN_DATA_PATH = DATA_DIR / "train_data.jsonl"
EVAL_HISTORY_PATH = DATA_DIR / "eval_history.jsonl"


def _env_int(name: str, default: int) -> int:
    """يقرأ متغير بيئة كرقم صحيح، ويتجاهل القيم الفارغة (GitHub Actions يبعت '' لو
    الـ variable مش معرّف بدل ما يشيل المتغير خالص)."""
    raw = os.environ.get(name, "").strip()
    return int(raw) if raw else default


TRAIN_STEPS = _env_int("TRAIN_STEPS", 300)
BATCH_SIZE = _env_int("TRAIN_BATCH_SIZE", 16)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def load_state() -> dict:
    if STATE_PATH.exists():
        return json.loads(STATE_PATH.read_text(encoding="utf-8"))
    return {"run_count": 0, "total_examples": 0}


def save_state(state: dict):
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    STATE_PATH.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")


def _example_key(ex: dict) -> tuple:
    """مفتاح توحيد يحدد إذا كان مثالين متطابقين فعليًا (يتجاهل فروق المسافات/الحالة)."""
    return (
        str(ex.get("prompt", "")).strip().lower(),
        str(ex.get("completion", "")).strip().lower(),
    )


def _load_existing_keys() -> set:
    """يبني مجموعة بمفاتيح كل الأمثلة المخزّنة حاليًا، لاستخدامها في منع التكرار."""
    keys = set()
    if not TRAIN_DATA_PATH.exists():
        return keys
    with open(TRAIN_DATA_PATH, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                ex = json.loads(line)
            except json.JSONDecodeError:
                continue
            if "prompt" in ex and "completion" in ex:
                keys.add(_example_key(ex))
    return keys


def append_examples(examples: list) -> tuple:
    """
    يضيف أمثلة جديدة لملف التدريب، ويتجاهل تلقائيًا أي مثال مكرر حرفيًا
    (نفس prompt + نفس completion، بغض النظر عن حالة الأحرف/المسافات الزائدة)
    سواء كان مكررًا مع بيانات محفوظة سابقًا أو مكررًا داخل نفس الدفعة الجديدة.
    يرجع (عدد المُضاف فعليًا, عدد المتجاهَل لأنه مكرر).
    """
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    existing_keys = _load_existing_keys()
    added, skipped = 0, 0

    with open(TRAIN_DATA_PATH, "a", encoding="utf-8") as f:
        for ex in examples:
            if "prompt" not in ex or "completion" not in ex:
                continue
            key = _example_key(ex)
            if key in existing_keys:
                skipped += 1
                continue
            existing_keys.add(key)
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")
            added += 1

    return added, skipped


def load_all_examples() -> list:
    if not TRAIN_DATA_PATH.exists():
        return []
    examples = []
    with open(TRAIN_DATA_PATH, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                examples.append(json.loads(line))
    return examples


def load_eval_history() -> list:
    """كامل سجل الاختبارات من كل التشغيلات السابقة (يُرسَل كاملاً لـ Groq، بدون سقف)."""
    if not EVAL_HISTORY_PATH.exists():
        return []
    history = []
    with open(EVAL_HISTORY_PATH, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                history.append(json.loads(line))
    return history


def save_checkpoint(model, config: GPTConfig):
    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), CKPT_PATH)
    CONFIG_PATH.write_text(json.dumps(config.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")


def main():
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError("متغير البيئة GROQ_API_KEY غير موجود. أضفه كـ Secret في GitHub.")

    groq_model = os.environ.get("GROQ_MODEL", "").strip() or "llama-3.3-70b-versatile"
    agent = GroqAgent(api_key=api_key, model=groq_model)
    tokenizer = ByteTokenizer()
    state = load_state()

    is_first_run = not CKPT_PATH.exists()

    if is_first_run:
        print(">>> أول تشغيل: إنشاء نموذج جديد بأوزان عشوائية")
        config = GPTConfig(vocab_size=ByteTokenizer.VOCAB_SIZE)
        model = GPT(config)
        n_params = model.num_params()
        print(f"عدد باراميترات النموذج: {n_params:,}")

        model_info = {
            "param_count": n_params,
            "architecture": "GPT-style transformer (from scratch)",
            "tokenizer": "byte-level",
            "n_layer": config.n_layer,
            "n_embd": config.n_embd,
            "n_head": config.n_head,
            "block_size": config.block_size,
            "status": "randomly initialized, zero training so far",
        }

        print(">>> طلب 100 مثال تدريب تأسيسي من Groq...")
        examples = agent.generate_bootstrap_dataset(model_info, n_examples=100)
        added, skipped = append_examples(examples)
        print(f"تم استلام {len(examples)} مثال | أُضيف {added} فريد | "
              f"تم تجاهل {skipped} مكرر")

        state["run_count"] = 0
    else:
        print(">>> تشغيل لاحق: تحميل النموذج المُدرَّب سابقًا")
        cfg_dict = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
        config = GPTConfig(**cfg_dict)
        model = GPT(config)
        model.load_state_dict(torch.load(CKPT_PATH, map_location="cpu"))
        print(f"عدد باراميترات النموذج: {model.num_params():,}")

        print(">>> طلب 10 أسئلة اختبار من Groq...")
        questions = agent.generate_eval_questions(n_questions=10)
        print(f"تم استلام {len(questions)} سؤال")

        qa_pairs = []
        model.to(DEVICE)
        for q in questions:
            question_text = q.get("question", "")
            answer = generate_answer(model, tokenizer, question_text, config, device=DEVICE)
            qa_pairs.append({"question": question_text, "model_answer": answer})
            print(f"  س: {question_text}\n  ج: {answer}\n")

        DATA_DIR.mkdir(parents=True, exist_ok=True)
        with open(EVAL_HISTORY_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(
                {"run": state["run_count"] + 1, "qa": qa_pairs}, ensure_ascii=False
            ) + "\n")

        full_eval_history = load_eval_history()  # كل التشغيلات، شامل التشغيل الحالي
        print(f">>> إرسال كامل سجل الاختبارات ({len(full_eval_history)} تشغيل سابق) "
              f"إلى Groq ليحلل مستوى النموذج وتقدمه ويولّد بيانات تدريب جديدة...")
        examples = agent.generate_training_data_from_weaknesses(full_eval_history, n_examples=100)
        added, skipped = append_examples(examples)
        print(f"تم استلام {len(examples)} مثال | أُضيف {added} فريد | "
              f"تم تجاهل {skipped} مكرر")

    all_examples = load_all_examples()
    print(f">>> بدء التدريب على {len(all_examples)} مثال متراكم (steps={TRAIN_STEPS})...")
    data_ids = build_training_ids(all_examples, tokenizer)
    model = train_model(model, data_ids, config, steps=TRAIN_STEPS,
                         batch_size=BATCH_SIZE, device=DEVICE)

    model.to("cpu")
    save_checkpoint(model, config)

    state["run_count"] = state.get("run_count", 0) + 1
    state["total_examples"] = len(all_examples)
    save_state(state)

    print(f">>> تم حفظ النموذج والحالة بنجاح (run_count={state['run_count']}, "
          f"total_examples={state['total_examples']})")


if __name__ == "__main__":
    main()
