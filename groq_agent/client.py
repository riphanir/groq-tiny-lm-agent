"""
الوكيل المسؤول عن التواصل مع Groq API.
Groq يلعب دور "المعلّم": يولّد بيانات تدريب، أسئلة اختبار، ويحلّل نقاط ضعف
النموذج الصغير بناءً على إجاباته الفعلية عبر كل التشغيلات.

ملاحظة مهمة: كل المحتوى المُولَّد بالإنجليزية عمدًا، لأن الـ tokenizer يعمل
على مستوى البايت (byte-level)، والحرف الإنجليزي = بايت واحد في UTF-8 بينما
الحرف العربي يحتاج غالبًا بايتين أو أكثر، فنموذج صغير جدًا (10M) يتعلم
الإنجليزية بكفاءة وسرعة أعلى بوضوح من العربية ضمن نافذة سياق 256 بايت فقط.
"""

import json
import re

from groq import Groq


class GroqAgent:
    def __init__(self, api_key: str, model: str = "llama-3.3-70b-versatile"):
        self.client = Groq(api_key=api_key)
        self.model = model

    # ---------- أدوات مساعدة داخلية ----------

    def _call_json(self, system_prompt: str, user_prompt: str,
                    max_tokens: int = 6000, temperature: float = 0.7, retries: int = 3):
        last_err = None
        for attempt in range(retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    max_tokens=max_tokens,
                    temperature=temperature,
                )
                text = response.choices[0].message.content.strip()
                result = self._extract_json(text)
                if isinstance(result, list) and result:
                    return result
                raise ValueError(f"شكل JSON غير متوقع أو فارغ: {type(result)}")
            except Exception as e:  # noqa: BLE001
                last_err = e
                print(f"[GroqAgent] محاولة {attempt + 1} فشلت: {e}")
        raise RuntimeError(f"فشل الاتصال بـ Groq بعد {retries} محاولات: {last_err}")

    @staticmethod
    def _extract_json(text: str):
        """
        يحاول استخراج قائمة JSON صالحة من رد Groq بعدة استراتيجيات متتالية،
        لأن النموذج أحيانًا يرجع:
          1) JSON array عادي وصحيح [{...}, {...}]
          2) NDJSON: كل عنصر JSON في سطر منفصل بدون أقواس مصفوفة تجمعها
          3) نص ملفوف بـ ```json ... ```
          4) نص فيه تعليق/مقدمة قبل أو بعد الـ JSON
        """
        cleaned = re.sub(r"^```(json)?", "", text.strip())
        cleaned = re.sub(r"```$", "", cleaned.strip()).strip()

        # 1) محاولة مباشرة: JSON array/object صالح كامل
        try:
            parsed = json.loads(cleaned)
            if isinstance(parsed, list):
                return parsed
            if isinstance(parsed, dict):
                for v in parsed.values():
                    if isinstance(v, list):
                        return v
        except json.JSONDecodeError:
            pass

        # 2) محاولة NDJSON: كل سطر عنصر JSON مستقل
        items = []
        for line in cleaned.splitlines():
            line = line.strip().rstrip(",")
            if not line or line in ("[", "]", "{", "}"):
                continue
            try:
                obj = json.loads(line)
                if isinstance(obj, dict):
                    items.append(obj)
            except json.JSONDecodeError:
                continue
        if items:
            return items

        # 3) استخراج كل {...} منفصل عبر regex بغض النظر عن الأسطر
        objects = re.findall(r"\{[^{}]*\}", cleaned, re.DOTALL)
        if objects:
            parsed_objs = []
            for obj_str in objects:
                try:
                    parsed_objs.append(json.loads(obj_str))
                except json.JSONDecodeError:
                    continue
            if parsed_objs:
                return parsed_objs

        # 4) آخر محاولة: أكبر [] أو {} موجود في النص
        match = re.search(r"(\[.*\]|\{.*\})", cleaned, re.DOTALL)
        if match:
            parsed = json.loads(match.group(1))
            if isinstance(parsed, list):
                return parsed
            if isinstance(parsed, dict):
                for v in parsed.values():
                    if isinstance(v, list):
                        return v

        raise ValueError("تعذر استخراج أي JSON صالح من رد Groq")

    # ---------- الوظائف الرئيسية ----------

    def generate_bootstrap_dataset(self, model_info: dict, n_examples: int = 100):
        """أول تشغيل: توليد بيانات تدريب تأسيسية لنموذج بأوزان عشوائية بالكامل."""
        system_prompt = (
            "You are a training-data generator for a tiny language model "
            "(~10 million parameters, randomly initialized, zero training so far) "
            "that uses a byte-level tokenizer. Generate very short, simple training "
            "examples (short sentences, basic facts, simple Q&A, common phrases, "
            "simple arithmetic) that help the model learn basic English grammar "
            "and vocabulary from scratch.\n\n"
            "IMPORTANT RULES:\n"
            "1. All content must be in English only. Do not use any other language.\n"
            "2. Every example must be unique - never repeat the same example twice.\n"
            "3. Return ONLY a single valid JSON array, starting with [ and ending "
            "with ]. Do not output one JSON object per line (NDJSON). Do not add "
            "trailing commas. Do not include any text, explanation, or markdown "
            "before or after the JSON array."
        )
        user_prompt = (
            f"Model info: {json.dumps(model_info, ensure_ascii=False)}\n\n"
            f"Generate exactly {n_examples} diverse, unique training examples. "
            "Each item must follow this exact shape:\n"
            '[{"prompt": "...", "completion": "..."}, ...]\n'
            "Keep each prompt and completion very short (under 15 words each)."
        )
        return self._call_json(system_prompt, user_prompt, max_tokens=6500)

    def generate_eval_questions(self, n_questions: int = 10):
        """تشغيلات لاحقة: توليد أسئلة اختبار لقياس مستوى النموذج الحالي."""
        system_prompt = (
            "You generate short evaluation prompts to test a tiny language model "
            "(~10 million parameters) that is still in early training. Vary the "
            "difficulty and type (sentence completion, simple fact question, "
            "simple arithmetic, etc.).\n\n"
            "IMPORTANT RULES:\n"
            "1. All content must be in English only.\n"
            "2. Return ONLY a single valid JSON array, starting with [ and ending "
            "with ]. Do not output one JSON object per line. No text before or "
            "after the JSON array."
        )
        user_prompt = (
            f"Generate exactly {n_questions} short, diverse evaluation prompts. "
            "Return ONLY a valid JSON array in this exact shape:\n"
            '[{"question": "..."}, ...]'
        )
        return self._call_json(system_prompt, user_prompt, max_tokens=2000)

    def generate_training_data_from_weaknesses(self, eval_history: list, n_examples: int = 100):
        """
        تحليل نقاط ضعف النموذج وتوليد بيانات تدريب تستهدفها.

        eval_history: كامل سجل الاختبارات من كل التشغيلات السابقة (وليس فقط
        آخر تشغيل)، بالشكل:
            [{"run": 1, "qa": [{"question": "...", "model_answer": "..."}, ...]},
             {"run": 2, "qa": [...]}, ...]
        إرسال كل السجل بدل آخر تشغيل فقط يخلي Groq يقدر يشوف مستوى النموذج
        وتطوره عبر الوقت (تحسّن أو ثبات أو تراجع)، مو بس صورة لحظية واحدة.
        """
        system_prompt = (
            "You are an expert at analyzing the evaluation history of a tiny "
            "language model (~10 million parameters) still in early training, and "
            "at generating short training examples that target its current "
            "weaknesses.\n\n"
            "You will receive the FULL evaluation history across ALL training runs "
            "so far - each run contains the questions asked and the model's actual "
            "answers at that point in time. Use the full history to understand the "
            "model's level and how it has progressed (or not) over time, but focus "
            "your generated training examples on fixing the weaknesses shown in the "
            "MOST RECENT run specifically (grammar mistakes, incoherence, "
            "repetition, wrong or garbled answers).\n\n"
            "IMPORTANT RULES:\n"
            "1. All content must be in English only, even if some questions in the "
            "history are in another language - translate the intent and respond "
            "in English.\n"
            "2. Every example must be unique.\n"
            "3. Return ONLY a single valid JSON array, starting with [ and ending "
            "with ]. Do not output one JSON object per line (NDJSON). Do not add "
            "trailing commas. No text before or after the JSON array."
        )
        user_prompt = (
            "Full evaluation history across all runs so far (most recent run is "
            f"last):\n{json.dumps(eval_history, ensure_ascii=False)}\n\n"
            "Analyze the model's progress across these runs, focus on the "
            f"weaknesses in the latest run, then generate exactly {n_examples} new, "
            "short training examples in English that address them specifically. "
            "Return ONLY a valid JSON array in this exact shape:\n"
            '[{"prompt": "...", "completion": "..."}, ...]\n'
            "Keep each prompt and completion very short (under 15 words each)."
        )
        return self._call_json(system_prompt, user_prompt, max_tokens=6500)
