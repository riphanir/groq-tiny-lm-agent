"""
الوكيل المسؤول عن التواصل مع Groq API.
Groq يلعب دور "المعلّم": يولّد بيانات تدريب، أسئلة اختبار، ويحلّل نقاط ضعف
النموذج الصغير بناءً على إجاباته الفعلية.
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
                    max_tokens: int = 6000, temperature: float = 0.85, retries: int = 3):
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
                return self._extract_json(text)
            except Exception as e:  # noqa: BLE001
                last_err = e
                print(f"[GroqAgent] محاولة {attempt + 1} فشلت: {e}")
        raise RuntimeError(f"فشل الاتصال بـ Groq بعد {retries} محاولات: {last_err}")

    @staticmethod
    def _extract_json(text: str):
        cleaned = re.sub(r"^```(json)?", "", text.strip())
        cleaned = re.sub(r"```$", "", cleaned.strip()).strip()
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            match = re.search(r"(\[.*\]|\{.*\})", cleaned, re.DOTALL)
            if match:
                return json.loads(match.group(1))
            raise

    # ---------- الوظائف الرئيسية ----------

    def generate_bootstrap_dataset(self, model_info: dict, n_examples: int = 100):
        """أول تشغيل: توليد بيانات تدريب تأسيسية لنموذج بأوزان عشوائية بالكامل."""
        system_prompt = (
            "أنت مولّد بيانات تدريب لنموذج لغوي صغير جدًا (حوالي 10 مليون باراميتر) "
            "بأوزان عشوائية غير مدرّب إطلاقًا، يستخدم tokenizer على مستوى البايت "
            "ونافذة سياق قصيرة جدًا. مهمتك توليد أمثلة تدريب بسيطة وقصيرة جدًا "
            "(جمل قصيرة، حقائق أساسية، أسئلة وأجوبة بسيطة، عبارات شائعة، عمليات حسابية "
            "بسيطة) تساعد النموذج على تعلّم أساسيات اللغة والنحو من الصفر."
        )
        user_prompt = (
            f"معلومات النموذج الحالي:\n{json.dumps(model_info, ensure_ascii=False)}\n\n"
            f"ولّد بالضبط {n_examples} مثال تدريب متنوع. أعد النتيجة فقط بصيغة JSON "
            "array صالحة، بدون أي نص إضافي أو Markdown، بهذا الشكل تمامًا:\n"
            '[{"prompt": "...", "completion": "..."}, ...]\n'
            "اجعل كل prompt وكل completion قصيرين جدًا (أقل من 30 كلمة لكل منهما)."
        )
        return self._call_json(system_prompt, user_prompt, max_tokens=6500)

    def generate_eval_questions(self, n_questions: int = 10):
        """تشغيلات لاحقة: توليد أسئلة اختبار لقياس مستوى النموذج الحالي."""
        system_prompt = (
            "أنت تولّد أسئلة اختبارية بسيطة لتقييم مستوى نموذج لغوي صغير جدًا "
            "(~10 مليون باراميتر) في مراحل تدريب مبكرة. نوّع في درجة الصعوبة "
            "وفي نوع السؤال (أكمل الجملة، سؤال حقيقة بسيطة، عملية حسابية، إلخ)."
        )
        user_prompt = (
            f"ولّد بالضبط {n_questions} سؤال/برومبت اختباري قصير ومتنوع. "
            "أعد النتيجة فقط بصيغة JSON array صالحة بدون أي نص إضافي، بهذا الشكل:\n"
            '[{"question": "..."}, ...]'
        )
        return self._call_json(system_prompt, user_prompt, max_tokens=2000)

    def generate_training_data_from_weaknesses(self, qa_pairs: list, n_examples: int = 100):
        """تحليل إجابات النموذج الفعلية وتوليد بيانات تدريب تستهدف نقاط ضعفه."""
        system_prompt = (
            "أنت خبير في تحليل مخرجات نموذج لغوي صغير جدًا ما زال في مراحل تدريب "
            "مبكرة، وفي توليد بيانات تدريب تستهدف نقاط ضعفه تحديدًا "
            "(أخطاء نحوية، عدم تماسك، تكرار، إجابات خاطئة أو غير مفهومة)."
        )
        user_prompt = (
            "هذه أسئلة طُرحت على النموذج مع إجاباته الفعلية الحالية:\n"
            f"{json.dumps(qa_pairs, ensure_ascii=False)}\n\n"
            f"حلّل نقاط الضعف الظاهرة في الإجابات، ثم ولّد بالضبط {n_examples} مثال "
            "تدريب جديد وقصير يعالج هذه النقاط تحديدًا. أعد النتيجة فقط بصيغة JSON "
            "array صالحة بدون أي نص إضافي، بهذا الشكل تمامًا:\n"
            '[{"prompt": "...", "completion": "..."}, ...]\n'
            "اجعل كل prompt وكل completion قصيرين جدًا (أقل من 30 كلمة لكل منهما)."
        )
        return self._call_json(system_prompt, user_prompt, max_tokens=6500)
