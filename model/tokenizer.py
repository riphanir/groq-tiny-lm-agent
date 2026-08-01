"""
Tokenizer بسيط على مستوى البايت (byte-level).
يبقي حجم الـ vocab صغيرًا جدًا (260) بحيث تذهب أغلب باراميترات النموذج
لطبقات الـ Transformer نفسها وليس لجدول الـ embedding، ويدعم أي لغة تلقائيًا
(بما فيها العربية) لأنه يعمل على البايتات الخام UTF-8.
"""


class ByteTokenizer:
    PAD_ID = 256
    BOS_ID = 257
    EOS_ID = 258
    VOCAB_SIZE = 260  # 256 بايت + PAD + BOS + EOS + خانة احتياط

    def encode(self, text: str, add_bos: bool = True, add_eos: bool = True):
        ids = list(text.encode("utf-8"))
        if add_bos:
            ids = [self.BOS_ID] + ids
        if add_eos:
            ids = ids + [self.EOS_ID]
        return ids

    def decode(self, ids) -> str:
        byte_vals = [i for i in ids if 0 <= i < 256]
        return bytes(byte_vals).decode("utf-8", errors="ignore")
