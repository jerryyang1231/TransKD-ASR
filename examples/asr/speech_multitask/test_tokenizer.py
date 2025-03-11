import sys
sys.path.insert(0, "/share/nas169/jerryyang/NeMo")
from nemo.collections.common.tokenizers.canary_tokenizer import CanaryTokenizer
from nemo.collections.common.tokenizers.sentencepiece_tokenizer import SentencePieceTokenizer

# 設定 special tokens tokenizer 與英文 tokenizer 的目錄
spl_tokens_dir = "/share/nas169/jerryyang/NeMo/examples/asr/speech_multitask/nemo_experiments/special_tokenizer"
en_tokenizer_dir = "/share/nas169/jerryyang/NeMo/examples/asr/speech_multitask/nemo_experiments/english_tokenizer/tokenizer_spe_bpe_v1024"

# 載入 tokenizer 模型
spl_tokenizer = SentencePieceTokenizer(f"{spl_tokens_dir}/tokenizer.model")
en_tokenizer = SentencePieceTokenizer(f"{en_tokenizer_dir}/tokenizer.model")

# 組合 tokenizer，直接使用 tokenizer 物件，不包在額外的字典中
tokenizer_dict = {
    "spl_tokens": spl_tokenizer,
    "en": en_tokenizer,
}

# 建立 aggregate tokenizer（CanaryTokenizer）
agg_tokenizer = CanaryTokenizer(tokenizers=tokenizer_dict)

# 測試英文句子
sentence = "This is a test sentence."
# encoded = agg_tokenizer.text_to_ids(sentence, lang_id="en")
# decoded = agg_tokenizer.ids_to_text(encoded)
encoded_special = agg_tokenizer.text_to_ids("<|translate|>", lang_id="spl_tokens")
decoded = agg_tokenizer.ids_to_text(encoded_special)

print("原始句子：", sentence)
# print("編碼後：", encoded)
print("編碼後：", encoded_special)
print("解碼後：", decoded)
