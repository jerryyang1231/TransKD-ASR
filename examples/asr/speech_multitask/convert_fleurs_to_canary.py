import os
import re
import json
from pathlib import Path
import torchaudio
import inflect

# 初始化 inflect
p = inflect.engine()

# Fleurs 英文資料集路徑
DATA_ROOT = "/share/nas169/jerryyang/corpus/fleurs/en_us"
OUTPUT_DIR = "/share/nas169/jerryyang/corpus/fleurs/en_us/manifest_canary"

# 確保輸出目錄存在
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 需要處理的子集
SUBSETS = ["train", "validation", "test"]

def convert_numbers_to_words(text):
    # 利用正則表達式找到所有獨立的數字，並轉換成文字
    return re.sub(r'\b\d+\b', lambda m: p.number_to_words(m.group(0)), text)

# 轉換 Fleurs 到 Canary 格式的函數
def convert_fleurs_to_canary(subset):
    input_dir = Path(DATA_ROOT) / subset
    output_manifest_path = Path(OUTPUT_DIR) / f"{subset}_manifest.json"

    # 讀取 transcriptions.txt
    transcription_file = input_dir / "transcriptions.txt"
    if not transcription_file.exists():
        print(f"❌ 找不到 {transcription_file}，跳過 {subset}。")
        return

    # 解析 transcriptions.txt，建立索引對應
    transcriptions = {}
    with open(transcription_file, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split(": ", 1)  # index: transcription
            if len(parts) == 2:
                index, text = parts
                transcriptions[index.strip()] = text.strip()

    # 生成 Canary manifest
    with open(output_manifest_path, "w", encoding="utf-8") as manifest_file:
        for index, transcript in transcriptions.items():
            audio_file = input_dir / f"{index}.wav"

            if not audio_file.exists():
                print(f"⚠️ 找不到音檔 {audio_file}，跳過。")
                continue
            
            # 讀取音訊時長
            try:
                waveform, sample_rate = torchaudio.load(str(audio_file))
                duration = waveform.shape[1] / sample_rate  # 以秒為單位
            except Exception as e:
                print(f"❌ 讀取音檔 {audio_file} 發生錯誤: {e}")
                continue

            # 將數字轉成文字
            transcript = convert_numbers_to_words(transcript)

            # 建立 Canary 格式的 JSON 條目
            manifest_entry = {
                "audio_filepath": str(audio_file),
                "duration": duration,
                "taskname": "asr",
                "source_lang": "en",
                "target_lang": "en",
                "lang": "en",
                "pnc": "no",
                "text": transcript,
            }

            # 寫入 JSON 行
            manifest_file.write(json.dumps(manifest_entry, ensure_ascii=False) + "\n")
    
    print(f"✅ {subset} 集的 Canary manifest 已生成: {output_manifest_path}")

# 轉換所有子集
for subset in SUBSETS:
    convert_fleurs_to_canary(subset)

print("🚀 所有 Fleurs 英文資料集已成功轉換為 Canary manifest！")
