import re
import inflect

# 初始化 inflect
p = inflect.engine()

def convert_numbers_to_words(text):
    # 利用正則表達式找到所有獨立的數字，並轉換成文字
    return re.sub(r'\b\d+\b', lambda m: p.number_to_words(m.group(0)), text)

text = "robin uthappa made the innings highest score 70 runs in just 41 balls by hitting 11 fours and 2 sixes"

# 將數字轉成文字
final_text = convert_numbers_to_words(text)

print(final_text)
