import re
from collections import Counter

# 读取
with open("luotuo.txt", "r", encoding="utf-8") as f:
    text = f.read()

# 去掉 BOM 和常见乱码标记
text = text.replace("\ufeff", "")
text = re.sub(r"\(www\.jingdianbook\.com.*?\)", "", text)
text = re.sub(r"第【\d+】段：.*?\n", "", text)
text = re.sub(r"◆JingDianBook\.com经典书库◆", "", text)

# 去掉注释符号 ①②③ 这类
text = re.sub(r"[①-⑳]", "", text)

# 只保留：中文、英文、数字、常用标点、换行
keep = re.compile(r"[\u4e00-\u9fffA-Za-z0-9，。！？；：、“”‘’（）《》\n ]")
text = "".join(ch for ch in text if keep.match(ch))

# 去掉多余空行
text = re.sub(r"\n{3,}", "\n\n", text)

print("清洗后长度:", len(text))
print("前 200 字:")
print(text[:200])

# 统计字符频率
counter = Counter(text)
print("\n不同字符数:", len(counter))
print("最常见的 30 个:")
for ch, cnt in counter.most_common(30):
    print(repr(ch), cnt)

# 过滤低频字符：出现次数少于 10 的替换为 <unk>
min_count = 10
vocab = {ch for ch, cnt in counter.items() if cnt >= min_count}
print("\n过滤后字符数:", len(vocab))

filtered = "".join(ch if ch in vocab else "<unk>" for ch in text)
# 把 <unk> 当作一个整体处理，先替换成特殊字符
filtered = filtered.replace("<unk>", "?")

with open("luotuo_clean.txt", "w", encoding="utf-8") as f:
    f.write(filtered)

print("已保存 luotuo_clean.txt")
print("最终长度:", len(filtered))