import urllib.request

urls = [
    "https://cdn.jsdelivr.net/gh/karpathy/char-rnn@master/data/tinyshakespeare/input.txt",
    "https://ghproxy.net/https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt",
    "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt",
]

for u in urls:
    try:
        print("尝试:", u)
        urllib.request.urlretrieve(u, "shakespeare.txt")
        print("成功")
        break
    except Exception as e:
        print("失败:", e)

text = open("shakespeare.txt", encoding="utf-8").read()
print("文本长度:", len(text))
print(text[:200])