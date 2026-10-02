import urllib.request
import os

urls = [
    "https://cdn.jsdelivr.net/gh/karpathy/makemore@master/names.txt",
    "https://fastly.jsdelivr.net/gh/karpathy/makemore@master/names.txt",
    "https://gcore.jsdelivr.net/gh/karpathy/makemore@master/names.txt",
    "https://raw.githubusercontent.com/karpathy/makemore/master/names.txt",
    "https://ghproxy.net/https://raw.githubusercontent.com/karpathy/makemore/master/names.txt",
    "https://mirror.ghproxy.com/https://raw.githubusercontent.com/karpathy/makemore/master/names.txt",
]

ok = False
for url in urls:
    try:
        print("尝试:", url)
        urllib.request.urlretrieve(url, "names.txt")
        ok = True
        print("下载成功:", url)
        break
    except Exception as e:
        print("失败:", type(e).__name__, e)

if not ok:
    print("\n全部失败。请手动下载 names.txt")
else:
    with open("names.txt", "r", encoding="utf-8") as f:
        words = f.read().splitlines()
    print("名字总数:", len(words))
    print("前 10 个:", words[:10])