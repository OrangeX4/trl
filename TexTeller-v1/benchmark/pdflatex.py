# 调用 pdflatex 编译 test.tex
# pdf 到 png
from pdf2image import convert_from_path
import os
import time

loop = 100

start_time = time.time()

for _ in range(loop):
    os.system("pdflatex test.tex")

mid_time = time.time()

for _ in range(loop):
    with open("test.pdf", "rb") as f:
        images = convert_from_path(f.name)

end_time = time.time()

print(f"pdflatex: {(mid_time - start_time) / loop * 1000:.2f} ms")
print(f"pdf2image: {(end_time - mid_time) / loop * 1000:.2f} ms")
print(f"total: {(end_time - start_time) / loop * 1000:.2f} ms")
