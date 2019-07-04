from PIL import Image, ImageDraw
from math import sqrt, pi, cos, sin
from logo_features import canny_edge_detector
from collections import defaultdict
import numpy as np
import cv2
import os

clear = lambda: os.system('cls')
clear()

# Carregando imagem como um objeto do PIL que contem a matrix 3D
input_image = Image.open("test.png")
output_image = Image.new("RGB", input_image.size)
output_image.paste(input_image)
draw_result = ImageDraw.Draw(output_image)

# Debug para desenhar as bordas
bordas = Image.new("RGB", input_image.size)
bordasDraw = ImageDraw.Draw(bordas)

edges = canny_edge_detector(input_image)

for x, y in edges:
    bordasDraw.point((x, y), (255, 255, 255))

bordas.save("resultado_bordas.png")

# Testes para converter em cinza
matriz3D = input_image.load()
pixel3D = matriz3D[14, 94]
print(pixel3D)
print((pixel3D[0] + pixel3D[1] + pixel3D[2]) / 3)

# Parâmetros para criação de circulos
rmin = 15
rmax = 60
steps = 100
threshold = 0.3

print('limiar: {}'.format(threshold))
print('coss: {}'.format(43 * cos(2 * pi * steps / steps)))
print('seno: {}'.format(43 * sin(2 * pi * steps / steps)))

points = []
for r in range(rmin, rmax + 1):
    for t in range(steps):
        points.append(
            (
                r, 
                int(r * cos(2 * pi * t / steps)), 
                int(r * sin(2 * pi * t / steps))
            )
        )

acc = defaultdict(int)
for x, y in edges:
    for r, dx, dy in points:
        a = x - dx
        b = y - dy
        acc[(a, b, r)] += 1

circles = []
for k, v in sorted(acc.items(), key=lambda i: -i[1]):
    x, y, r = k
    if v / steps >= threshold and all((x - xc) ** 2 + (y - yc) ** 2 > rc ** 2 for xc, yc, rc in circles):
        print(v / steps, v, x, y, r)
        circles.append((x, y, r))

# Circulos conhecidos
# Limiar    cosseno seno    raio
# 0.74      46      140     43
# 0.71      152     140     43
# 0.7       99      56      43

for x, y, r in circles:
    draw_result.ellipse((x-r, y-r, x+r, y+r), outline=(255,0,0,0))

# Save output image
output_image.save("resultado_circulos.png")
#output_image.show()