from logo_caracteristicas import obter_bordas
from PIL import Image, ImageDraw, ImageFilter, ImageChops
from math import sqrt, pi, cos, sin
from collections import defaultdict
import numpy as np
import os

clear = lambda: os.system('cls')
clear()

# Carregando imagem como um objeto do PIL que contem a matrix 3D
img = Image.open("test.png")

# Testes para converter em cinza
matriz_3d       = img.load()
matriz_width    = img.width
matriz_height   = img.height

pixel_3d = matriz_3d[25, 105]

print( pixel_3d)
print((pixel_3d[0] + pixel_3d[1] + pixel_3d[2]) / 3)

# Criar imagem de saída
img_circulos = Image.new("RGB", img.size)
#img_circulos.paste(img)
img_circulos_draw = ImageDraw.Draw(img_circulos)

# Debug para desenhar as bordas
img_bordas = Image.new("RGB", img.size)
img_bordas_draw = ImageDraw.Draw(img_bordas)

# Aplicar implementação de Canny
limiares = obter_bordas(matriz_3d, matriz_width, matriz_height)

for x, y in limiares:
    img_bordas_draw.point((x, y), (255, 255, 255))

# Parâmetros para criação de círculos
rmin = 15
rmax = 60
qtd = 100

print('coss: {}'.format(43 * cos(2 * pi * qtd / qtd)))
print('seno: {}'.format(43 * sin(2 * pi * qtd / qtd)))

circulos_previstos = []

for r in range(rmin, rmax + 1):
    for t in range(qtd):
        circulos_previstos.append(
            (
                r, 
                int(r * cos(2 * pi * t / qtd)), 
                int(r * sin(2 * pi * t / qtd))
            )
        )

adic = defaultdict(int)
for x, y in limiares:
    for r, dx, dy in circulos_previstos:
        a = x - dx
        b = y - dy
        adic[(a, b, r)] += 1

circulos = []
limite = 0.3

print('limite: {}'.format(limite))

for k, v in sorted(adic.items(), key=lambda i: -i[1]):
    x, y, r = k
    if v / qtd >= limite and all((x - xc) ** 2 + (y - yc) ** 2 > rc ** 2 for xc, yc, rc in circulos):
        print(v / qtd, v, x, y, r)
        circulos.append((x, y, r))

# Círculos conhecidos
# Limiar    cosseno seno    raio
# 0.74      46      140     43
# 0.71      152     140     43
# 0.7       99      56      43

for x, y, r in circulos:
    img_circulos_draw.ellipse((x-r, y-r, x+r, y+r), outline=(255,255,255,0))

# Escreve as imagem para testes
img_bordas.save("resultado_bordas.png")
#img_bordas.show()

img_circulos.save("resultado_circulos.png")
#img_circulos.show()

img_dilatada = img_circulos.filter(ImageFilter.MaxFilter(5))
img_dilatada.save("resultado_circulos_max.png")
#img_dilatada.show()

img_dif = ImageChops.subtract(img_bordas, img_dilatada)
img_dif.save("resultado_circulos_subtraidos.png")
#img_dif.show()

#img_dif = img_dif.filter(ImageFilter.MaxFilter(5))
#img_dif = img_dif.filter(ImageFilter.MinFilter(5))
#img_dif.save("resultado_circulos_subtraidos.png")