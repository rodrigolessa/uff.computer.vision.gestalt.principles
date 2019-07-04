# Funções: raiz quadrada, arco tangente, valo de Pi
from math import sqrt, atan2, pi
# Funções matemáticas, de álgebra linear e transformada de Fourier
# Aplicações para zeros, empty, array
import numpy as np

# Transformar a imagem em uma matriz 2D com cinza
def compute_grayscale(input_pixels, width, height):
    grayscale = np.empty((width, height))
    for x in range(width):
        for y in range(height):
            pixel = input_pixels[x, y]
            grayscale[x, y] = (pixel[0] + pixel[1] + pixel[2]) / 3
    return grayscale

# 1. Suavizar a imagem de entrada com um filtro Gaussiano
# Desvio padrão: Sigma = 1
# Tamanho do kernel: 5
def compute_blur(input_pixels, width, height):
    # Garantir que as coordenadas estão dentro da imagem
    clip = lambda x, l, u: l if x < l else u if x > u else x

    # kernel gaussiana - bidimensional 
    # http://dev.theomader.com/gaussian-kernel-calculator/
    # Esses pesos abaixo serão usados ​​diretamente em um algoritmo de desfoque 
    # de passagem única: amostras por pixel.
    # 0.003765	0.015019	0.023792	0.015019	0.003765
    # 0.015019	0.059912	0.094907	0.059912	0.015019
    # 0.023792	0.094907	0.150342	0.094907	0.023792
    # 0.015019	0.059912	0.094907	0.059912	0.015019
    # 0.003765	0.015019	0.023792	0.015019	0.003765
    kernel = np.array([
        [1 / 256,  4 / 256,  6 / 256,  4 / 256, 1 / 256],
        [4 / 256, 16 / 256, 24 / 256, 16 / 256, 4 / 256],
        [6 / 256, 24 / 256, 36 / 256, 24 / 256, 6 / 256],
        [4 / 256, 16 / 256, 24 / 256, 16 / 256, 4 / 256],
        [1 / 256,  4 / 256,  6 / 256,  4 / 256, 1 / 256]
    ])

    # Middle of the kernel
    offset = len(kernel) // 2

    # Compute the blurred image
    blurred = np.empty((width, height))
    for x in range(width):
        for y in range(height):
            acc = 0
            for a in range(len(kernel)):
                for b in range(len(kernel)):
                    xn = clip(x + a - offset, 0, width - 1)
                    yn = clip(y + b - offset, 0, height - 1)
                    acc += input_pixels[xn, yn] * kernel[a, b]
            blurred[x, y] = int(acc)
    return blurred

# 2. Calcular a direção e magnitude do gradiente da imagem suavizada 
def compute_gradient(input_pixels, width, height):
    gradient = np.zeros((width, height))
    direction = np.zeros((width, height))
    for x in range(width):
        for y in range(height):
            if 0 < x < width - 1 and 0 < y < height - 1:
                magx = input_pixels[x + 1, y] - input_pixels[x - 1, y]
                magy = input_pixels[x, y + 1] - input_pixels[x, y - 1]
                gradient[x, y] = sqrt(magx**2 + magy**2)
                # Ao calcular a imagem gradiente, 
                # também calculamos a direção do gradiente atan2(y, x). 
                # Usando isso, apenas mantemos os pixels que são o máximo 
                # entre os vizinhos na direção do gradiente. 
                # Isso vai diluir as arestas
                direction[x, y] = atan2(magy, magx)
    return gradient, direction

# 3. Supressão dos Não-Máximos (Non-Maxima Suppression)
def filter_out_non_maximum(gradient, direction, width, height):
    for x in range(1, width - 1):
        for y in range(1, height - 1):
            angle = direction[x, y] if direction[x, y] >= 0 else direction[x, y] + pi
            rangle = round(angle / (pi / 4))
            mag = gradient[x, y]
            if ((rangle == 0 or rangle == 4) and (gradient[x - 1, y] > mag or gradient[x + 1, y] > mag)
                    or (rangle == 1 and (gradient[x - 1, y - 1] > mag or gradient[x + 1, y + 1] > mag))
                    or (rangle == 2 and (gradient[x, y - 1] > mag or gradient[x, y + 1] > mag))
                    or (rangle == 3 and (gradient[x + 1, y - 1] > mag or gradient[x - 1, y + 1] > mag))):
                gradient[x, y] = 0

# 4. Utilizar limiar duplo e análise de conectividade para detectar e encadear bordas
def filter_strong_edges(gradient, width, height, low, high):
    # Keep strong edges
    keep = set()
    for x in range(width):
        for y in range(height):
            if gradient[x, y] > high:
                keep.add((x, y))

    # Keep weak edges next to a pixel to keep
    lastiter = keep
    while lastiter:
        newkeep = set()
        for x, y in lastiter:
            for a, b in ((-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)):
                if gradient[x + a, y + b] > low and (x+a, y+b) not in keep:
                    newkeep.add((x+a, y+b))
        keep.update(newkeep)
        lastiter = newkeep

    return list(keep)

# Canny
# Boa detecção, boa localização e resposta mínima
# Algoritmo de quatro passos 
# 1. Suavizar a imagem de entrada com um filtro Gaussiano
# 2. Calcular a direção e magnitude do gradiente da imagem suavizada
# 3. Aplicar supressão dos não-máximos (non-maxima suppression)
# 4. Utilizar limiar duplo e análise de conectividade para detectar e encadear bordas
def canny_edge_detector(input_image):

    input_pixels = input_image.load()
    width = input_image.width
    height = input_image.height

    # Transformar a imagem em uma matriz 2D com cinza
    grayscaled = compute_grayscale(input_pixels, width, height)

    # 1. Suavizar a imagem de entrada com um filtro Gaussiano
    # Efetivo para remover ruídos, porem nossa imagem é bem limpa
    blurred = compute_blur(grayscaled, width, height)

    # 2. Calcular a direção e magnitude do gradiente
    gradient, direction = compute_gradient(blurred, width, height)

    # 3. Aplicar supressão dos não-máximos
    filter_out_non_maximum(gradient, direction, width, height)

    # 4. Utilizar limiar duplo e análise de conectividade para detectar e encadear bordas
    # Um filtro para 
    keep = filter_strong_edges(gradient, width, height, 20, 25)

    #print(keep)

    return keep