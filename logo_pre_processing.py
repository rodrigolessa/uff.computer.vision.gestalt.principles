import numpy as np
import cv2
import imutils

class LogoPreProcessing:

	def __init__(self, path, size, borderSize, whiteBackground = False):
		self.path = path
		self.size = size
		self.borderSize = borderSize
		self.whitebg = whiteBackground
		self.original = cv2.imread(path)
		# Convert it to grayscale
		# Podemos desconsiderar qualquer informação de cor na imagem 
		# para essa primeira implementação.
		# O próprio artigo não considera as cores ou camadas da imagem.
		self.grayscale = cv2.cvtColor(self.original, cv2.COLOR_BGR2GRAY)
		self.normalize = self._get_threshold()

	def _get_threshold(self):
		# Blur
		# Bilateral Filter can reduce unwanted noise very well and preserve border
		# blur = cv2.bilateralFilter(grayscale, 9, 75, 75)

		# TODO: Verificar se o fundo da imagem é preto.
		# Usar o primeiro pixel ou maior ocorrencia

		_, threshold = cv2.threshold(self.grayscale, 0 , 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
		#THRESH_BINARY = fundo preto or THRESH_BINARY_INV = fundo branco

		return threshold

	def crop(self, img):
		"""
		Find where the signature is and make a cropped region

		Return
			Only part of the image that contains drawings
		"""
		y, x = img.shape

		# Find where the black pixels are
		points = np.argwhere(self.normalize == 0)
		# Store them in x,y coordinates instead of row, col indices
		points = np.fliplr(points)
		
		# Create a rectangle around those points
		x, y, w, h = cv2.boundingRect(points)
		
		del points
		
		# Make the box a little bigger
		x, y, w, h = x - 10, y - 10, w + 20, h + 20
		
		if x < 0: x = 0
		if y < 0: y = 0

		# Crop the image
		return img[y:y+h, x:x+w]

	def scale(self, img):
		"""
		Diminuindo a imagem para o processo ser mais rápido
		Adicionado bordas para melhor visualização do logo
		"""
		new = imutils.resize(img, height = self.size)

		if new.shape[1] > self.size:
			new = imutils.resize(new, width = self.size)

		#border_size_x = (size - new.shape[1])//2
		#border_size_y = (size - new.shape[0])//2

		#borderColor = [0, 0, 0]

		#if bgWhite:
			#borderColor = [255,255,255]

		# TODO: Escrever meu próprio BorderMaker
		new = cv2.copyMakeBorder(
			new, 
			top = self.borderSize, 
			bottom = self.borderSize, 
			left = self.borderSize, 
			right = self.borderSize,
			borderType = cv2.BORDER_CONSTANT,
			value = cv2.BORDER_REPLICATE #borderColor
		)

		return new

