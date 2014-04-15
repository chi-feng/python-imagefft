from PIL import Image
import numpy as np
from scipy import fftpack
from matplotlib import pyplot as plt

def transform(filename, inverse=False, shift=False):
	img = Image.open(filename).convert(mode='L')
	src = np.asarray(img) / 255.0
	if shift: 
		src = fftpack.fftshift(src)
	if inverse: 
		transformed = fftpack.ifft2(src)
	else: 
		transformed = fftpack.fft2(src)
	transformed = fftpack.fftshift(transformed)
	magnitude = np.abs(transformed)
	return src, magnitude

def save_image(magnitude, filename):
	output = magnitude * 255 / (np.max(magnitude) - np.min(magnitude))
	Image.fromarray(output.astype(np.uint8), 'L').save(filename)

def make_plot(src, magnitude, filename=None):
	plt.clf()
	plt.subplot(131)
	plt.imshow(src, cmap=plt.cm.jet)
	plt.title('Original Image')
	plt.subplot(132)
	plt.imshow(magnitude, cmap=plt.cm.jet)
	plt.title('FFT (linear scale)')
	plt.subplot(133)
	offset = (np.max(magnitude) - np.min(magnitude)) / 255 / 5
	plt.imshow(np.log10(magnitude + offset), cmap=plt.cm.jet)
	plt.title('FFT (log scale)')
	if filename is not None: 
		plt.savefig(filename, figsize=(10, 5), dpi=120, bbox_inches='tight')

src, magnitude = transform('octagon.png', inverse=False, shift=False)
make_plot(src, magnitude, filename='octagon_transformed.pdf')

src, magnitude = transform('hexagon.png', inverse=False, shift=False)
make_plot(src, magnitude, filename='hexagon_transformed.pdf')

src, magnitude = transform('circle.png', inverse=False, shift=False)
make_plot(src, magnitude, filename='circle_transformed.pdf')