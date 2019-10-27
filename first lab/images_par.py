# 1024 x 768, 
# 1280 x 960, 
# 2048 x 1536

import sys
import pycuda.driver as cuda
import pycuda.autoinit
import numpy
import time
from pycuda.compiler import SourceModule
from PIL import Image, ImageDraw

threads_per_block = 1024

kernel_func = SourceModule(
	'''
		__global__ void halftone_substraction(float *a, float *b) {
			int idx = threadIdx.x + blockDim.x * blockIdx.x;
			float pixelA = a[idx];
			float pixelB = b[idx];
			a[idx] = ((pixelA - pixelB) + 255) / 2;
		}
	'''
)

if __name__ == '__main__':

	imageA = numpy.array(Image.open(str(sys.argv[1])).convert('RGB')).astype(numpy.float32)
	imageB = numpy.array(Image.open(str(sys.argv[2])).convert('RGB')).astype(numpy.float32)
	width = imageA.shape[0]	# because the size is the same for both images 
	height = imageA.shape[1]
	total =  width * height

	iA = []
	iB = []

	for i in range(width):
		for j in range(height):
			rA, rB = pixA[i, j][0], pixB[i, j][0]
			gA, gB = pixA[i, j][1], pixB[i, j][1]
			bA, bB = pixA[i, j][2], pixB[i, j][2]
			iA.append((rA + gA + bA) // 3)
			iB.append((rB + gB + bB) // 3)

	gpu_imageA_intensity = cuda.mem_alloc(iA.nbytes) # allocation of device memory iA
	gpu_imageB_intensity = cuda.mem_alloc(iB.nbytes) # allocation of device memory iB
	cuda.memcpy_htod(gpu_imageA_intensity, iA) # host (CPU) to device (GPU) for iA
	cuda.memcpy_htod(gpu_imageB_intensity, iB) # host (CPU) to device (GPU) for iB

	function = kernel_func.get_function("halftone_substraction")

	function(gpu_imageA_intensity, 
			 gpu_imageB_intensity, 
			 numpy.float32(total),
			 block = (threads_per_block, 1, 1),
			 grid = ((total + threads_per_block - 1) / threads_per_block, 1))

	out_image = Image.new('RGB', (width, height))
	draw = ImageDraw.Draw(out_image)

	start_time = time.time()

	for i in range(width):
		for j in range(height):
			draw.point((i, j), (iA[i][j], iA[i][j], iA[i][j]))

	print("time: {0:.4f}".format(time.time() - start_time))

	out_image.save('out_num_one.jpg', 'JPEG')
	del draw