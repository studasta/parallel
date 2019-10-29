import sys
import pycuda.driver as cuda
import pycuda.autoinit
import numpy
import time
from pycuda.compiler import SourceModule
from PIL import Image, ImageDraw

threads_per_block = 1024

start = cuda.Event()
end = cuda.Event()

kernel_func_subs = SourceModule(
	'''
		__global__ void halftone_substraction(float *a, float *b) {
			int idx = (threadIdx.x + blockDim.x * blockIdx.x) * 3;

			float r_comp = a[idx];
			float g_comp = a[idx+1];
			float b_comp = a[idx+2];
			float insty_a = (r_comp + g_comp + b_comp) / 3;

			r_comp = b[idx];
			g_comp = b[idx+1];
			b_comp = b[idx+2];
			float insty_b = (r_comp + g_comp + b_comp) / 3;

			float subs_insty = (insty_a - insty_b + 255) / 2;

			a[idx] = subs_insty;
			a[idx+1] = subs_insty;
			a[idx+2] = subs_insty;
		}
	'''
)

if __name__ == '__main__':

	img_a = numpy.array(Image.open(str(sys.argv[1])).convert('RGB')).astype(numpy.float32)	
	img_b = numpy.array(Image.open(str(sys.argv[2])).convert('RGB')).astype(numpy.float32)	
	width = img_a.shape[0]	# because the size is the same for both images 
	height = img_a.shape[1]
	total =  width * height

	block = (threads_per_block, 1, 1)
	grid = (int(total / threads_per_block), 1)

	gpu_img_a = cuda.mem_alloc(img_a.nbytes)
	gpu_img_b = cuda.mem_alloc(img_b.nbytes)
	cuda.memcpy_htod(gpu_img_a, img_a)
	cuda.memcpy_htod(gpu_img_b, img_b)

	start.record()
	func = kernel_func_subs.get_function("halftone_substraction")
	func(gpu_img_a, gpu_img_b, block = block, grid = grid)
	end.record()
	end.synchronize()
	secs = start.time_till(end)*1e-3

	print("time on GPU: {}".format(secs))

	cuda.memcpy_dtoh(img_a, gpu_img_a)
	img_out = Image.fromarray(img_a.astype(numpy.uint8))
	img_out.save("resultGPU.jpg")