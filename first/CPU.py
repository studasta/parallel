import sys
import time 
from PIL import Image, ImageDraw

img_a, img_b = Image.open(str(sys.argv[1])), Image.open(str(sys.argv[2]))
width = img_a.size[0]	# because the size is the same for both images 
height = img_a.size[1]
pix_a, pix_b = img_a.load(), img_b.load()

start_time = time.time()

for i in range(width):
	for j in range(height):
		r_comp_a, r_comp_b = pix_a[i, j][0], pix_b[i, j][0]
		g_comp_a, g_comp_b = pix_a[i, j][1], pix_b[i, j][1]
		b_comp_a, b_comp_b = pix_a[i, j][2], pix_b[i, j][2]
		insty_a = (r_comp_a + g_comp_a + b_comp_a) // 3
		insty_b = (r_comp_b + g_comp_b + b_comp_b) // 3
		subs_insty = (insty_a - insty_b + 255) // 2
		pix_a[i,j] = (subs_insty, subs_insty, subs_insty)

print("time on CPU: {}".format(time.time() - start_time))

img_a.save('resultCPU.jpg', 'JPEG')