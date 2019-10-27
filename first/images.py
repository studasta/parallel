import time 
from PIL import Image, ImageDraw

imageA, imageB = Image.open('first.jpg'), Image.open('second.jpg')
width = imageA.size[0]	# because the size is the same for both images 
height = imageA.size[1]
pixA, pixB = imageA.load(), imageB.load()

out_image = Image.new('RGB', (width, height))
draw = ImageDraw.Draw(out_image)

start_time = time.time()

for i in range(width):
	for j in range(height):
		rA, rB = pixA[i, j][0], pixB[i, j][0]
		gA, gB = pixA[i, j][1], pixB[i, j][1]
		bA, bB = pixA[i, j][2], pixB[i, j][2]
		iA = (rA + gA + bA) // 3
		iB = (rB + gB + bB) // 3
		iC = ((iA - iB) + 255) // 2
		draw.point((i, j), (iC, iC, iC))

print("time: {}".format(time.time() - start_time))

out_image.save('out_num_one.jpg', 'JPEG')
del draw

