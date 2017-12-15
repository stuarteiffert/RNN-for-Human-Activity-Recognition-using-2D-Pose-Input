from __future__ import print_function
import os

from PIL import Image
datapath = "/home/stuart/data/HAR_pose_activities/images/boxing"
images = 24
start = 24

for i in range(start, images+start):
	files = [
	  datapath+"/l01_c01_s01_a04_r01/pose/img_l01_c01_s01_a04_r01_000"+str(i)+"_rendered.png",
	  datapath+"/l02_c01_s01_a04_r01/pose/img_l02_c01_s01_a04_r01_000"+str(i)+"_rendered.png",
	  datapath+"/l03_c01_s01_a04_r01/pose/img_l03_c01_s01_a04_r01_000"+str(i)+"_rendered.png",
	  datapath+"/l04_c01_s01_a04_r01/pose/img_l04_c01_s01_a04_r01_000"+str(i)+"_rendered.png"]

	result = Image.new("RGB", (640, 480))

	for index, file in enumerate(files):
	  path = os.path.expanduser(file)
	  img = Image.open(path)
	  img.thumbnail((320, 240), Image.ANTIALIAS)
	  x = index // 2 * 320
	  y = index % 2 * 240
	  w, h = img.size
	  print('pos {0},{1} size {2},{3}'.format(x, y, w, h))
	  result.paste(img, (x, y, x + w, y + h))

	result.save(os.path.expanduser('~/boxing_image_'+str(i)+'.jpg'))
