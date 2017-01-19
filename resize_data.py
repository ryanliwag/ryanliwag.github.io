import PIL
import os
import tensorflow as tf
from PIL import Image

basewidth = 300


n = 0
img_dirset = []
new_width  = 300
new_height = 300


for root, dirs, files in os.walk("/home/elements/Desktop/tensor_test/flower_photos/daisy"):
    for file in files:
        p=os.path.join(root, file)
        img = Image.open(str(p))
        img = img.resize((new_width, new_height), Image.ANTIALIAS)
    	img.save("daisy_" + str(n) +".jpg")
    	n = n + 1
	




