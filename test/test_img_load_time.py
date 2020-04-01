from image_loader import ImageLoader
import sys
import json
import time
from cvtools import cv_load_image

with open(sys.argv[1], 'r') as f:
    d = json.load(f)

imloader = ImageLoader('opencv', 'bgr')

st = time.time()
for ix, i in enumerate(d['images']):
    img = imloader.load(i['file_name'])
    sys.stdout.write("{}\r".format((time.time()-st)/(ix+1)))
    sys.stdout.flush()
