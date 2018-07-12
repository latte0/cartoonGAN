import cv2
import numpy as np
import os


path = "./dataset/ukiyoe2photo/anime_train/anime/"
save = "./dataset/ukiyoe2photo/animeblur_train/animeblur/"


files = os.listdir(path)

for file in files:

	img = cv2.imread(path + file)

	blur = cv2.GaussianBlur(img,(5,5),0)

	print(file)

	cv2.imwrite(save + file, blur)
