import glob
import shutil
import random as rm
dst = '/home/rinzler/Documents/MemeNotMeme/Data/Validation/memes'
cnt = 1
for file in glob.glob("/home/rinzler/Documents/MemeNotMeme/Data/Train/memes/*.jpg"):
	num = rm.random()
	if (num >= 0.75):
		shutil.move(file,dst)
		cnt += 1
		print(cnt)
		if(cnt==91):
			break