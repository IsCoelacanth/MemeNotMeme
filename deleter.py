import os
import random as rm
path = '/home/rinzler/Documents/MemeNotMeme/Data/Train/memes'
files = os.listdir(path)
cnt = 0
print("prepping deletion")
while(cnt<21):
	files = os.listdir(path)
	for file in files:
		num = rm.random()
		if (num >= 0.75):
			os.unlink(file)
			cnt += 1
			print(cnt)
			if(cnt == 21):
				break