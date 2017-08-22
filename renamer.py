import os
path = '/home/rinzler/Documents/MemeNotMeme/Data/memes'

files = os.listdir(path)

i = 1

for file in files:
	new_name = "meme_{}.jpg".format(i)
	i+=1
	print("Current file = ",file)
	os.rename(os.path.join(path, file), os.path.join(path, new_name))