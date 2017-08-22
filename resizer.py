import os
from PIL import Image
resize_method = Image.ANTIALIAS
    #Image.NEAREST)  # use nearest neighbour
    #Image.BILINEAR) # linear interpolation in a 2x2 environment
    #Image.BICUBIC) # cubic spline interpolation in a 4x4 environment
    # Image.ANTIALIAS) # best down-sizing filter
directory = '/home/rinzler/Documents/MemeNotMeme/Data/Validation/memes'

count = 1
i = 1
for file_name in os.listdir(directory):
  print("Processing %s" % file_name)
  print("Current file # = {}".format(count))
  count += 1
  image = Image.open(os.path.join(directory, file_name))
  if image.mode != 'RGB':
  	image = image.convert('RGB')
  x,y = image.size
  new_dimensions = (150, 150)
  output = image.resize(new_dimensions, Image.ANTIALIAS)
  new_name = "meme_{}.jpg".format(i)
  i+=1
  print("Current file = ",file_name)
  output_file_name = os.path.join(directory,new_name)
  output.save(output_file_name, "JPEG", quality = 98)

print("All done")