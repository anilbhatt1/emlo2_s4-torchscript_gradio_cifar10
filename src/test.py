import os

example1_img_path = './images/biard.jpeg'

if os.path.exists(example1_img_path):
    print('Image exists.')
else:
    print('Image does not exist.')