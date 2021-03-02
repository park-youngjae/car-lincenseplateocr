from PIL import Image
import os
path = r'C:\Users\PARK\Downloads\Sample'
dataset_list = []
for img in os.listdir(path):
    image = Image.open(os.path.join(path,img))
    label = img[0:img.find(".")]
    dataset = (image,label)
    dataset_list.append(dataset)
print(dataset_list)