import numpy as np
import math
from PIL import Image
import torch
import torch.nn.functional as F

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

labelMap = ([0,0,0],
            [0,255,0],
            [0,0,255],
            [255,0,0],
            [0,255,255],
            [255,255,0],
            [255,0,255],
            [255,255,255],
            [128,128,128],
            [255,192,192],
            [0,128,128])

def convert_to_rgb_label(img_arr):
    mappedResult = np.zeros((img_arr.shape[0],img_arr.shape[1],3),dtype=np.uint8)
    for y in range(img_arr.shape[0]):
        for x in range(img_arr.shape[1]):
            mappedResult[y,x] = labelMap[img_arr[y,x]]
    return mappedResult

TESTSIZE = 344
inputImg = Image.open('YOURIMAGEHERE')

newWidth = TESTSIZE
newHeight = TESTSIZE
if inputImg.height < inputImg.width:
   newHeight = TESTSIZE
   newWidth = TESTSIZE * inputImg.width / inputImg.height
else:
   newWidth = TESTSIZE
   newHeight = TESTSIZE * inputImg.height / inputImg.width

inputImg = inputImg.resize((int(newWidth),int(newHeight)),Image.BICUBIC)
deltaX = newWidth - TESTSIZE
deltaY = newHeight - TESTSIZE
if deltaX > 0 or deltaY > 0:
  leftX = int(deltaX / 2)
  rightX = deltaX - leftX
  topY = int(deltaY /2)
  bottomY = deltaY - topY
  inputImg = inputImg.crop((leftX,topY,newWidth-rightX,newHeight-bottomY))

inputArray = np.asarray(inputImg)
inputArray = (inputArray / 127.5) -1
inputArray = inputArray.transpose((2,0,1))
inputTensor = torch.as_tensor(inputArray,dtype=torch.float).to(device)
inputTensor = inputTensor.unsqueeze(0)
model = torch.jit.load("facesegmentation.pt").to(device)
model.eval()
output = model(inputTensor)
pred = F.softmax(output,dim=1)
numpy_data = pred[0].detach().cpu().numpy()
maxval = np.amax(numpy_data,axis=0)
idx = np.zeros((numpy_data.shape[1],numpy_data.shape[2]),dtype=np.uint8)
for n in range(numpy_data.shape[0]):
   for y in range(numpy_data.shape[1]):
      for x in range(numpy_data.shape[2]):
         if (maxval[y][x] == numpy_data[n][y][x]):
              idx[y][x] = n

rgbImage = convert_to_rgb_label(idx)
im = Image.fromarray(rgbImage)
im.save("demo.png")
