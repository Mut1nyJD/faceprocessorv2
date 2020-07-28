import numpy as np
import onnxruntime as rt
import math
from PIL import Image

labelMap = ([0,0,0],          # background
            [0,255,0],        # eyes
            [0,0,255],        # nose
            [255,0,0],        # lips/mouth
            [0,255,255],      # ear
            [255,255,0],      # hair
            [255,0,255],      # eyebrows
            [255,255,255],    # teeth
            [128,128,128],    # general face
            [255,192,192],    # beard
            [0,128,128])      # specs/sunglasses

def convert_to_rgb_label(img_arr):
    mappedResult = np.zeros((img_arr.shape[0],img_arr.shape[1],3),dtype=np.uint8)
    for y in range(img_arr.shape[0]):
        for x in range(img_arr.shape[1]):
            mappedResult[y,x] = labelMap[img_arr[y,x]]
    return mappedResult

def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()

sess = rt.InferenceSession("facesegmentation_344.onnx")
input = sess.get_inputs()[0]
output = sess.get_outputs()[0]
print('Input : ',input.name,' Shape: ',input.shape)
print('Output: ',output.name,' Shape: ',output.shape)
inputImg = Image.open('CHANGEWITHYOURSOURCEIMAGE')
newWidth = input.shape[3]
newHeight = input.shape[2]
if inputImg.height < inputImg.width:
   newHeight = input.shape[2]
   newWidth = input.shape[2] * inputImg.width / inputImg.height
else:
   newWidth = input.shape[3]
   newHeight = input.shape[3] * inputImg.height / inputImg.width

inputImg = inputImg.resize((int(newWidth),int(newHeight)),Image.BICUBIC)
deltaX = newWidth - input.shape[3]
deltaY = newHeight - input.shape[2]
if deltaX > 0 or deltaY > 0:
  leftX = int(deltaX / 2)
  rightX = deltaX - leftX
  topY = int(deltaY /2)
  bottomY = deltaY - topY
  inputImg = inputImg.crop((leftX,topY,newWidth-rightX,newHeight-bottomY))

inputArray = np.asarray(inputImg)
inputArray = inputArray.astype(np.float32)
inputArray = (inputArray-127.5)/128.0
inputArray = inputArray.transpose([2, 0, 1])
inputArray = inputArray.reshape((1,3,input.shape[2],input.shape[3]))
pred_res = sess.run(None, {input.name: inputArray})[0]
result = softmax(pred_res)
maxval = np.amax(result,axis=1)
idx = np.zeros((pred_res.shape[2],pred_res.shape[3]),dtype=np.uint8)
for n in range(pred_res.shape[1]):
   for y in range(pred_res.shape[2]):
      for x in range(pred_res.shape[3]):
         if (maxval[0][y][x] == result[0][n][y][x]):
              idx[y][x] = n

rgbImage = convert_to_rgb_label(idx)
im = Image.fromarray(rgbImage)
im.save("demo.png")
