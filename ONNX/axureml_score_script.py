import json
import sys
import numpy as np
import onnxruntime as rt
from PIL import Image
import math
from azureml.core.model import Model

def softmax(x):
   e_x = np.exp(x - np.max(x))
   return e_x / e_x.sum()

def init():
   global model_path
   model_path = Model.get_model_path(model_name = "FaceSeg")

def run(raw_data):
   try:
     data = json.loads(raw_data)['data']
     data = np.array(data,dtype=np.uint8)
     img = Image.fromarray(data)
     session = rt.InferenceSession(model_path)
     input = session.get_inputs()[0]
     output = session.get_outputs()[0]
     newWidth = input.shape[3]
     newHeight = input.shape[2]
     if img.height < img.width:
       newHeight = input.shape[2]
       newWidth = input.shape[2] * img.width / img.height
     else:
       newWidth = input.shape[3]
       newHeight = input.shape[3] * img.height / img.width
     inputImg = img.resize((int(newWidth),int(newHeight)),Image.BICUBIC)
     deltaX = newWidth - input.shape[3]
     deltaY = newHeight - input.shape[2]
     if deltaX > 0 or deltaY > 0:
       leftX = int(deltaX / 2)
       rightX - deltaX - leftX
       topY = int(deltaY / 2)
       bottomY = deltaY - topY
       inputImg = inputImg.crop((leftX,topY,newWidth-rightX,newHeight-bottomY))
     inputArray = np.asarray(inputImg)
     inputArray = inputArray.astype(np.float32)
     inputArray = (inputArray - 127.5) / 128.0
     inputArray = inputArray.transpose([2,0,1])
     inputArray = inputArray.reshape((1,3,input.shape[2],input.shape[3]))
     pred_res = session.run(None,{input.name: inputArray})[0]
     result = softmax(pred_res)
     maxval = np.amax(result,axis=1)
     idx = np.zeros((pred_res.shape[2],pred_res.shape[3]),dtype=np.float32)
     for y in range(pred_res.shape[2]):
        for x in range(pred_res.shape[3]):
           for n in range(pred_res.shape[1]):
               if (maxval[0][y][x] = pred_res[0][n][y][x]):
                  idx[y][x] = n

     returnRes = idx.tolist()
     return {"result" : returnRes}
   except Exception as e:
      result = str(e)
      return {"error" : result}
