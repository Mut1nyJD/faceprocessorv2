import io
import os
import subprocess
import shutil
from PIL import Image,ImageDraw,ImageFont
import base64
import numpy as np
import onnxruntime as rt
import math
import cv2

def sigmoid(x):
   return 1./ (1. + math.exp(-x))

anchorsLevel = np.array([[[116,90],[156,198],[373,326]],
                         [[30,61],[62,45],[59,119]],
                         [[10,13],[16,30],[33,23]]])

scale = np.array([32,16,8])

detThreshold = 0.5
IoUThreshold = 0.5

session = rt.InferenceSession("YOURMUT1NYDETMODEL.onnx")

inputH = session.get_inputs()
outputH = session.get_outputs()

for n in range(len(inputH)):
   print('Input',n,inputH[n].name,' Shape: ',inputH[n].shape)

for n in range(len(outputH)):
   print('Output',n,outputH[n].name,' Shape: ',outputH[n].shape)

def non_max_suppression_fast(boxes, overlapThresh):
	# if there are no boxes, return an empty list
	if len(boxes) == 0:
		return []
	# if the bounding boxes integers, convert them to floats --
	# this is important since we'll be doing a bunch of divisions
	if boxes.dtype.kind == "i":
		boxes = boxes.astype("float")
	# initialize the list of picked indexes	
	pick = []
	# grab the coordinates of the bounding boxes
	x1 = boxes[:,0]
	y1 = boxes[:,1]
	x2 = boxes[:,2]
	y2 = boxes[:,3]
	scores = boxes[:,4]
	# compute the area of the bounding boxes and sort the bounding
	# boxes by the bottom-right y-coordinate of the bounding box
	area = (x2 - x1 + 1) * (y2 - y1 + 1)
	idxs = np.argsort(scores)
	# keep looping while some indexes still remain in the indexes
	# list
	while len(idxs) > 0:
		# grab the last index in the indexes list and add the
		# index value to the list of picked indexes
		last = len(idxs) - 1
		i = idxs[last]
		pick.append(i)
		# find the largest (x, y) coordinates for the start of
		# the bounding box and the smallest (x, y) coordinates
		# for the end of the bounding box
		xx1 = np.maximum(x1[i], x1[idxs[:last]])
		yy1 = np.maximum(y1[i], y1[idxs[:last]])
		xx2 = np.minimum(x2[i], x2[idxs[:last]])
		yy2 = np.minimum(y2[i], y2[idxs[:last]])
		# compute the width and height of the bounding box
		w = np.maximum(0, xx2 - xx1 + 1)
		h = np.maximum(0, yy2 - yy1 + 1)
		# compute the ratio of overlap
		overlap = (w * h) / area[idxs[:last]]
		# delete all indexes from the index list that have
		idxs = np.delete(idxs, np.concatenate(([last],
			np.where(overlap > overlapThresh)[0])))
	# return only the bounding boxes that were picked using the
	# integer data type
	return boxes[pick]

def doScoring(pred_res,threshold,IoUThresh):
   numboxes = pred_res[0].shape[1]
   numClasses = pred_res[0].shape[4] - 5
   detects = []
   for nlevel in range(3):
       curLevel = pred_res[nlevel][0]
       for box in range(numboxes):
          for gridy in range(curLevel.shape[1]):
             for gridx in range(curLevel.shape[2]):
                bx = (sigmoid(curLevel[box][gridy][gridx][0]) * 2. - 0.5 + gridx) * scale[nlevel]
                by = (sigmoid(curLevel[box][gridy][gridx][1]) * 2. - 0.5 + gridy) * scale[nlevel]
                bw = (sigmoid(curLevel[box][gridy][gridx][2]) * 2) ** 2 * anchorsLevel[nlevel][box][0]
                bh = (sigmoid(curLevel[box][gridy][gridx][3]) * 2) ** 2 * anchorsLevel[nlevel][box][1]
                objectness = sigmoid(curLevel[box][gridy][gridx][4])
                class_scrs = curLevel[box][gridy][gridx][5:]
                for n in range(class_scrs.shape[0]):
                   class_scrs[n] = sigmoid(class_scrs[n])
                max_cls_score = np.amax(class_scrs)

                if (objectness * max_cls_score > threshold):
                   class_idx = -1
                   for n in range(class_scrs.shape[0]):
                       if (max_cls_score == class_scrs[n]):
                          class_idx = n
                   detects.append(np.array([bx-bw/2,by-bh/2,bx+bw/2,by+bh/2,objectness*max_cls_score]))
   boxes = np.array(detects)
   boxes = non_max_suppression_fast(boxes,IoUThresh)
   return boxes

def doDetection(srcImg):
   srcWidth = srcImg.width
   srcHeight = srcImg.height
   newWidth = inputH[0].shape[3]
   newHeight = inputH[0].shape[2]
   if srcImg.height < srcImg.width:
      newWidth = inputH[0].shape[3]
      newHeight = inputH[0].shape[3] * srcImg.height / srcImg.width
   else:
      newHeight = inputH[0].shape[2]
      newWidth = inputH[0].shape[2] * srcImg.width / srcImg.height
   srcImgR = srcImg.resize((int(newWidth),int(newHeight)),Image.BICUBIC)
   outImg = Image.new('RGB',(inputH[0].shape[3],inputH[0].shape[2]),(0,0,0))
   
   padX = (inputH[0].shape[3] - newWidth) / 2
   padY = (inputH[0].shape[2] - newHeight) / 2
   xScale = srcWidth / newWidth
   yScale = srcHeight / newHeight
   outImg.paste(srcImgR,(int(padX),int(padY)))
#   outImg.save("todetect.png")
#   inputArray = cv2.imread("todetect.png")
   inputArray = np.asarray(outImg)
   inputArray = inputArray.astype(np.float32)
   inputArray = (inputArray/255.0)
   inputArray = inputArray.transpose([2,0,1])
   inputArray = inputArray.reshape((1,inputH[0].shape[1],inputH[0].shape[2],inputH[0].shape[3]))
   pred_res = session.run(None,{inputH[0].name: inputArray})
   detects = doScoring(pred_res,detThreshold,IoUThreshold)
   detects[:,0] = (detects[:,0] - padX) * xScale
   detects[:,1] = (detects[:,1] - padY) * yScale
   detects[:,2] = (detects[:,2] - padX) * xScale
   detects[:,3] = (detects[:,3] - padY) * yScale
   
   print('# dets = ',detects.shape[0])
   return detects

img = Image.open("YOURTESTIMAGE.png")
boxes = doDetection(img)
