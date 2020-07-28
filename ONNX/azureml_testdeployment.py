import json
import numpy as np
import requests
from PIL import Image

inputImg = Image.open('YOURIMAGE')
inputArray = np.asarray(inputImg)
input_data = json.dumps({'data' : inputArray.tolist()})
headers = {'Content-Type':'application/json'}
score_url = "ENTER YOUR ENDPOINT"
resp = requests.post(score_url,input_data,headers=headers)

try:
  result = json.loads(resp.text)['result']
  print(result)
except:
  error = json.loads(resp.text)['error']
  print(error)


