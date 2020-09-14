import torch
import base64
import json
from ResponseModel import ResponseModel
import torch.nn as nn
from flask import Flask
from flask import request,jsonify
from PIL import Image
from io import BytesIO
from models import CNN
from torchvision.transforms import Compose, ToTensor, Resize
from collections import OrderedDict

app = Flask(__name__)

#release
model_path = '/home/VerificationDemo/checkpoints/model.pth'
# debug
# model_path = './checkpoints/model.pth'
source = [str(i) for i in range(0, 10)]
source += [chr(i) for i in range(97, 97+26)]
alphabet = ''.join(source)
@app.route('/api/predict', methods=['POST'])
def post_predict():
  text="failure"
  result=ResponseModel(0,'success','0000')
  try:
      if request.method == 'POST':
        file = request.get_data()
        
        ctype = request.headers.get('Content-Type',None)
        if ctype != 'image/png':
            j_data = json.loads(file)
            if(len(j_data) != 0):
                image_base64=j_data['image_base64']
                if(len(image_base64)!=0):
                    file = base64.b64decode(image_base64)
                else:
                  result.code=1
                  result.message='[image_base64] must not be empty!'
            else:
                result.code=1
                result.message='Request body must not be empty!'

        img = Image.open(BytesIO(file)).convert('RGB')
        print('宽：%d,高：%d'%(img.size[0],img.size[1]))
        width=img.size[0]
        height=img.size[1]
        transform = Compose([Resize(height, width), ToTensor()])
        img = transform(img)
        cnn = CNN()
        if torch.cuda.is_available():
          cnn = cnn.cuda()
        cnn.eval()
        cnn.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
        # GPU
        # img = img.view(1, 3, height, width).cuda()
        # CPU
        img = img.view(1, 3, height, width)
        output = cnn(img)
        output = output.view(-1, 36)
        output = nn.functional.softmax(output, dim=1)
        output = torch.argmax(output, dim=1)
        output = output.view(-1, 4)[0]
        text=''.join([alphabet[i] for i in output.cpu().numpy()])
        result.result=text

  except Exception as ex:
      result.code=1
      result.message=ex.message


  return json.dumps(result,default=lambda obj: obj.__dict__ ,sort_keys=True,indent=4)

@app.route('/api/values')
def index():
  return "Hello, World!"

if __name__ == '__main__':
  #release
  app.run(host='0.0.0.0',port='80') 
  # debug
  # app.run(host='127.0.0.1',port='80') 