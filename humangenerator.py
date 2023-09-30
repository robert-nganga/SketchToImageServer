from keras.models import load_model
from flask import jsonify, request
import base64
import io
from keras.preprocessing.image import img_to_array, save_img
import numpy as np
#import tensorflow as tf
from PIL import Image
from flask_restful import Resource, Api, reqparse
from flask import Flask
from pathlib import Path
import requests
from tqdm import tqdm
import os

application = Flask(__name__)
api = Api(application)

# Dropbox direct download URL
url = "https://www.dropbox.com/s/yp80qcyyyq0j9ns/fully_trained.h5?dl=1"

# Destination file name
destination_file = "fully_trained.h5"

# Check if the destination file already exists
if os.path.exists(destination_file):
    # Get the file size of the existing file
    existing_file_size = os.path.getsize(destination_file)
else:
    existing_file_size = 0

# Send a GET request to the URL with a 'Range' header to resume the download
headers = {'Range': f'bytes={existing_file_size}-'}

response = requests.get(url, headers=headers, stream=True)

if response.status_code == 206:
    total_size = int(response.headers.get("content-length", existing_file_size))
    progress_bar = tqdm(total=total_size, initial=existing_file_size, unit="B", unit_scale=True)
    with open(destination_file, 'ab') as f:
        for chunk in response.iter_content(8192):
            f.write(chunk)
            progress_bar.update(len(chunk))
    progress_bar.close()
    print(f"Resumed download of {destination_file} successfully!")
else:
    print(f"Failed to resume the download. Status code: {response.status_code}")

#BASE_DIR = Path(__file__).resolve(strict=True).parent
model = load_model('fully_trained.h5')
print('model loaded')

@application.route('/')
def index():
    return 'Hello, Ready to start predicting!'

def prepare_image(image, target):
    if image.mode != "RGB":
        image = image.convert('RGB')
        
    image = image.resize(target)
    image = img_to_array(image)
    
    image = (image - 127.5) / 127.5
    image = np.expand_dims(image, axis=0)
    return image
    
    

class Predict(Resource):
    def post(self):
        json_data = request.get_json()
        img_data = json_data['image']
        
        image = base64.b64decode(str(img_data))
        img = Image.open(io.BytesIO(image))
        
        prepared_image = prepare_image(img, target=(256, 256))
        preds = model.predict(prepared_image)
        
        outputfile = 'output.png'
        savePath = './output/'
        
        #output = tf.reshape(preds, [256,256,3])
        output = preds.reshape(256,256,3)
        output = (output + 1)/2
        save_img(savePath+outputfile, img_to_array(output))
        
        imageNew = Image.open(savePath+outputfile)
        imageNew = imageNew.resize((50,50))
        imageNew.save(savePath+"new_"+outputfile)
        
        with open(savePath+"new_"+outputfile, 'rb') as image_file:
            encoded_string = base64.b64encode(image_file.read())
        
        outputdata = {'image': str(encoded_string)}
        return outputdata


api.add_resource(Predict, '/predict')

if __name__ == '__main__':
    application.run(debug=True, host='0.0.0.0', port=8080)