from flask import Flask, request, render_template
import pickle
import pandas as pd
import numpy as np
import cv2
import keras
from keras.models import load_model,save_model
from skimage.io import imread, imshow
from skimage.transform import resize
from skimage.feature import hog
from PIL import Image
from skimage import transform

app = Flask(__name__)
model = keras.models.load_model('C:/Users/umara/OneDrive/Desktop/Covid/my_model')

name = ['Normal','Covid']
def load(filename):
    img = cv2.imread(filename)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    image = cv2.resize(gray, (150,150), interpolation=cv2.INTER_CUBIC)
    image = image.reshape(1, 150, 150, 1)
    image = image.astype('float32')
    image /= 255
    return image

@app.route('/',methods=['GET'])
def hello_world():
    return render_template('index.html')

@app.route('/',methods=['POST'])
def predict():
    imagefile = request.files['imagefile']
    imagepath = "C:/Users/umara/OneDrive/Desktop/Covid/images/" + imagefile.filename
    image = load(imagepath)
    y_pred = model.predict(image)


    return render_template('index.html',prediction=name[np.argmax(y_pred)])

if __name__ == '__main__':
    app.run(port=3000,debug=True)
