
import cv2
import imutils
import mahotas as mh
import numpy as np
import argparse
import joblib

# function loads the image and rescale and thresholds it for maximum accuracy
def loadImg(dir):
    img = cv2.imread(dir)
    img = imutils.resize(img, width = 128, height = 128)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)

    T= mh.thresholding.otsu(blurred)
    gray[gray>T] = 255
    gray[gray<255] = 0
    data = np.reshape(gray, (1, 128*128))  
    return data

# loading the model
model = joblib.load('model.pkl')

# loading the image
parse = argparse.ArgumentParser()
parse.add_argument('-i', '--image', required=True, help='path to image')
args = vars(parse.parse_args())

# process image to be passed through the model
img = loadImg(args['image'])
result = model.predict(img)

# read original image for the purpose of displaying the results
original = cv2.imread(args['image'])

# label: paper 0 rock 1 scissors 2
if result[0] == 0:
    f = 'paper'
elif result[0] == 0:
    f = 'rock'
else:
    f = 'scissors'

# show the results on the image and display
cv2.putText(original, f, (0, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), thickness= 2)
cv2.imshow('result', original)
cv2.waitKey(0)