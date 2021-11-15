import cv2
import imutil
import time
import mahotas as mh
import joblib
import numpy as np


def main():
    # load model
    model = joblib.load('model.pkl')
    
    mirror=True
    cam = cv2.VideoCapture(0)
    
    pTime = 0
    while True:
        # read webcam
        ret_val, img = cam.read()

        # resize original image for displaying
        display = imutil.resize(img.copy(), width = 128, height = 128)

        # flip image if mirrored
        if mirror: 
            img = cv2.flip(img, 1)

        # resize image for processing
        # process image like in trainset
        img = imutil.resize(img, width = 128, height = 128)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (3, 3), 0)

        T= mh.thresholding.otsu(blurred)
        gray[gray>T] = 255
        gray[gray<255] = 0

        data = np.reshape(gray, (1, 128*128))  
        
        result = model.predict(data)

        # label: paper 0 rock 1 scissors 2
        if result[0] == 0:
            f = 'paper'
        elif result[0] == 1:
            f = 'rock'
        else:
            f = 'scissors'

        # frame rate
        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime
        cv2.putText(display, f'FPS:{int(fps)}', (10, 10), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 255, 0), 1)

        # show the results on the image and display
        cv2.putText(display, f, (46, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), thickness= 1)

        cv2.imshow('my webcam', display)
        if cv2.waitKey(1) == 27: 
            break  # esc to quit
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()