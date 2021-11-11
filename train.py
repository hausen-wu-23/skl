from numpy.core.fromnumeric import size
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import plot_confusion_matrix
import joblib
import numpy as np
from matplotlib import pyplot
import os
import random
import seaborn as sns
import cv2
import mahotas as mh
import imutils

# directory of training data
train_dir = './rock-paper-scissor/rps/rps'
test_dir = './rock-paper-scissor/rps-test-set/rps-test-set'

# function loads the image and rescale and thresholds it for maximum accuracy
def loadImg(dir):
    img = cv2.imread(dir)
    img = imutils.resize(img, width = 128)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)

    T= mh.thresholding.otsu(blurred)
    gray[gray>T] = 255
    gray[gray<255] = 0
    data = np.reshape(gray, (128*128))  
    return data

def main():
    # 840 images per train set, 2520 total
    train_rock = os.listdir(train_dir + '/rock')
    train_paper = os.listdir(train_dir + '/paper')
    train_scissors = os.listdir(train_dir + '/scissors')

    # 124 images per test set, 372 total
    test_rock = os.listdir(test_dir + '/rock')
    test_paper = os.listdir(test_dir + '/paper')
    test_scissors = os.listdir(test_dir + '/scissors')


    # show random sample data
    x = random.randint(0, len(train_rock))
    cv2.imshow('rock', cv2.imread(train_dir + '/rock/' + train_rock[x]))
    cv2.imshow('paper', cv2.imread(train_dir + '/paper/' + train_paper[x]))
    cv2.imshow('scissors', cv2.imread(train_dir + '/scissors/' + train_scissors[x]))
    cv2.waitKey(0)

    # label: paper 0 rock 1 scissors 2

    # processing image for train set
    trainX = np.ndarray(shape = (2520,128*128), dtype=np.uint8)
    trainY = np.ndarray(shape = (2520, 1), dtype=np.uint8)

    # index of the the current importing images index 
    cur_index = 0
    for i in range(len(train_paper)):
        dir = train_dir + '/paper/' + train_paper[i]
        trainX[cur_index] = loadImg(dir)
        trainY[cur_index] = 0
        cur_index += 1
        print('\rProcessed %s%% of the train set.' % int((cur_index/2520)*100), end='')
    
    for i in range(len(train_rock)):
        dir = train_dir + '/rock/' + train_rock[i]
        trainX[cur_index] = loadImg(dir)
        trainY[cur_index] = 1
        cur_index += 1
        print('\rProcessed %s%% of the train set.' % int((cur_index/2520)*100), end='')

    for i in range(len(train_rock)):
        dir = train_dir + '/scissors/' + train_scissors[i]
        trainX[cur_index] = loadImg(dir)
        trainY[cur_index] = 2
        cur_index += 1
        print('\rProcessed %s%% of the train set.' % int((cur_index/2520)*100), end='')
    trainY = np.reshape(trainY, (2520))
    print('\nFinished processing train set.')

    # importing image to create the test set
    testX = np.ndarray(shape = (372,128*128), dtype=np.uint8)
    testY = np.ndarray(shape = (372, 1), dtype=np.uint8)
    
    # index of the the current importing images index 
    cur_index = 0

    for i in range(len(test_paper)):
        dir = test_dir + '/paper/' + test_paper[i]
        testX[cur_index] = loadImg(dir)
        testY[cur_index] = 0
        cur_index += 1
        print('\rProcessed %s%% of the test set.' % int((cur_index/372)*100), end='')
    
    for i in range(len(test_rock)):
        dir = test_dir + '/rock/' + test_rock[i]
        testX[cur_index] = loadImg(dir)
        testY[cur_index] = 1
        cur_index += 1
        print('\rProcessed %s%% of the test set.' % int((cur_index/372)*100), end='')

    for i in range(len(test_rock)):
        dir = test_dir + '/scissors/' + test_scissors[i]
        testX[cur_index] = loadImg(dir)
        testY[cur_index] = 2
        cur_index += 1
        print('\rProcessed %s%% of the test set.' % int((cur_index/372)*100), end='')
    testY = np.reshape(testY, (372))
    print('\nFinished processing test set.')

    print('start training...')
    # using MLP for the highest accuracy
    classifier = MLPClassifier(hidden_layer_sizes=(256,128,64,32), activation="relu", random_state=1, max_iter=10000)
    classifier.fit(trainX, trainY)
    preds = classifier.predict(testX)

    # validation of the accuracy of the model
    correct = 0
    incorrect = 0
    for pred, gt in zip(preds, testY):
        if pred == gt: correct += 1
        else: incorrect += 1
    print(f'Correct: {correct}, Incorrect: {incorrect}, % Correct: {correct/(correct + incorrect): 5.2}')

    plot_confusion_matrix(classifier, testX, testY)
    pyplot.show()

    # saving the model
    joblib_file = 'model.pkl'
    joblib.dump(classifier, joblib_file)

if __name__ == '__main__':
    main()