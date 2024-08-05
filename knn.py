import samples
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn import decomposition
from statistics import mean, stdev 
import random
import time
import argparse

DATUM_WIDTH_DIGIT=28
DATUM_HEIGHT_DIGIT=28
DATUM_WIDTH_FACE=60
DATUM_HEIGHT_FACE=70

def digit_knn():
    trainItems = samples.loadDataFile("data/digitdata/trainimages",5000,DATUM_WIDTH_DIGIT,DATUM_HEIGHT_DIGIT)
    trainLabels = samples.loadLabelsFile("data/digitdata/trainlabels",5000)
    testItems = samples.loadDataFile("data/digitdata/testimages",1000,DATUM_WIDTH_DIGIT,DATUM_HEIGHT_DIGIT)
    testLabels = samples.loadLabelsFile("data/digitdata/testlabels",1000)
    loopCount = 0
    for i in range(10):
        loopCount += 1
        #Defining the model
        model = KNeighborsClassifier(n_neighbors=3)
        samplesLen = loopCount*int(len(trainItems)/10)
        accuracy = []
        timeCount = []
        for j in range(5):
            trainLabelsNew = []
            feature = []
            featureTest = []
            sampleList = random.sample(range(0, len(trainItems)), samplesLen)
            
            for l in sampleList:
                feature.append(trainItems[l].getPixels())  
                trainLabelsNew.append(trainLabels[l])
            for l in range(len(testItems)):
                featureTest.append(testItems[l].getPixels())  

            featureTrain = np.array(feature)
            featureTest = np.array(featureTest)
            nsamples, nx, ny = featureTrain.shape
            d2TrainDataset = featureTrain.reshape((nsamples,nx*ny))

            nsamplestest, nxtest, nytest = featureTest.shape
            d2TestDataset = featureTest.reshape((nsamplestest,nxtest*nytest))
 
            start = time.time()
            model.fit(d2TrainDataset, trainLabelsNew)
            end = time.time()
            timeCount.append(end - start)
            predictedOutputDigit = model.predict(d2TestDataset)

            #Calculating accuracy
            count = 0 
            for k in range(len(testLabels)):
                if predictedOutputDigit[k] == testLabels[k]:
                    count += 1
            accuracy.append((count/len(testLabels))*100)
        #print(accuracy)
        accuracyMean = mean(accuracy)
        accuracyStdev = stdev(accuracy)
        timeMean = mean(timeCount)
        print("Accuracy at ", loopCount*10 ,"% :", accuracyMean)    
        print("Accuracy standard deviation at ", loopCount*10 ,"% :", accuracyStdev)  
        print("Time to train at ", loopCount*10 ,"% :", timeMean)
        print("\n")

def face_knn():
    trainItemsFace = samples.loadDataFile("data/facedata/trainimages",450,DATUM_WIDTH_FACE,DATUM_HEIGHT_FACE)
    trainLabelsFace = samples.loadLabelsFile("data/facedata/trainlabels",450)
    testItemsFace = samples.loadDataFile("data/facedata/testimages",150,DATUM_WIDTH_FACE,DATUM_HEIGHT_FACE)
    testLabelsFace = samples.loadLabelsFile("data/facedata/testlabels",150)
    validationData = samples.loadDataFile("data/facedata/validationimages", 300,DATUM_WIDTH_FACE,DATUM_HEIGHT_FACE)
    validationLabels = samples.loadLabelsFile("data/facedata/validationlabels", 300)
    
    trainItemsFace.extend(validationData)
    trainLabelsFace.extend(validationLabels)
    
    print(len(trainItemsFace))
    print(len(trainLabelsFace))
    
    loopCount = 0
    for i in range(10):
        loopCount += 1
        #Defining the model
        model = KNeighborsClassifier(n_neighbors=11, metric = 'braycurtis')
        samplesLen = loopCount*int(len(trainItemsFace)/10)
        accuracy = []
        timeCount = []
        for j in range(5):
            trainLabelsNew = []
            featureFace = []
            featureTestFace = []
            sampleList = random.sample(range(0, len(trainItemsFace)), samplesLen)
            
            for l in sampleList:
                featureFace.append(trainItemsFace[l].getPixels())  
                trainLabelsNew.append(trainLabelsFace[l])
            for l in range(len(testItemsFace)):
                featureTestFace.append(testItemsFace[l].getPixels())                             
             
            featureTrainFace = np.array(featureFace)
            
            featureTestFace = np.array(featureTestFace)

            nsamples, nx, ny = featureTrainFace.shape
            d2TrainDatasetFace = featureTrainFace.reshape((nsamples,nx*ny))

            nsamplestest, nxtest, nytest = featureTestFace.shape
            d2TestDataset = featureTestFace.reshape((nsamplestest,nxtest*nytest))

            start = time.time()
            model.fit(d2TrainDatasetFace, trainLabelsNew)
            end = time.time()
            timeCount.append(end - start)
            predictedOutputFace = model.predict(d2TestDataset)
                                       
            #Calculating accuracy
            count = 0 
            for k in range(len(testLabelsFace)):
                if predictedOutputFace[k] == testLabelsFace[k]:
                    count += 1
            accuracy.append((count/len(testLabelsFace))*100)
        #print(accuracy)
        accuracyMean = mean(accuracy)
        accuracyStdev = stdev(accuracy)
        timeMean = mean(timeCount)
        print("Accuracy at ", loopCount*10 ,"% :", accuracyMean)    
        print("Accuracy standard deviation at ", loopCount*10 ,"% :", accuracyStdev)  
        print("Time to train at ", loopCount*10 ,"% :", timeMean)
        print("\n")
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Get the input type")
    parser.add_argument('--file_type', type=str, default="face")
    args = parser.parse_args()
    file_type = args.file_type
    
    if file_type == "face":
        face_knn()
    else:
        digit_knn()