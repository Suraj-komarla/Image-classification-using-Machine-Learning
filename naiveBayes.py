import util
import samplesNaiveBayes as samples
import os
import numpy as np
import math
import random
import argparse
import time
import matplotlib.pyplot as plt

FACE_DATUM_WIDTH=60
DIGIT_DATUM_WIDTH=28
DIGIT_DATUM_HEIGHT=28
FACE_DATUM_HEIGHT=70

def trainAndTuneandTest(data_type,itemsTotal,labelsTotal,rawValidationDataTotal,validationLabelsTotal,testItems,testLabel,testCount,trainCount,validationCount):
	trainPerc = [10,20,30,40,50,60,70,80,90,100]
	mean =[]
	sd = []
	timeArr = []
	for trained in trainPerc:
		print("training with",trained,"percentage of data")
		digitTrain = round((trained*trainCount)/100)
		digitValidate = round((trained*validationCount)/100)
		featureLabels = util.Counter()
		acc = []
		avgTime = 0
		for i in range(5):
			predictedLabels = [] 
			idxDT = np.random.choice(np.arange(trainCount),digitTrain,replace=False)
			items = []
			labels = []
			for k in idxDT:
				items.append(itemsTotal[k])
				labels.append(labelsTotal[k])
			start = time.time()
			for i in range(digitTrain):
				featureLabels = basicFeatureExtractorDigit(items[i],labels[i],featureLabels,data_type)
			idxDV = np.random.choice(np.arange(validationCount),digitValidate,replace = False)
			rawValidationData = []
			validationLabels = []
			for l in idxDV:
				rawValidationData.append(rawValidationDataTotal[l])
				validationLabels.append(validationLabelsTotal[l])
			for u in range(digitValidate):
				featureLabels = basicFeatureExtractorDigit(rawValidationData[u],validationLabels[u],featureLabels,data_type)
			featureConsolidated,featurePercDict,labelsDict = featureConsolidator(labels,featureLabels)
			end = time.time()
			avgTime = avgTime + (end - start)
			#print(avgTime)
			for j in range(testCount):
				featureTestLabels = basicFeatureExtractorDigit(testItems[j],testLabels[j],{},data_type)
				predictedLabel = testLabel(featureTestLabels[testLabels[j]][0],featurePercDict,labelsDict,data_type)
				predictedLabels.append(predictedLabel)
			count = 0
			for o in range(testCount):
				if(predictedLabels[o] == testLabels[o]):
					count += 1
			acc.append(count/testCount)
		print(acc)
		accNp = np.array(acc)
		mean.append(np.mean(accNp))
		timeArr.append(avgTime/5)
		sd.append(np.std(accNp))
		print("mean of arr for random ", trained," percent of data is: ", np.mean(accNp))
		print("std of arr for random ", trained," percent of data is: ", np.std(accNp))
	return(mean,sd,timeArr)

def testLabel(featureTestLabels,featurePercDict,labelsDict,datatype):
	if(datatype == "digit"):
		labels = [0,1,2,3,4,5,6,7,8,9]
	else:
		labels =[0,1]
	percList = [0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]
	predictedArr = []
	for val in labels:
		prod = 1
		for i in range(len(featureTestLabels)):
			x = featureTestLabels[i]
			if(x != 0):
				x = round(featureTestLabels[i],1)
			y = percList.index(x)
			prod = prod * featurePercDict[val][i][y]
		prod = prod * (labelsDict[val]/5000)
		predictedArr.append(prod)
	ind = predictedArr.index(max(predictedArr))
	return labels[ind]


def featureConsolidator(labels,featureLabels):
	featureConsolidated = {}
	featurePerc = []
	featurePercDict = {}
	labelsDict = {}
	percList = [0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]
	for key in featureLabels:
		labelsDict[key] = len(featureLabels[key])
		featureConsolidated[key] = {}
		for img in featureLabels[key]:
			p = len(img)
			for i in range(p):
				if(str(key)+str(i+1) in featureConsolidated[key]):
					featureConsolidated[key][str(key)+str(i+1)].append(round(img[i],1))
				else:
					featureConsolidated[key][str(key)+str(i+1)] = [round(img[i],1)]
		t = 1
		for f in featureConsolidated[key]:
			t+=1
			fArr = np.array(featureConsolidated[key][f])
			n = len(fArr)
			percArr = []
			for y in percList:
				count = np.count_nonzero(fArr == y)
				perc = 0
				if(count == 0):
					perc = 0.001
				else:
					perc = round(count/n,2)
				percArr.append(perc)
			featurePerc.append(percArr)
		featurePercDict[key] = featurePerc
		featurePerc = []

	return featureConsolidated,featurePercDict, labelsDict

def basicFeatureExtractorDigit(datum,label,featureLabels,datatype):
  """
  Returns a set of pixel features indicating whether
  each pixel in the provided datum is white (0) or gray/black (1)
  """
  a = datum.getPixels()
  b = np.array(a)
  if(datatype == "face"):
  	c = b.reshape(100,7,6)
  else:
  	c = b.reshape(49,4,4)
  featuresArray = []
  for feature in c:
  	pixelCount = 0
  	for x in range(len(feature)):
  		for y in range(len(feature[0])):
  			pixelCount += feature[x][y]
  	featurePercentage = pixelCount/(len(feature)*len(feature[0]))
  	featuresArray.append(featurePercentage)
  if label in featureLabels:
  	featureLabels[label].append(featuresArray)
  else:
  	featureLabels[label] = [featuresArray]
  return featureLabels


if __name__ == '__main__':
	parser = argparse.ArgumentParser(description="Get the input type")
	parser.add_argument('--data_type', type=str, default=None)
	args = parser.parse_args()
	data_type = args.data_type
	print(data_type)
	if(data_type == "digit"):
		# import pdb;pdb.set_trace()
		itemsTotal = samples.loadDataFile("data/digitdata/trainimages",5000,DIGIT_DATUM_WIDTH,DIGIT_DATUM_HEIGHT)
		labelsTotal = samples.loadLabelsFile("data/digitdata/trainlabels",5000)
		rawValidationDataTotal = samples.loadDataFile("data/digitdata/validationimages", 1000,DIGIT_DATUM_WIDTH,DIGIT_DATUM_HEIGHT)
		validationLabelsTotal = samples.loadLabelsFile("data/digitdata/validationlabels", 1000)
		testItems = samples.loadDataFile("data/digitdata/testimages",1000,DIGIT_DATUM_WIDTH,DIGIT_DATUM_HEIGHT)
		testLabels = samples.loadLabelsFile("data/digitdata/testlabels",1000)
		trainCount = 5000
		validationCount = 1000
		testCount = 1000
	else:
		itemsTotal = samples.loadDataFile("data/facedata/trainimages",450,FACE_DATUM_WIDTH,FACE_DATUM_HEIGHT)
		labelsTotal = samples.loadLabelsFile("data/facedata/trainlabels",450)
		rawValidationDataTotal = samples.loadDataFile("data/facedata/validationimages", 300,FACE_DATUM_WIDTH,FACE_DATUM_HEIGHT)
		validationLabelsTotal = samples.loadLabelsFile("data/facedata/validationlabels", 300)
		testItems = samples.loadDataFile("data/facedata/testimages",150,FACE_DATUM_WIDTH,FACE_DATUM_HEIGHT)
		testLabels = samples.loadLabelsFile("data/facedata/testlabels",150)
		trainCount = 450
		validationCount = 300
		testCount = 150
	print("training and testing for ",data_type)
	mean,sd,timeArr = trainAndTuneandTest(data_type,itemsTotal,labelsTotal,rawValidationDataTotal,validationLabelsTotal,testItems,testLabel,testCount,trainCount,validationCount)
	print(timeArr)
	#trainPerc = [10,20,30,40,50,60,70,80,90,100]
	"""plt.plot(trainPerc,mean, label = "Naive bayes mean")
	plt.show()
	plt.plot(trainPerc,sd,label = "Naive bayes standard deviation")
	plt.show()"""