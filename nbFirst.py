import util
import samples
import os
import numpy as np
import math
import difflib

FACE_DATUM_WIDTH=60
DIGIT_DATUM_WIDTH=28
DIGIT_DATUM_HEIGHT=28
FACE_DATUM_HEIGHT=70

def testLabel(featureTestLabels,featurePercDict,labelsDict,datatype):
	if(datatype == "digit"):
		labels = [0,1,2,3,4,5,6,7,8,9]
	else:
		labels =[0,1]
	percList = [0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]
	predictedArr = []
	#print(featureTestLabels)
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
	#print(predictedArr)
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
		#print(len(featureConsolidated[key]))
		t = 1
		for f in featureConsolidated[key]:
			t+=1
			fArr = np.array(featureConsolidated[key][f])
			n = len(fArr)
			#print(n)
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
	#print(a)
	b = np.array(a)
	if(datatype == "face"):
		c = b.reshape(100,7,6)
	else:
		c = b.reshape(49,4,4)
	#print(c)
	#print(label)
	#global featureLabels
	featuresArray = []
	for feature in c:
		pixelCount = 0
	for x in range(len(feature)):
		for y in range(len(feature[0])):
			pixelCount += feature[x][y]
	featurePercentage = pixelCount/(len(feature)*len(feature[0]))
	featuresArray.append(featurePercentage)
	#print(featuresArray)
	if label in featureLabels:
		featureLabels[label].append(featuresArray)
	else:
		featureLabels[label] = [featuresArray]
  	
	return featureLabels


if __name__ == '__main__':
	print(round(0.49857142857142855,1))
	featureLabels = util.Counter()
	predictedLabels = []
	items = samples.loadDataFile("data/digitdata/trainingimages",5000,DIGIT_DATUM_WIDTH,DIGIT_DATUM_HEIGHT)
	labels = samples.loadLabelsFile("data/digitdata/traininglabels",5000)
	rawValidationData = samples.loadDataFile("data/digitdata/validationimages", 1000,DIGIT_DATUM_WIDTH,DIGIT_DATUM_HEIGHT)
	validationLabels = samples.loadLabelsFile("data/digitdata/validationlabels", 1000)
	testItems = samples.loadDataFile("data/digitdata/testimages",1000,DIGIT_DATUM_WIDTH,DIGIT_DATUM_HEIGHT)
	testLabels = samples.loadLabelsFile("data/digitdata/testlabels",1000)
	for i in range(5000):
		featureLabels = basicFeatureExtractorDigit(items[i],labels[i],featureLabels,"digit")
	for u in range(1000):
		#print(u)
		featureLabels = basicFeatureExtractorDigit(rawValidationData[u],validationLabels[u],featureLabels,"digit")
	"""for key in featureLabels:
		print(key)
		print(featureLabels[key])"""
	#print(featureLabels)
	import pdb;pdb.set_trace()
	featureConsolidated,featurePercDict,labelsDict = featureConsolidator(labels,featureLabels)
	"""for key in featurePercDict:
		print(key)
		print(featurePercDict[key])
		print(len(featurePercDict[key]))"""
	"""for val in labelsDict:
		print(val)
		print(labelsDict[val])"""
	for j in range(1000):
		featureTestLabels = basicFeatureExtractorDigit(testItems[j],testLabels[j],{},"digit")
		"""for val in featureTestLabels:
			print(val)
			print(featureTestLabels[val])"""
		predictedLabel = testLabel(featureTestLabels[testLabels[j]][0],featurePercDict,labelsDict,"digit")
		predictedLabels.append(predictedLabel)
	#print(testLabels)
	#print(predictedLabels)
	import pdb;pdb.set_trace()
	count = 0
	for o in range(1000):
		if(predictedLabels[o] == testLabels[o]):
			count += 1
	print(count)
	print(count/1000)

	faceFeatureLabels = util.Counter()
	facePredictedLabels = []
	faceItems = samples.loadDataFile("data/facedata/trainingimages",450,FACE_DATUM_WIDTH,FACE_DATUM_HEIGHT)
	faceLabels = samples.loadLabelsFile("data/facedata/traininglabels",450)
	faceRawValidationData = samples.loadDataFile("data/facedata/validationimages", 300,FACE_DATUM_WIDTH,FACE_DATUM_HEIGHT)
	faceValidationLabels = samples.loadLabelsFile("data/facedata/validationlabels", 300)
	faceTestItems = samples.loadDataFile("data/facedata/testimages",150,FACE_DATUM_WIDTH,FACE_DATUM_HEIGHT)
	faceTestLabels = samples.loadLabelsFile("data/facedata/testlabels",150)
	
	for i in range(450):
		faceFeatureLabels = basicFeatureExtractorDigit(faceItems[i],faceLabels[i],faceFeatureLabels,"face")
	for u in range(300):
		faceFeatureLabels = basicFeatureExtractorDigit(faceRawValidationData[u],faceValidationLabels[u],faceFeatureLabels,"face")

	faceFeatureConsolidated,faceFeaturePercDict,faceLabelsDict = featureConsolidator(faceLabels,faceFeatureLabels)

	for j in range(150):
		faceFeatureTestLabels = basicFeatureExtractorDigit(faceTestItems[j],faceTestLabels[j],{},"face")
		facePredictedLabel = testLabel(faceFeatureTestLabels[faceTestLabels[j]][0],faceFeaturePercDict,faceLabelsDict,"face")
		facePredictedLabels.append(facePredictedLabel)

	faceCount = 0
	for o in range(150):
		if(facePredictedLabels[o] == faceTestLabels[o]):
			faceCount += 1
	print(facePredictedLabels)
	print(faceCount)
	print(faceCount/150)