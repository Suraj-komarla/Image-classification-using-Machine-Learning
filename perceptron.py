import numpy as np
from time import time
from sklearn.metrics import accuracy_score
from statistics import stdev
from data_manuplation import *

FACE_IMAGES = (451, 150)
DIGIT_IMAGES = (5000, 1000)
FACE_LABELS = 2
DIGIT_LABELS = 10

def train(training_data, training_labels, file_type):
    labels = FACE_LABELS if file_type == "face" else DIGIT_LABELS
    
    weights = {key:np.zeros(training_data[0].shape) for key in range(labels)}
    labels = [e for e in range(labels)]
    
    for epoch in range(2):
        for idx, image in enumerate(training_data):
            max_score = 0
            max_ind = 0
            
            for label in labels:
                score = np.tensordot(weights[label], image)
                if score > max_score:
                    max_score = score
                    max_ind = label

            if max_ind != training_labels[idx]:
                weights[training_labels[idx]] = np.add(weights[training_labels[idx]], image)
                weights[max_ind] = np.subtract(weights[max_ind], image)

    return weights

def classify(testing_data, weights, file_type):
    labels = FACE_LABELS if file_type == "face" else DIGIT_LABELS
    labels = [e for e in range(labels)]
    predictions = []
    for data in testing_data:
        max_score = 0
        max_ind = 0
        for label in labels:
            score = np.tensordot(weights[label], data)
            if score > max_score:
                max_score = score
                max_ind = label
        predictions.append(max_ind)
        
    return predictions

if __name__ == "__main__":
    # import pdb;pdb.set_trace()
    file_type = get_file_args()
    train_dir, test_dir = get_dir(file_type)
    train_images, test_images = DIGIT_IMAGES if file_type == "digit" else FACE_IMAGES
    train_labels_dir, test_labels_dir = get_dir_labels(file_type)
    
    testing_img, test_labels = read_image(test_images, test_dir, test_labels_dir, file_type, train_flag=False)
    testing_img = np.array(testing_img)
    
    
    for i in range(1, 11):
        acc = []
        tim = []
        for j in range(5):
            images = int(i * train_images / 10)
            start_time = time()
            input_img, train_labels = read_image(images, train_dir, train_labels_dir, file_type)
                        
            weights = train(input_img, train_labels, file_type)
            predictions = classify(testing_img, weights, file_type)
            
            end_time = time()       
            acc.append(accuracy_score(predictions, test_labels) * 100)
            tim.append(end_time - start_time)

        print("Images=",images, "Time=",np.average(tim), "Accuracy=",np.average(acc), "Standard Dev=",stdev(acc))

    
    