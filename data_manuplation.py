import os
import argparse
import sys
from random import randint
import numpy as np

FACE_IMAGES = (451, 150)
DIGIT_IMAGES = (5000, 1000)

FACE_DIM = (70, 60)
DIGIT_DIM = (28, 28)

def get_image(img):
    key_dict = {' ': 0, '+': 1, '#': 2}
    for idx, line in enumerate(img):
        line = list(line)
        line = list(map(lambda x: key_dict[x], line))
        img[idx] = line
    return img

def read_image(images, input_dir, labels_dir, file_type, train_flag=True): 
    img = []
    height, width = FACE_DIM if file_type == "face" else DIGIT_DIM
    train, test = FACE_IMAGES if file_type == "face" else DIGIT_IMAGES
    if train_flag:
        images = [randint(1, train - 1) for step in range(images)]
    else:
        images = list(range(1, images + 1))
    
    for i in images:
        file_name = "image " + str(i) + ".txt"
        file = os.path.join(input_dir, file_name)
        with open(file, "r") as f:
            temp_file = list(filter(None, f.read().splitlines()))
            image = get_image(temp_file)
            if len(image) != height:
                image.append([0] * width)
            img.append(image)

    with open(labels_dir, "r") as f:
        labels = f.read().splitlines()
        labels = [int(labels[ind - 1]) for ind in images]
        
    img = np.array(img)
    return img, labels

def get_file_args():
    parser = argparse.ArgumentParser(description="Get the input type")
    parser.add_argument('--data_type', type=str, default="digit")
    args = parser.parse_args()
    file_type = args.data_type
    
    if file_type != "face" and file_type != "digit":
        print("Invalid type")
        sys.exit()
    
    return file_type

def get_dir(type):
    dir = os.path.join(os.getcwd(), "data", "processed_" + type)
    train_dir = os.path.join(dir, "train")
    test_dir = os.path.join(dir, "test")
    return train_dir, test_dir

def get_dir_labels(type):
    train_labels = os.path.join(os.getcwd(), "data", type + "data", "trainlabels")
    test_labels = os.path.join(os.getcwd(), "data", type + "data", "testlabels")
    return train_labels, test_labels