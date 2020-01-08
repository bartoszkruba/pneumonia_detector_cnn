import numpy as np
import cv2
from tqdm import tqdm
import os
import matplotlib.pyplot as plt

# data used for training can be found on https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia/data


data = []

training_data = []
test_data = []


def setup_data():
    IMAGE_SIZE = 100

    training_positive_dir = "/home/bartosz/Desktop/machine_learning/pneumonia_detector_cnn/chest_xray/train/PNEUMONIA"
    training_negative_dir = "/home/bartosz/Desktop/machine_learning/pneumonia_detector_cnn/chest_xray/train/NORMAL"
    test_positive_dir = "/home/bartosz/Desktop/machine_learning/pneumonia_detector_cnn/chest_xray/test/PNEUMONIA"
    test_negative_dir = "/home/bartosz/Desktop/machine_learning/pneumonia_detector_cnn/chest_xray/test/NORMAL"

    total_positive = 0
    total_negative = 0

    for dir in [training_positive_dir, training_negative_dir, test_positive_dir, test_negative_dir]:

        for f in tqdm(os.listdir(dir)):
            try:
                path = os.path.join(dir, f)
                img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
                img = cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE))
                img = np.array(img)

                if dir == training_positive_dir or dir == test_positive_dir:
                    img = [img, np.eye(2)[0]]
                    total_positive += 1
                else:
                    img = [img, np.eye(2)[1]]
                    total_negative += 1

                data.append(img)
            except Exception as e:
                print("error processing data:", path)
                print("error:", str(e))
    print(f"Total Positive: {total_positive}")
    print(f"Total Negative: {total_negative}")


setup_data()

plt.imshow(data[0][0], cmap="gray")
plt.show()
