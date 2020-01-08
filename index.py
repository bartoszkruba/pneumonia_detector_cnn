import numpy as np
import cv2
from tqdm import tqdm
import os
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# data used for training can be found on https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia/data

PROCESS_DATA = False
IMAGE_SIZE = 100


def setup_data():
    training_positive_dir = "/home/bartosz/Desktop/machine_learning/pneumonia_detector_cnn/chest_xray/train/PNEUMONIA"
    training_negative_dir = "/home/bartosz/Desktop/machine_learning/pneumonia_detector_cnn/chest_xray/train/NORMAL"
    test_positive_dir = "/home/bartosz/Desktop/machine_learning/pneumonia_detector_cnn/chest_xray/test/PNEUMONIA"
    test_negative_dir = "/home/bartosz/Desktop/machine_learning/pneumonia_detector_cnn/chest_xray/test/NORMAL"

    total_positive = 0
    total_negative = 0

    positive_data = []
    negative_data = []

    data = []

    for dir in [training_positive_dir, training_negative_dir, test_positive_dir, test_negative_dir]:

        print("loading files from", dir)
        for f in tqdm(os.listdir(dir)):
            try:
                path = os.path.join(dir, f)
                img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
                img = cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE))
                img = np.array(img)

                if dir == training_positive_dir or dir == test_positive_dir:
                    img = [img, np.eye(2)[0]]
                    total_positive += 1
                    positive_data.append(img)
                else:
                    img = [img, np.eye(2)[1]]
                    total_negative += 1
                    negative_data.append(img)

            except Exception as e:
                print("error processing data:", path)
                print(str(e))

    # print(f"Total Positive: {len(positive_data)}")
    # print(f"Total Negative: {len(negative_data)}")

    for d in positive_data[:len(negative_data)]:
        data.append(d)
    for d in negative_data:
        data.append(d)

    np.random.shuffle(data)
    np.save("training_data.npy", data)
    return data


class Net(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(1, 32, 5)
        self.conv2 = nn.Conv2d(32, 64, 5)
        self.conv3 = nn.Conv2d(64, 128, 5)

        self.to_linear = self.calculate_conv_output_size()

        self.fcl1 = nn.Linear(self.to_linear, 512)
        self.fcl2 = nn.Linear(512, 2)

    def calculate_conv_output_size(self):
        X = torch.randn(50, 50).view(-1, 1, 50, 50)
        X = F.max_pool2d(F.relu(self.conv1(X)), (2, 2))
        X = F.max_pool2d(F.relu(self.conv2(X)), (2, 2))
        X = F.max_pool2d(F.relu(self.conv3(X)), (2, 2))

        print("conv output shape: " + str(X[0].shape))

        return X[0].shape[0] * X[0].shape[1] * X[0].shape[2]

    def forward(self, X):
        X = F.max_pool2d(F.relu(self.conv1(X)), (2, 2))
        X = F.max_pool2d(F.relu(self.conv2(X)), (2, 2))
        X = F.max_pool2d(F.relu(self.conv3(X)), (2, 2))
        X = F.relu(self.fcl1(X.view(-1, self.to_linear)))
        X = F.relu(self.fcl2(X))

        return X


if PROCESS_DATA:
    data = setup_data()
else:
    data = np.load("training_data.npy", allow_pickle=True)

# plt.imshow(data[0][0], cmap="gray")
# plt.show()

training_data = data[int(len(data) * 0.1):]
test_data = data[:int(len(data) * 0.1)]

print(f"training_data length: {len(training_data)}")
print(f"test_data_length: {len(test_data)}")

net = Net()
