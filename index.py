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
BATCH_SIZE = 64
EPOCHS = 5
TEST_EVERY = 5


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
        X = torch.randn(IMAGE_SIZE, IMAGE_SIZE).view(-1, 1, IMAGE_SIZE, IMAGE_SIZE)
        X = F.max_pool2d(F.relu(self.conv1(X)), (2, 2))
        X = F.max_pool2d(F.relu(self.conv2(X)), (2, 2))
        X = F.max_pool2d(F.relu(self.conv3(X)), (2, 2))

        return X[0].shape[0] * X[0].shape[1] * X[0].shape[2]

    def forward(self, X):
        X = F.max_pool2d(F.relu(self.conv1(X)), (2, 2))
        X = F.max_pool2d(F.relu(self.conv2(X)), (2, 2))
        X = F.max_pool2d(F.relu(self.conv3(X)), (2, 2))
        X = F.relu(self.fcl1(X.view(-1, self.to_linear)))
        X = self.fcl2(X)

        return X


if PROCESS_DATA:
    data = setup_data()
else:
    data = np.load("training_data.npy", allow_pickle=True)

# plt.imshow(data[0][0], cmap="gray")
# plt.show()

# print(f"training_data length: {len(training_data)}")
# print(f"test_data_length: {len(test_data)}")

X = [i[0] / 255.0 for i in data]

y = [i[1] for i in data]

training_size = int(len(data) * 0.1)

training_X = X[training_size:]
training_y = y[training_size:]

test_X = X[:training_size]
test_y = y[:training_size]

net = Net()

optimizer = optim.Adam(net.parameters(), lr=0.001)
loss_function = nn.MSELoss()


def test(size=32):
    start_position = np.random.randint(0, len(test_X) - size)
    batch_X = test_X[start_position: start_position + size]
    batch_y = test_y[start_position: start_position + size]

    with torch.no_grad():
        outputs = net(torch.Tensor(batch_X).view(-1, 1, IMAGE_SIZE, IMAGE_SIZE))
        matches = [torch.argmax(i) == torch.argmax(j) for i, j in zip(outputs, torch.Tensor(batch_y))]
        acc = matches.count(True) / len(matches)
        loss = loss_function(outputs, torch.Tensor(batch_y))
    return acc, loss


metrics = []

for epoch in range(EPOCHS):
    for i in tqdm(range(0, len(training_X), BATCH_SIZE)):
        batch_X = torch.Tensor(training_X[i:i + BATCH_SIZE]).view(-1, 1, IMAGE_SIZE, IMAGE_SIZE)
        batch_y = torch.Tensor(training_y[i:i + BATCH_SIZE])

        net.zero_grad()
        outputs = net(batch_X)
        matches = [torch.argmax(i) == torch.argmax(j) for i, j in zip(outputs, batch_y)]
        acc = matches.count(True) / len(matches)
        loss = loss_function(outputs, batch_y)
        loss.backward()
        optimizer.step()

        if i % TEST_EVERY == 0:
            training_acc, training_loss = test()
            metrics.append((training_acc, training_loss, acc, loss))

plt.plot(range(len(metrics)), [i[0] * 100 for i in metrics], label="Test Accuracy")
plt.plot(range(len(metrics)), [i[0] * 100 for i in metrics], label="Test Loss")
plt.plot(range(len(metrics)), [i[2] * 100 for i in metrics], label="Training Accuracy")
plt.plot(range(len(metrics)), [i[3] * 100 for i in metrics], label="Training Loss")
plt.legend()
plt.show()
