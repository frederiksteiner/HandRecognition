import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from timeit import default_timer
import os
from sklearn.model_selection import train_test_split
from sklearn import preprocessing

class HandNetwork(nn.Module):
    def __init__(self, size_in, num_of_classes):
        super().__init__()
        self.fc0 = nn.Linear(size_in, 128)
        self.fc1 = nn.Linear(128, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 32)
        self.fc4 = nn.Linear(32, num_of_classes)



    def forward(self, x):
        x = F.relu(self.fc0(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.softmax(self.fc4(x))

        return x


fileList = sorted(os.listdir('./data'))
print(fileList)
fileDir = '.\data'

for idx, file in enumerate(fileList):
    filePath = os.path.join(fileDir, file)

    if idx == 0:
        x_train = np.transpose(np.load(filePath))
        y_train = idx  * np.ones((x_train.shape[0],))

    else:
        dataLen = np.transpose(np.load(filePath)).shape[0]
        x_train = np.append(x_train, np.transpose(np.load(filePath)), axis= 0)
        y_train = np.append(y_train, idx * np.ones((dataLen,)))

# lb = preprocessing.LabelBinarizer()
# lb.fit(range(6))
# y_train = lb.transform(y_train)
#y_train = np.expand_dims(y_train, axis = 1)
################################################################
# configurations
################################################################
batch_size = 20
learning_rate = 0.001
epochs = 100
step_size = 20
gamma = 0.5
num_of_classes = 12

################################################################
# load data
################################################################


x_train, x_test, y_train, y_test = train_test_split(x_train, y_train, test_size=0.2, random_state=42)

train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(torch.from_numpy(x_train).float(),  torch.from_numpy(y_train).float()), batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(torch.from_numpy(x_test).float(), torch.from_numpy(y_test).float()), batch_size=batch_size, shuffle=False)
ntrain = x_train.shape[0]
ntest = x_test.shape[0]

################################################################
# prep for training
################################################################
model = HandNetwork(42, num_of_classes)
total_losses_train = torch.zeros(epochs)
total_losses_test = torch.zeros(epochs)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
myloss = torch.nn.CrossEntropyLoss()

################################################################
# prep for training
################################################################

for ep in range(epochs):
    model.train()
    t1 = default_timer()
    train_mse = 0
    counter = 0
    for x, y in train_loader:
        optimizer.zero_grad()
        out = model(x)
        loss = myloss(out, y.type(torch.LongTensor))
        loss.backward()
        optimizer.step()
        train_mse += loss.item()



    scheduler.step()


    model.eval()
    abs_err = 0.0
    rel_err = 0.0
    with torch.no_grad():
        for x, y in test_loader:

            out = model(x)

            rel_err += myloss(out, y.type(torch.LongTensor)).item()

    train_mse/= ntrain
    abs_err /= ntest
    rel_err /= ntest

    total_losses_test[ep] = rel_err
    total_losses_train[ep] = train_mse

    t2 = default_timer()
    print(ep, t2-t1, train_mse, rel_err)

torch.save(model, './models/model2')
