import torch
import torch.nn as nn

# from tensorboardX import SummaryWriter
import tensorwatch as tw
from utility import load_customer_data
import numpy as np
from  torch.utils import data
import torch




train_data_set, test_data_set, train_label_set, test_label_set = load_customer_data()
customer_train_data=torch.from_numpy(train_data_set.astype(np.float32))
customer_train_label_data=torch.from_numpy(train_label_set.astype(np.int64))
customer_test_data=torch.from_numpy(test_data_set.astype(np.float32))
customer_test_label_data=torch.from_numpy(test_label_set.astype(np.int64))

train_data=data.TensorDataset(customer_train_data,customer_train_label_data)
train_loader=data.DataLoader(
    dataset=train_data,
    batch_size=64,
    shuffle=True,
    num_workers=0,
)




import torch
import torch.nn as nn


# Check Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Define Hyper-parameters
input_size = 22
hidden_size = 22
num_classes = 2
num_epochs = 10
batch_size = 64
learning_rate = 0.001
vis_model = False


# spam mail dataset
train_data=data.TensorDataset(customer_train_data,customer_train_label_data)
test_data =data.TensorDataset(customer_test_data,customer_test_label_data)

# Data loader
train_loader=data.DataLoader(
    dataset=train_data,
    batch_size=batch_size,
    shuffle=True,
    num_workers=0,
)
test_loader=data.DataLoader(
    dataset=test_data,
    batch_size=64,
    shuffle=False,
    num_workers=0,
)

# Fully connected neural network
class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(NeuralNet, self).__init__()

        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, num_classes)



    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)

        return out


model = NeuralNet(input_size, hidden_size, num_classes).to(device)
#tw.draw_model(model, [90366,22])
# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Train the model
losses= []
accuary= []
total_step = len(train_loader)
for epoch in range(num_epochs):
    train_losses = 0
    train_accuary =0
    for i, (churns, labels) in enumerate(train_loader):
        # Move tensors to the configured device
        churns = churns.to(device)
        labels = labels.to(device)
         # Forward pass
        outputs = model(churns)

        loss = criterion(outputs, labels)

        # Backprpagation and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_losses += loss.item()

        _, pred = outputs.max(1)
        num_correct = (pred == labels).sum().item()
        accuary_ = num_correct / batch_size
        train_accuary += accuary_

        if (i + 1) % 10 == 0:
            print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                  .format(epoch + 1, num_epochs, i + 1, total_step, loss.item()))
    losses.append(train_losses / len(train_loader))
    accuary.append(train_accuary/ len(train_loader))

# Test the model
# In the test phase, don't need to compute gradients (for memory efficiency)
test_accuary=[]
g=None
with torch.no_grad():
    correct = 0
    total = 0
    for churns, labels in test_loader:
        churns = churns.to(device)
        labels = labels.to(device)

        outputs = model(churns)
        #g = make_dot(outputs)
        # with SummaryWriter(comment='churn_model') as w:
        #     w.add_graph(model, churns)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        test_accuary.append(correct / total)
    print('Accuracy of the network on the  test churns: {} %'.format(100 * correct / total))
#g.render('churns_model', view=False)

import matplotlib.pyplot as plt
#plot the train accuracy and test accuracy of epoches
x1 = np.arange(0, 1+epoch)
plt.plot(x1, losses)
plt.plot(x1,accuary)
plt.title("train losses")
plt.xlabel("epoch")
plt.ylabel("train accuary")
plt.grid()
plt.show()


x2 = np.arange(0,len(test_loader))
plt.plot(x2,test_accuary)
plt.title("test accuary")
plt.xlabel("epoch")
plt.ylabel("loss")
plt.grid()
plt.show()