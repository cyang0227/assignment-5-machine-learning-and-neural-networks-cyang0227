import torch
import torchvision
import torchvision.transforms.v2 as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
import matplotlib.pyplot as plt

'''
In this file you will write end-to-end code to train a neural network to categorize fashion-mnist data
'''
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

'''
PART 1:
Preprocess the fashion mnist dataset and determine a good batch size for the dataset.
Anything that works is accepted.
'''

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
    ])  # Use transforms to convert images to tensors and normalize them
batch_size = 10
'''
PART 2:
Load the dataset. Make sure to utilize the transform and batch_size you wrote in the last section.
'''

trainset = torchvision.datasets.FashionMNIST(root='dataset', download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True)

testset = torchvision.datasets.FashionMNIST(root='dataset', train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False)

'''
PART 3:
Design a multi layer perceptron. Since this is a purely Feedforward network, you mustn't use any convolutional layers
Do not directly import or copy any existing models.
'''

# Report split sizes
print('Training set has {} instances'.format(len(trainset)))
print('Validation set has {} instances'.format(len(testset)))

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        #refer to the documentation to see how to use the layers: https://medium.com/@aaysbt/fashion-mnist-data-training-using-pytorch-7f6ad71e96f4
        self.fc1 = nn.Linear(784, 392)
        self.fc2 = nn.Linear(392, 146)
        self.fc3 = nn.Linear(146, 73)
        self.fc4 = nn.Linear(73, 10)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        x = x.view(-1, 784)
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.dropout(F.relu(self.fc2(x)))
        x = self.dropout(F.relu(self.fc3(x)))
        #sigmoid output layer
        x = torch.sigmoid(self.fc4(x))
        return x
        
net = Net().to(device)

'''
PART 4:
Choose a good loss function and optimizer
'''

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

'''
PART 5:
Train your model!
'''

num_epochs = 100
'''Choose the number of epochs '''

for epoch in range(num_epochs):  # loop over the dataset multiple times
    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data[0].to(device), data[1].to(device)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    print(f"Epoch {epoch+1}, Training loss: {running_loss / len(trainloader)}")

print('Finished Training')


'''
PART 6:
Evalute your model! Accuracy should be greater or equal to 80%

'''
net.eval()

correct = 0
total = 0
# since we're not training, we don't need to calculate the gradients for our outputs
with torch.no_grad():
    for data in testloader:
        images, labels = data[0].to(device), data[1].to(device)
        # calculate outputs by running images through the network
        outputs = net(images)
        # the class with the highest energy is what we choose as prediction
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print('Accuracy: ', correct/total)

'''
PART 7:
Check the written portion. You need to generate some plots. 
'''
