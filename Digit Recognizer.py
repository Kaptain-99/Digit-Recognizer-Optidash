# Importing Valid Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
from torch.utils.data import DataLoader, Dataset, random_split
import torchvision
import torchvision.transforms as transforms

# Load Data
class DatasetMNIST(Dataset):
    
    def __init__(self,file_path,transform=None,testset=False):
        self.data = pd.read_csv(file_path)
        self.transform = transform
        self.testset = testset
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self,index):
        if self.testset:
            image = self.data.iloc[index,0:].values.astype(np.uint8).reshape((28,28,1))
            label = ''
        else:
            image = self.data.iloc[index,1:].values.astype(np.uint8).reshape((28,28,1))
            label = self.data.iloc[index,0]
        
        if self.transform is not None:
            image = self.transform(image)
            
        return image,label

# Load train and test dataset and create respective dataloaders
transform = transforms.Compose([transforms.ToTensor()])

dataset = DatasetMNIST('../input/digit-recognizer/train.csv',transform = transform)
testset  = DatasetMNIST('../input/digit-recognizer/test.csv',transform = transform,testset = True)

train_len = int(dataset.__len__() * 0.8)
valid_len = int(dataset.__len__() * 0.2)

trainset, validset = random_split(dataset, lengths = [train_len,valid_len])

trainloader = DataLoader(trainset,batch_size = 8,shuffle = True)
validloader = DataLoader(validset,batch_size = 8, shuffle = True)
testloader = DataLoader(testset, batch_size = 8, shuffle = True)

# Print out a random batch
def imshow(img):
    plt.imshow(np.transpose(img.numpy(),(1,2,0)))
    plt.axis('off')
    plt.title('Random Batch')
    plt.show()
    
dataiter = iter(trainloader)
images,labels = dataiter.next()

imshow(torchvision.utils.make_grid(images))

# Defining the CNN
import torch.nn.functional as F

class Net(nn.Module):
    
    def __init__(self):
        super(Net,self).__init__()
        self.conv1 = nn.Conv2d(1,6,3)
        self.conv2 = nn.Conv2d(6,16,3)
        
        self.fc1 = nn.Linear(16*5*5,120)
        self.fc2 = nn.Linear(120,84)
        self.fc3 = nn.Linear(84,10)
        
    def forward(self,x):
        x = F.max_pool2d(F.relu(self.conv1(x)),2)
        x = F.max_pool2d(F.relu(self.conv2(x)),2)
        x = x.view(-1,self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        
        return x
    
    def num_flat_features(self,x):
        size = x.size()[1:]
        num_features = 1
        for s in size:
            num_features *= s
        return num_features
    
net = Net()
print(net)

# Check if shapes match for all layer interconnections
input = torch.randn(1,1,28,28)
out = net(input)
print(out)

# Define Loss and choose Optimizer
import torch.optim as optim

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(),lr = 0.001,momentum = 0.9)

# Begin Training
if torch.cuda.is_available():
    net = net.cuda()
    criterion = criterion.cuda()

for epoch in range(5):
    
    running_loss = 0.0
    for i,[inputs,labels] in enumerate(trainloader,0):
        
        if torch.cuda.is_available():
            inputs = inputs.cuda()
            labels = labels.cuda()
        
        optimizer.zero_grad()
        
        outputs = net(inputs)
        loss = criterion(outputs,labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        if i % 2000 == 1999:    # print every 2000 mini-batches
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 2000))
            running_loss = 0.0

print('Finished Training')

PATH = 'mnist.pth'
torch.save(net.state_dict(),PATH)

correct = 0
total = 0

with torch.no_grad():
    for data in validloader:
        inputs,labels = data
        
        if torch.cuda.is_available():
            inputs = inputs.cuda()
            labels = labels.cuda()
        
        outputs = net(inputs)
        _, predicted = torch.max(outputs,1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print('Accuracy of the network on the validation images : %d %%' % (100*correct/total))

outputs = net(images)
for res in outputs:
    print(res.argmax())

# Testest Predictions
test_pred =  torch.LongTensor()

with torch.no_grad():
    for data in testloader:
        inputs,_ = data
        
        if torch.cuda.is_available():
            inputs = inputs.cuda()
        
        outputs = net(inputs)
        _,pred = outputs.cpu().max(1,keepdim = True)
        test_pred = torch.cat((test_pred,pred),dim = 0)

out_df = pd.DataFrame(np.c_[np.arange(1,len(testset)+1)[:,None],test_pred.numpy()],columns=['ImageId', 'Label'])
out_df.head()
out_df.to_csv('submission.csv',index = False)