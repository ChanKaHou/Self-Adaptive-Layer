'''
Python program source code
for research article "Self-Adaptive Layer: An Application of Function Approximation Theory to Enhance Convergence Efficiency in Neural Networks"

Version 1.0
(c) Copyright 2019 Ka-Hou Chan <chankahou (at) ipm.edu.mo>

The Python program source code is free software: you can redistribute
it and/or modify it under the terms of the GNU General Public License
as published by the Free Software Foundation, either version 3 of the License,
or (at your option) any later version.

The Clustering program source code is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY
or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License for
more details.

You should have received a copy of the GNU General Public License
along with the Kon package.  If not, see <http://www.gnu.org/licenses/>.
'''

import torch
import torchvision

train_set = torchvision.datasets.MNIST(root='./mnist/', train=True, transform=torchvision.transforms.ToTensor(), download=True)
#print(train_set.train_data.type(), train_set.train_data.shape) #torch.ByteTensor torch.Size([60000, 28, 28])
#print(train_set.train_labels.type(), train_set.train_labels.shape) #torch.LongTensor torch.Size([60000])
train_loader = torch.utils.data.DataLoader(dataset=train_set, batch_size=100, shuffle=True)

# prepare test_set
test_set = torchvision.datasets.MNIST(root='./mnist/', train=False, transform=torchvision.transforms.ToTensor(), download=True)
#print(test_set.train_data.type(), test_set.train_data.shape) #torch.ByteTensor torch.Size([10000, 28, 28])
#print(test_set.train_labels.type(), test_set.train_labels.shape) #torch.LongTensor torch.Size([10000])
test_loader = torch.utils.data.DataLoader(dataset=test_set, batch_size=test_set.__len__(), shuffle=False)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#####################################################################################
class SelfAdaptive(torch.nn.Module):
    def __init__(self, degree_poly):
        super(SelfAdaptive, self).__init__()
        self.perimeter = torch.arange(degree_poly, dtype=torch.float).to(device)
        self.linear = torch.nn.Linear(degree_poly, 1, bias=False)

    def forward(self, input):
        t = input.unsqueeze(-1)
        t = torch.cos(self.perimeter * torch.acos(t))
        t = self.linear(t).squeeze(-1)
        return t

class SelfRNN(torch.nn.Module):
    def __init__(self):
        super(SelfRNN, self).__init__()
        self.rnn = torch.nn.RNN(input_size=28, hidden_size=64, batch_first=True)
        self.selfAdaptive = SelfAdaptive(10)
        self.linear = torch.nn.Linear(64, 10)

    def forward(self, input):
        t = input.view(-1, 28, 28)
        t, _ = self.rnn(t, None)
        t = self.selfAdaptive(t[:, -1, :])
        t = torch.relu(self.linear(t))
        return t

#####################################################################################
nn = SelfRNN().to(device)
optimizer = torch.optim.Adagrad(nn.parameters())
loss_func = torch.nn.CrossEntropyLoss()

epoch = 1
while (epoch < 200):
    for step, (train_data, train_label) in enumerate(train_loader):
        train_data = train_data.to(device)
        train_label = train_label.to(device)

        label = nn(train_data)
        loss = loss_func(label, train_label)
        #print('Epoch: ', epoch, '| Train Loss: %.4f' % loss.data)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    for step, (test_data, test_label) in enumerate(test_loader):
        test_data = test_data.to(device)
        test_label = test_label.to(device)

        label = nn(test_data)
        loss = loss_func(label, test_label)
        accuracy = float((label.max(-1)[1] == test_label).sum()) / float(label.size(0))
        print('Epoch: ', epoch, '| Test Loss: %.4f' % loss.data, '| Test Accuracy: %.4f' % accuracy)

    epoch += 1
