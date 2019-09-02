'''
Clustering program source code
for research article "Self-Adaptive Layer: An Application of Function Approximation Theory to Enhance Convergence Efficiency in Neural Networks"

Version 1.0
(c) Copyright 2019 Ka-Hou Chan <chankahou (at) ipm.edu.mo>

The Clustering program source code is free software: you can redistribute
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
import matplotlib.pyplot as plt

x = torch.unsqueeze(torch.linspace(-1, +1, 100), dim=1)
y = torch.sigmoid(x) * x + 0.2 * torch.rand(x.size())

#####################################################################################
class SelfAdaptive(torch.nn.Module):
    def __init__(self, degree_poly):
        super(SelfAdaptive, self).__init__()
        self.perimeter = torch.arange(degree_poly, dtype=torch.float)
        self.linear = torch.nn.Linear(degree_poly, 1, bias=False)

    def forward(self, input):
        t = input.unsqueeze(-1)
        t = torch.cos(self.perimeter * torch.acos(t))
        t = self.linear(t).squeeze(-1)
        return t

class SelfLINEAR(torch.nn.Module):
    def __init__(self):
        super(SelfLINEAR, self).__init__()
        self.selfAdaptive = SelfAdaptive(200)
        self.linear1 = torch.nn.Linear(1, 10)
        self.linear2 = torch.nn.Linear(10, 1)

    def forward(self, input):
        t = input
        t = self.selfAdaptive(t)
        t = torch.relu(self.linear1(t))
        t = self.linear2(t)
        return t    

#####################################################################################
nn = SelfLINEAR()
optimizer = torch.optim.Adagrad(nn.parameters())
loss_func = torch.nn.MSELoss()

while (True):
    pred = nn(x)

    loss = loss_func(pred, y)
    optimizer.zero_grad()

    loss.backward()
    optimizer.step()

    plt.cla()
    plt.scatter(x.data.numpy(), y.data.numpy())
    plt.plot(x.data.numpy(), pred.data.numpy(), 'r-', lw=2)
    plt.text(0.5, 0, 'Loss=%.6f' % loss.data.numpy())
    plt.pause(0.01)
