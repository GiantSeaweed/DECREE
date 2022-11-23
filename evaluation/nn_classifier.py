import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import TensorDataset, DataLoader
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm


class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size_list, num_classes):
        super(NeuralNet, self).__init__()
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(input_size, hidden_size_list[0])
        self.fc2 = nn.Linear(hidden_size_list[0], hidden_size_list[1])
        self.fc3 = nn.Linear(hidden_size_list[1], num_classes)

    def forward(self, x):
        out = self.fc1(x)
        out = F.relu(out)
        out = self.dropout2(out)
        out = self.fc2(out)
        out = F.relu(out)
        out = self.fc3(out)
        return out
'''
class NeuralNetMultiLayer(nn.Module):
    def __init__(self, input_size, hidden_size_list, num_classes):
        super(NeuralNetMultiLayer, self).__init__()
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(input_size, hidden_size_list[0])
        self.fc2 = nn.Linear(hidden_size_list[0], hidden_size_list[1])
        self.fc3 = nn.Linear(hidden_size_list[1], hidden_size_list[2])
        self.fc4 = nn.Linear(hidden_size_list[2], hidden_size_list[3])
        self.fc5 = nn.Linear(hidden_size_list[3], num_classes)

    def forward(self, x):
        out = self.fc1(x)
        out = F.relu(out)
        out = self.dropout2(out)
        out = self.fc2(out)
        out = self.fc3(F.relu(out))
        out = self.fc4(F.relu(out))
        out = self.fc5(F.relu(out))
        return out
'''

def create_torch_dataloader(feature_bank, label_bank, batch_size, shuffle=False, num_workers=2, pin_memory=True):
    # transform to torch tensor
    tensor_x, tensor_y = torch.Tensor(feature_bank), torch.Tensor(label_bank)

    dataloader = DataLoader(
        TensorDataset(tensor_x, tensor_y),
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory
    )

    return dataloader


def net_train(net, train_loader, optimizer, epoch, criterion, args):
    device = torch.device(f'cuda:{args.gpu}')
    """Training"""
    net.train()
    overall_loss = 0.0
    for batch_idx, (data, label) in enumerate(train_loader):
        data, label = data.to(device), label.to(device)
        optimizer.zero_grad()
        output = net(data)
        loss = criterion(output, label.long())

        loss.backward()
        optimizer.step()
        overall_loss += loss.item()
    print('Train Epoch: {} \tLoss: {:.6f}'.format(epoch, overall_loss*train_loader.batch_size/len(train_loader.dataset)))


def net_test(net, test_loader, epoch, criterion, args, keyword='Accuracy'):
    device = torch.device(f'cuda:{args.gpu}')
    """Testing"""
    net.eval()
    test_loss = 0.0
    correct = 0.0

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = net(data)
            # print('output:', output)
            # print('target:', target)
            test_loss += criterion(output, target.long()).item()
            pred = output.argmax(dim=1, keepdim=True)
            # if 'ASR' in keyword:
            #     print(f'output:{np.asarray(pred.flatten().detach().cpu())}')
            #     print(f'target:{np.asarray(target.flatten().detach().cpu())}\n')
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_acc = 100. * correct / len(test_loader.dataset)
    test_loss /= len(test_loader.dataset)
    print('{{"metric": "Eval - {}", "value": {}, "epoch": {}}}'.format(
        keyword, 100. * correct / len(test_loader.dataset), epoch))

    return test_acc


def predict_feature(net, data_loader, args=None):
    if args is None:
        device = torch.device('cuda:0')
    else:
        device = torch.device(f'cuda:{args.gpu}')
    net.eval()
    feature_bank, target_bank = [], []
    with torch.no_grad():
        # generate feature bank
        for data, target in tqdm(data_loader, desc='Feature extracting'):
            feature = net(data.to(device))
            feature = F.normalize(feature, dim=1)
            feature_bank.append(feature)
            target_bank.append(target)
        # [D, N]
        feature_bank = torch.cat(feature_bank, dim=0).contiguous()
        target_bank = torch.cat(target_bank, dim=0).contiguous()

    return feature_bank.cpu().detach().numpy(), target_bank.detach().numpy()
