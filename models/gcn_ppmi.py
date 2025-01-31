import numpy as np

import torch
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data, DataLoader
from torchvision import models, transforms
from sklearn.neighbors import kneighbors_graph

class GCN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels[0])
        self.convh = []
        for i in range(len(hidden_channels) - 1):
            self.convh.append(GCNConv(hidden_channels[i], hidden_channels[i + 1]))
        self.conv2 = GCNConv(hidden_channels[-1], out_channels)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = torch.relu(x)
        for c in self.convh:
            x = c(x, edge_index)
            x = torch.relu(x)
        x = self.conv2(x, edge_index)
        return torch.log_softmax(x, dim=1)

from scipy.io import loadmat


def main():
    data = loadmat('PPMI.mat')

    print(data.keys())
    print(data['__header__'])

    print(data.keys())
    # print(data)
    print(data['X'].shape)
    print(data['X'].dtype)
    print(data['label'].shape)
    print(data['label'].dtype)

    print(data['X'][0].shape)

    # data['X'][718][1]
    print(data['X'][0][0].shape)
    print(data['X'][0][0].dtype)

    model = GCN(in_channels=512, hidden_channels=[256, ], out_channels=2)

    cnn = models.resnet18(pretrained=True)
    cnn = torch.nn.Sequential(*(list(cnn.children())[:-1]))

    num_nodes = int(data['X'].shape[0])

    features = []
    for i in range(num_nodes):
        img = torch.tensor(data['X'][i][0]).reshape(3, 84, 84).float().unsqueeze(0)
        feature = cnn(img).squeeze()
        features.append(feature)

    node_features = torch.stack(features).detach()
    print(node_features.shape)

    adj_matrix = kneighbors_graph(node_features.numpy(),
                                  n_neighbors=5, mode='connectivity', include_self=False)
    edge_index = torch.tensor(np.array(adj_matrix.nonzero()), dtype=torch.long)

    graph = Data(x=node_features, edge_index=edge_index)

    labels = torch.tensor(data['label'])
    labels = labels.reshape(labels.shape[0])

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=5e-4)

    train_mask = torch.rand(num_nodes) < 0.8
    test_mask = ~train_mask
    print(train_mask)
    print(test_mask)
    print(train_mask.dtype)
    print(test_mask.dtype)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
    loss_fn = torch.nn.CrossEntropyLoss()

    for epoch in range(200):
        model.train()
        optimizer.zero_grad()
        out = model(graph.x, graph.edge_index)
        loss = loss_fn(out[train_mask], labels[train_mask])
        loss.backward()
        optimizer.step()
        print(f"Epoch {epoch + 1}, Loss: {loss.item()}")

    # Evaluation
    model.eval()
    pred = model(graph.x, graph.edge_index).argmax(dim=1)

    print(pred.shape)
    print(test_mask.shape)
    print(labels.shape)
    print((pred[test_mask] == labels[test_mask]).shape)

    correct_predictions = (pred[test_mask] == labels[test_mask]).sum().item()
    total_test_nodes = test_mask.sum().item()
    accuracy = correct_predictions / total_test_nodes
    print(f"Correct Predictions: {correct_predictions}")
    print(f"Total Test Nodes: {total_test_nodes}")
    print(f"Accuracy: {accuracy:.2%}")


if __name__ == '__main__':
    main()