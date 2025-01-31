import torch
import torch.nn as nn
import numpy as np
from torch.cuda.amp import GradScaler, autocast
from sklearn.metrics import f1_score, roc_auc_score
from scipy.special import expit, softmax
from tqdm import tqdm

# Precompute the features from the encoder and store them
def precompute_features(encoder, loader, device):
    encoder.eval()
    all_features = []
    all_labels = []
    
    with torch.no_grad():
        print('Precomputing features...')
        for wave, label in tqdm(loader):
            bs, _, _  = wave.shape
            wave = wave.to(device)
            feature = encoder.representation(wave) # (bs,c*50,384)
            all_features.append(feature.cpu())
            all_labels.append(label)
                
    all_features = torch.cat(all_features)
    all_labels = torch.cat(all_labels)
    
    return all_features, all_labels


def features_dataloader(encoder, loader, batch_size=32, shuffle=True, device='cpu'):

    features, labels = precompute_features(encoder, loader, device=device)
    dataset = torch.utils.data.TensorDataset(features, labels)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=4)

    return dataloader

class LinearClassifier(nn.Module):
    def __init__(self, input_dim, num_labels, apply_bn=False):
        super(LinearClassifier, self).__init__()
        self.apply_bn = apply_bn
        self.bn = nn.BatchNorm1d(input_dim, affine=False, eps=1e-6)
        self.fc = nn.Linear(input_dim, num_labels)
        
    def forward(self, x):
        if self.apply_bn:
            x = self.bn(x)

        x = self.fc(x)
        return x

class FinetuningClassifier(nn.Module):
    def __init__(self, encoder, encoder_dim, num_labels, device='cpu', apply_bn=False):
        super(FinetuningClassifier, self).__init__()
        self.encoder = encoder
        self.encoder_dim = encoder_dim
        # self.bn = nn.BatchNorm1d(encoder_dim, affine=False, eps=1e-6) # this outputs nan value in mixed precision
        self.fc = LinearClassifier(encoder_dim, num_labels, apply_bn=apply_bn)
        
    def forward(self, x):
        bs,_,_ = x.shape
        x = self.encoder.representation(x)
        # x = self.bn(x)
        x = self.fc(x)
        return x

class SimpleLinearRegression(nn.Module):
    def __init__(self, input_dim=384):
        super(SimpleLinearRegression, self).__init__()
        self.linear = nn.Linear(input_dim, 1)

    def forward(self, x):
        x = self.linear(x)
        return x.view(-1)


def train_multilabel(num_epochs, linear_model, optimizer, criterion, scheduler, train_loader_linear, test_loader_linear, device, print_every=True):

    iterations_per_epoch = len(train_loader_linear)
    max_auc = 0.0
    
    for epoch in range(num_epochs):
        linear_model.train()
        
        for minibatch, (batch_features, batch_labels) in enumerate(train_loader_linear):
            batch_features = batch_features.to(device)
            batch_labels = batch_labels.to(device)
            outputs = linear_model(batch_features)
            loss = criterion(outputs, batch_labels.float())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if scheduler is not None:
                scheduler.step(epoch * iterations_per_epoch + minibatch)

        all_labels = []
        all_outputs = []
        
        with torch.no_grad():
            linear_model.eval()
            for batch_features, batch_labels in test_loader_linear:
                batch_features = batch_features.to(device)
                outputs = linear_model(batch_features)
                all_labels.append(batch_labels.cpu().numpy())
                all_outputs.append(outputs.cpu().numpy())

        all_labels = np.vstack(all_labels)
        all_outputs = np.vstack(all_outputs)
        all_probs = expit(all_outputs)

        auc_scores = [roc_auc_score(all_labels[:, i], all_outputs[:, i]) if np.unique(all_labels[:, i]).size > 1 else float('nan') for i in range(all_labels.shape[1])]
        avg_auc = np.nanmean(auc_scores)

        if avg_auc > max_auc:
            max_auc = avg_auc

        # Compute F1 score
        predicted_labels = (all_probs >= 0.5).astype(int)
        macro_f1 = f1_score(all_labels, predicted_labels, average='macro')
        
        if print_every:
            print(f'Epoch({epoch}) AUC: {avg_auc:.3f}({max_auc:.3f}), Macro F1: {macro_f1:.3f}')

    return avg_auc, macro_f1


def train_multiclass(num_epochs, model, criterion, optimizer, train_loader_linear, test_loader_linear, device, scheduler=None, print_every=False, amp=False):

    iterations_per_epoch = len(train_loader_linear)
    max_auc = 0.0
    macro_f1 = 0.0

    if amp:
        scaler = GradScaler()
    
    for epoch in range(num_epochs):
        model.train()
        
        for minibatch, (batch_features, batch_labels) in enumerate(train_loader_linear):
            batch_features = batch_features.to(device)
            batch_labels = batch_labels.to(device)
            optimizer.zero_grad()

            # Mixed precision training
            if amp:
                with torch.cuda.amp.autocast():
                    outputs = model(batch_features)
                    loss = criterion(outputs, batch_labels)
            
                scaler.scale(loss).backward()  # Scale the loss and backpropagate
                scaler.step(optimizer)  # Step the optimizer
                scaler.update()  # Update the scaler

            else:
                outputs = model(batch_features)
                loss = criterion(outputs, batch_labels)
                loss.backward()
                optimizer.step()

            if scheduler is not None:
                scheduler.step(epoch * iterations_per_epoch + minibatch)

        all_labels = []
        all_outputs = []
        
        with torch.no_grad():
            model.eval()
            for minibatch, (batch_features, batch_labels) in enumerate(test_loader_linear):
                batch_features = batch_features.to(device)

                if amp:
                    with torch.cuda.amp.autocast():
                        outputs = model(batch_features)
                else:
                    outputs = model(batch_features)


                all_labels.append(batch_labels.cpu().numpy())
                all_outputs.append(outputs.cpu().numpy())

        all_labels = np.concatenate(all_labels)
        all_outputs = np.vstack(all_outputs)

        # when using amp, all_outputs should be changed to float32
        if amp:
            all_outputs = np.float32(all_outputs)

        all_probs = softmax(all_outputs, axis=1)

        # Compute ROC AUC score
        avg_auc = roc_auc_score(all_labels, all_probs, average='macro', multi_class='ovo')
        if avg_auc > max_auc:
            max_auc = avg_auc

        # Compute F1 score
        predicted_labels = np.argmax(all_outputs, axis=1)
        macro_f1 = f1_score(all_labels, predicted_labels, average='macro')

        if print_every:
            print(f'Epoch({epoch}) AUC: {avg_auc:.3f}({max_auc:.3f}), Macro F1: {macro_f1:.3f}')

    print(f'Epoch({epoch}) AUC: {avg_auc:.3f}({max_auc:.3f}), Macro F1: {macro_f1:.3f}')
    return avg_auc, macro_f1



# from sklearn.metrics import accuracy_score, confusion_matrix
# import matplotlib.pyplot as plt
# import seaborn as sns
# from sklearn.metrics import accuracy_score, confusion_matrix

# def train_multiclass(num_epochs, model, criterion, optimizer, train_loader_linear, test_loader_linear, device, scheduler=None, print_every=False, amp=False):
#     iterations_per_epoch = len(train_loader_linear)
#     max_auc = 0.0
#     macro_f1 = 0.0

#     label_map = {5: 'BBB', 11: 'PVC', 13: 'Paced', 4: 'AVB4', 0: 'AFIB', 14: 'SA',
#                  6: 'BBB_AFIB', 12: 'PVC2', 3: 'AVB2', 1: 'AFL', 8: 'NSR', 15: 'VT',
#                  2: 'AVB1', 9: 'PAC', 10: 'PAC2', 7: 'NQT'}
    
#     if amp:
#         scaler = GradScaler()
    
#     for epoch in range(num_epochs):
#         model.train()
        
#         for minibatch, (batch_features, batch_labels) in enumerate(train_loader_linear):
#             batch_features = batch_features.to(device)
#             batch_labels = batch_labels.to(device)
#             optimizer.zero_grad()

#             # Mixed precision training
#             if amp:
#                 with torch.cuda.amp.autocast():
#                     outputs = model(batch_features)
#                     loss = criterion(outputs, batch_labels)
            
#                 scaler.scale(loss).backward()  # Scale the loss and backpropagate
#                 scaler.step(optimizer)  # Step the optimizer
#                 scaler.update()  # Update the scaler

#             else:
#                 outputs = model(batch_features)
#                 loss = criterion(outputs, batch_labels)
#                 loss.backward()
#                 optimizer.step()

#             if scheduler is not None:
#                 scheduler.step(epoch * iterations_per_epoch + minibatch)

#         all_labels = []
#         all_outputs = []
        
#         with torch.no_grad():
#             model.eval()
#             for minibatch, (batch_features, batch_labels) in enumerate(test_loader_linear):
#                 batch_features = batch_features.to(device)

#                 if amp:
#                     with torch.cuda.amp.autocast():
#                         outputs = model(batch_features)
#                 else:
#                     outputs = model(batch_features)

#                 all_labels.append(batch_labels.cpu().numpy())
#                 all_outputs.append(outputs.cpu().numpy())

#         all_labels = np.concatenate(all_labels)
#         all_outputs = np.vstack(all_outputs)

#         # when using amp, all_outputs should be changed to float32
#         if amp:
#             all_outputs = np.float32(all_outputs)

#         all_probs = softmax(all_outputs, axis=1)

#         # Compute ROC AUC score
#         avg_auc = roc_auc_score(all_labels, all_probs, average='macro', multi_class='ovo')
#         if avg_auc > max_auc:
#             max_auc = avg_auc

#         # Compute F1 score
#         predicted_labels = np.argmax(all_outputs, axis=1)
#         macro_f1 = f1_score(all_labels, predicted_labels, average='macro')

#         # Compute Accuracy
#         accuracy = accuracy_score(all_labels, predicted_labels)

#         # Compute Confusion Matrix
#         conf_matrix = confusion_matrix(all_labels, predicted_labels)

#         if print_every:
#             print(f'Epoch({epoch}) AUC: {avg_auc:.3f}({max_auc:.3f}), Macro F1: {macro_f1:.3f}, Accuracy: {accuracy:.3f}')
#             print(f'Confusion Matrix:\n{conf_matrix}')

#         # Plot and save the confusion matrix
#         plt.figure(figsize=(10, 8))
#         sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", 
#                     xticklabels=[label_map[i] for i in range(len(label_map))], 
#                     yticklabels=[label_map[i] for i in range(len(label_map))])
#         plt.title(f'Confusion Matrix - Epoch {epoch}')
#         plt.ylabel('True label')
#         plt.xlabel('Predicted label')
#         plt.savefig(f'./confusion_matrix_epoch_{epoch}.png')
#         plt.close()

#     # Print final metrics after training
#     print(f'Final Epoch({epoch}) AUC: {avg_auc:.3f}({max_auc:.3f}), Macro F1: {macro_f1:.3f}, Accuracy: {accuracy:.3f}')
#     print(f'Confusion Matrix:\n{conf_matrix}')
    
#     return avg_auc, macro_f1
