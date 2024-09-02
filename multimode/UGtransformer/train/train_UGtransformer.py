#! /usr/bin/env python
import torch
torch.manual_seed(123)

import numpy as np
np.random.seed(123)

import argparse
import glob
import os
import pickle
import time

from torch_geometric.loader import DataLoader
from torch.utils.data import random_split, ConcatDataset
from sklearn.metrics import classification_report, confusion_matrix
from UGtransformer.datasets.dataset_brca_pyg import CancerDataset
from UGtransformer.model.UGtransformer import *





# Parameters
# ==================================================

parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=777, help='random seed')
parser.add_argument('--batch_size', type=int, default=32, help='batch size')
parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
parser.add_argument('--weight_decay', type=float, default=0.001, help='weight decay')
parser.add_argument('--nhid', type=int, default=32, help='hidden size')
parser.add_argument('--sample_neighbor', type=bool, default=True, help='whether sample neighbors')
parser.add_argument('--sparse_attention', type=bool, default=True, help='whether use sparse attention')
parser.add_argument('--structure_learning', type=bool, default=True, help='whether perform structure learning')
#parser.add_argument('--pooling_ratio', type=float, default=0.4, help='pooling ratio')
parser.add_argument('--dropout_ratio', type=float, default=0.4, help='dropout ratio')
parser.add_argument('--lamb', type=float, default=1.0, help='trade-off parameter')
parser.add_argument('--dataset', type=str, default='STAD', help='')
parser.add_argument('--device', type=str, default='cuda:0', help='specify cuda devices')
parser.add_argument('--epochs', type=int, default=1000, help='maximum number of epochs')
parser.add_argument('--patience', type=int, default=100, help='patience for early stopping')
#parser.add_argument('--num_layers', type=int, default=3, help='number of pooling layers')
parser.add_argument("--ff_hidden_size", default=256, type=int, help="The hidden size for the feedforward layer")
parser.add_argument("--num_hidden_layers", default=1, type=int, help="")
parser.add_argument("--nhead", default=1, type=int, help="")
parser.add_argument("--num_timesteps", default=1, type=int, help="Number of self-attention layers within each UGformer layer")


def train(cancer_type):
    min_loss = 1e10
    patience_cnt = 0
    val_loss_values = []
    best_epoch = 0

    t = time.time()
    model.train()

    train_loss_list = []
    val_loss_list = []
    train_acc_list = []
    val_acc_list = []

    for epoch in range(args.epochs):
        loss_train = 0.0
        correct = 0
        for i, data in enumerate(train_loader):
            optimizer.zero_grad()
            data = data.to(args.device)
            out, add_outs, _, _ = model(data)
            # print(out, data.y)
            loss = F.nll_loss(out, data.y)
            # print(loss, mask_loss)
            # loss += mask_loss
            for add_out in add_outs:
                loss += F.nll_loss(add_out[0], data.y)
                loss += F.nll_loss(add_out[1], data.y)
                loss += torch.dist(add_out[0].max(dim=1)[1].to(torch.float32), add_out[1].max(dim=1)[1].to(torch.float32), p=1)
            loss.backward()
            optimizer.step()
            loss_train += loss.item()
            pred = out.max(dim=1)[1]
            correct += pred.eq(data.y).sum().item()
        acc_train = correct / len(train_loader.dataset)
        acc_val, loss_val = compute_val(val_loader)

        train_loss_list.append(loss_train)
        val_loss_list.append(loss_val)
        train_acc_list .append(acc_train)
        val_acc_list.append(acc_val)
        # # print('Epoch: {:04d}'.format(epoch + 1), 'loss_train: {:.6f}'.format(loss_train),
        #       'acc_train: {:.6f}'.format(acc_train), 'loss_val: {:.6f}'.format(loss_val),
        #       'acc_val: {:.6f}'.format(acc_val), 'time: {:.6f}s'.format(time.time() - t))

        val_loss_values.append(loss_val)
        torch.save(model.state_dict(), '{}_{}.pth'.format(cancer_type, epoch))
        if val_loss_values[-1] < min_loss:
            min_loss = val_loss_values[-1]
            best_epoch = epoch
            patience_cnt = 0
        else:
            patience_cnt += 1

        if patience_cnt == args.patience:
            break

        files = glob.glob('*.pth')
        for f in files:
            epoch_nb = int(f.split('.')[0].split('_')[1])
            type = f.split('.')[0].split('_')[0]
            if type == cancer_type and epoch_nb < best_epoch:
                os.remove(f)

    files = glob.glob('*.pth')
    for f in files:
        epoch_nb = int(f.split('.')[0].split('_')[1])
        type = f.split('.')[0].split('_')[0]
        if type == cancer_type and epoch_nb > best_epoch:
            os.remove(f)

    return best_epoch, time.time() - t

def compute_val(loader):
    model.eval()
    correct = 0.0
    loss_test = 0.0
    for data in loader:
        data = data.to(args.device)
        out, add_puts, _, _ = model(data)
        pred_list = [out.max(dim=1)[1].view(1, -1)]
        for add_put in add_puts:
            pred_list.append(add_put[0].max(dim=1)[1].view(1, -1))
            pred_list.append(add_put[1].max(dim=1)[1].view(1, -1))
        pred = torch.cat(pred_list, dim=0)
        pred, _ = torch.mode(pred, dim=0)

        correct += pred.eq(data.y).sum().item()
        loss_test += F.nll_loss(out, data.y).item()
    return correct / len(loader.dataset), loss_test


def compute_test(loader):
    model.eval()
    correct = 0.0
    loss_test = 0.0
    for data in loader:
        torch.cuda.empty_cache()
        data = data.to(args.device)
        out, add_outs, _, _ = model(data)
        pred_list = [out.max(dim=1)[1].view(1, -1)]
        for add_put in add_outs:
            pred_list.append(add_put[0].max(dim=1)[1].view(1, -1))
            pred_list.append(add_put[1].max(dim=1)[1].view(1, -1))
        pred = torch.cat(pred_list, dim=0)
        pred, _ = torch.mode(pred, dim=0)

        correct += pred.eq(data.y).sum().item()
        cm = confusion_matrix(data.y.cpu(), pred.cpu())
        print(cm)
        cm = classification_report(data.y.cpu(), pred.cpu(), digits=4)
        print(cm)
        loss_test += F.nll_loss(out, data.y).item()
    return correct / len(loader.dataset), loss_test


def evaluate_nodes(loader):
    model.eval()
    path = os.getcwd()

    for data in loader:
        torch.cuda.empty_cache()
        data = data.to(args.device)
        out, add_outs, s, m = model(data)
        pred_list = [out.max(dim=1)[1].view(1, -1)]
        for add_put in add_outs:
            pred_list.append(add_put[0].max(dim=1)[1].view(1, -1))
            pred_list.append(add_put[1].max(dim=1)[1].view(1, -1))
        pred = torch.cat(pred_list, dim=0)
        pred, _ = torch.mode(pred, dim=0)

        data_dict = {"ID": data.id, "Map": data.gene_map, "Type": data.cancer, "Label": data.y, "pred": pred,
                     "pred_labels": pred_list, "s": s, "m": m}
        dict_save = open(path + "\\masks_data\\" + str(data_dict["Type"][0]) + "\\" + str(data_dict["ID"][0]) + ".pkl",
                         "wb")
        pickle.dump(data_dict, dict_save)
        dict_save.close()
    return None


def get_kfold_data(k, i, dataset):
    fold_size = len(dataset) // k
    test_start = i * fold_size
    if i != k-1:
        test_end = (i+1) * fold_size
        X_test = dataset[test_start:test_end]
        X_train = ConcatDataset((dataset[0:test_start], dataset[test_end:]))
    else:
        X_test = dataset[test_start:]
        X_train = dataset[0:test_start]

    num_training = int(len(X_train) * 8 // 9)
    num_val = int(len(X_train) - num_training)

    training_set, validation_set = random_split(X_train, [num_training, num_val])

    return training_set, validation_set, X_test

if __name__ == '__main__':

    # Load data
    print("Loading data...")
    #file_names = os.listdir('../datasets/BRCA/mRNA/processed')
    #dataset = CancerDataset()
    all_times = 10
    args = parser.parse_args()
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)

    cancer_type = ["BRCA"]
    omics = "mRNA"
    for cancer in cancer_type:
        data_dir = '../datasets/' + cancer + '/' + omics
        filepath = os.path.join(data_dir,'processed')
        filenames = [f for f in os.listdir(filepath) if f.startswith('data_')]
        #for f in filenames:
            #os.remove(f)

        accs = 0
        times = 0
        args.dataset = cancer
        dataset = CancerDataset(data_dir)
        print(len(dataset))
        args.num_classes = dataset.num_classes
        args.num_features = dataset.num_features
        args.feature_dim_size = dataset.num_node_features
        print(args)

        num_training = int(len(dataset) * 0.8)
        num_val = int(len(dataset) * 0.1)
        num_test = len(dataset) - (num_training + num_val)
        print("Num_training, Num_val, Num_test :", num_training, num_val, num_test)

        min_loss = 1e10
        best_times = 0
        best_time_epoch = 0
        test_loss_values = []
        dataset = dataset.shuffle()

        for i in range(all_times):
            print("Time: ", i)

            # training_set, validation_set, test_set = random_split(dataset, [num_training, num_val, num_test])
            training_set, validation_set, test_set = get_kfold_data(all_times, i, dataset)

            train_loader = DataLoader(training_set, batch_size=args.batch_size, shuffle=True, drop_last=True)
            val_loader = DataLoader(validation_set, batch_size=128, shuffle=False)
            test_loader = DataLoader(test_set, batch_size=128, shuffle=False)

            evaluate_loader = DataLoader(dataset, batch_size=1, shuffle=False)

            #model = FullyConnectedGT_UGformerV2(args.num_features, args.num_classes, args).to(args.device)
            model = FullyConnectedGT_UGformerV2(feature_dim_size=args.feature_dim_size, ff_hidden_size=args.ff_hidden_size,
                        num_classes=args.num_classes, dropout=args.dropout_ratio,
                        num_self_att_layers=args.num_timesteps,
                        num_GNN_layers=args.num_hidden_layers,
                        nhead=1).to(args.device)
            optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

            # Model training
            best_model, train_time = train(cancer)

            print('Optimization Finished! Total time elapsed: {:.6f}'.format(train_time))
            # Restore best model for test set
            model.load_state_dict(torch.load('{}_{}.pth'.format(cancer, best_model)))
            test_acc, test_loss = compute_test(test_loader)
            print('Test set results, loss = {:.6f}, accuracy = {:.6f}'.format(test_loss, test_acc))
            test_loss_values.append(test_loss)
            if test_loss_values[-1] < min_loss:
                min_loss = test_loss_values[-1]
                best_times = i
                best_time_epoch = best_model
                torch.save(model.state_dict(),
                           os.getcwd() + '\\best_models\\{}_{}_{}.pth'.format(cancer, best_times, best_time_epoch))
            path = os.getcwd()
            accs += test_acc
            times += train_time

        print("Mean +*test acc = {:.6f}, Mean Train time = {:.6f}".format(accs / all_times, times / all_times))

        model.load_state_dict(torch.load(os.getcwd()
                              + '\\best_models\\{}_{}_{}.pth'.format(cancer, best_times, best_time_epoch)))
        evaluate_nodes(evaluate_loader)




