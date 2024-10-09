import random
import torch, os
import torch.cuda
import torch.nn as nn
from torch import Tensor
import torchvision
import numpy as np
from torchvision.transforms import v2
from torch.utils.data import DataLoader, SubsetRandomSampler
from torchvision.datasets import ImageFolder
from torch.utils.data import Subset
from torch.utils.data import default_collate
from PIL import Image
from sklearn.model_selection import StratifiedShuffleSplit, KFold
from torch.utils.tensorboard import SummaryWriter
from torchvision.models import resnet18, resnet34, resnext50_32x4d
from sklearn.metrics import confusion_matrix, roc_auc_score, average_precision_score, f1_score, recall_score,matthews_corrcoef

class Ealrystopping():
    '''早停法'''

    def __init__(self, patience, verbose=True, delta=0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_loss = np.inf
        self.early_stop = False
        self.delta = delta

    def __call__(self, val_loss):
        if val_loss > self.best_loss:
            self.counter += 1
            if self.verbose:
                print(f'Loss increased to {val_loss},Counter + 1')
        else:
            self.best_loss = val_loss
            self.counter = 0
            if self.verbose:
                print(f'Loss decreases to {val_loss},Counter set to 0')
        if self.counter >= self.patience:
            self.early_stop = True
            self.counter = 0
            if self.verbose:
                print(f'Reach to the max patience,Epoch break')

def train_model(model, optim, loss_func, train_loaders, test_loaders, epochs, scheduler=None):
    '''模型训练和测试'''
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    loss_func = loss_func.to(device)
    early_stopping = Ealrystopping(patience=15, verbose=True)

    best_score = -np.inf

    model.train()
    for step, epoch in enumerate(range(epochs)):
        print(f'Epoch:{step}\n')
        losses = 0.0
        for imgs, labels in train_loaders:
            imgs = imgs.to(device)
            labels = labels.to(device)

            optim.zero_grad()
            outputs = model(imgs)
            loss = loss_func(outputs, labels)
            loss.backward()
            optim.step()
            losses += loss.item() * imgs.size(0)
        mean_losses = losses / len(train_loaders.dataset)

        test_accuracy = epoch_test(model, loss_func, device, test_loaders)
        train_accuracy = epoch_train(model, loss_func, device, train_loaders)
        with open('result.txt', 'a') as f:
            f.write(f'Epoch:{step}->Test:{test_accuracy};Train:{train_accuracy};Losses:{mean_losses}\n')

        '''保存最佳模型'''
        if test_accuracy > best_score:
            best_score = test_accuracy
            parameters = model.state_dict()
            torch.save(parameters, 'model.pth')
        '''学习率衰减'''
        scheduler.step()
        '''早停'''
        early_stopping(mean_losses)
        if early_stopping.early_stop:
            break

    return model.state_dict()


def epoch_test(model, loss_func, device, test_loaders) -> float:
    model.eval()

    total = 0
    correct = 0
    losses = 0.0
    with torch.no_grad():
        for imgs, labels in test_loaders:
            imgs = imgs.to(device)
            labels = labels.to(device)
            outputs = model(imgs)
            # loss = loss_func(outputs,labels)
            # losses += loss.item()*imgs.size(0)

            predictions = Tensor.cpu(outputs.argmax(1)).data.numpy()
            total += labels.size(0)
            labels = Tensor.cpu(labels).data.numpy()
            correct += (predictions == labels).astype(int).sum()

        # mean_losses = losses/len(test_loaders.dataset)
        test_accuracy = round(correct / total, 4)
        return test_accuracy


def epoch_train(model, loss_func, device, train_loaders):
    model.eval()

    total = 0
    correct = 0
    losses = 0.0
    with torch.no_grad():
        for imgs, labels in train_loaders:
            imgs = imgs.to(device)
            labels = labels.to(device)
            outputs = model(imgs)
            # loss = loss_func(outputs, labels)
            # losses += loss.item() * imgs.size(0)

            predictions = Tensor.cpu(outputs.argmax(1)).data.numpy()
            total += labels.size(0)
            labels = Tensor.cpu(labels).data.numpy()
            correct += (predictions == labels).astype(int).sum()

        # mean_losses = losses / len(test_loaders.dataset)
        train_accuracy = round(correct / total, 4)
        return train_accuracy


def val_acc(model, loss_func, device, k_fold, batch_size, val_datasets, val_loaders):
    model.eval()

    '''交叉验证'''
    subset_indices = list(range(len(val_datasets)))
    fold_size = len(val_datasets) // k_fold
    fold_indices = [subset_indices[i * fold_size:(i + 1) * fold_size] for i in range(k_fold)]

    val_accuracy = []
    for fold in range(k_fold):
        val_indices = fold_indices[fold]

        val_sampler = SubsetRandomSampler(val_indices)

        val_loader = DataLoader(val_datasets, batch_size=batch_size, sampler=val_sampler, drop_last=False)

        fold_accuracy = epoch_test(model, loss_func, device, val_loader)
        val_accuracy.append(fold_accuracy)

        with open('result.txt', 'a') as f:
            f.write(f'Fold val score : {fold_accuracy}\n')

    val_accuracy = round(np.mean(val_accuracy), 4)
    with open('result.txt', 'a') as f:
        f.write(f'The final val score : {val_accuracy}\n')

    '''全部验证'''
    total = 0
    correct = 0
    with torch.no_grad():
        count = 0
        for imgs, labels in val_loaders:
            imgs = imgs.to(device)
            labels = labels.to(device)
            outputs = model(imgs)

            probabilities = nn.functional.softmax(outputs, dim=1)
            predictions = Tensor.cpu(outputs.argmax(1)).data.numpy()

            total += labels.size(0)
            labels = Tensor.cpu(labels).data.numpy()
            correct += (predictions == labels).astype(int).sum()

            if count == 0:
                labels_total = labels
                pred_total = predictions
                score_total = Tensor.cpu(probabilities).data.numpy()
            else:
                labels_total = np.append(labels_total, labels, axis=0)
                pred_total = np.append(pred_total, predictions, axis=0)
                score_total = np.append(score_total, Tensor.cpu(probabilities).data.numpy(), axis=0)
            count += 1

        val_accuracy = round(correct / total, 4)
        return labels_total, pred_total, score_total, val_accuracy


def access(labels_total_np, pred_total_np, score_total_np):
    test_labels_array = labels_total_np  # 测试集的label
    cbc_pred_test_array = pred_total_np  # 测试集预测的结果
    cbc_pred_test_proba = score_total_np  # 模型对测试集的打分

    cm = confusion_matrix(test_labels_array, cbc_pred_test_array)

    # 计算每个类别的 Recall 和 Specificity
    recall_per_class = recall_score(test_labels_array, cbc_pred_test_array, average=None)
    recall_per_class = np.array2string(recall_per_class, separator=',')

    specificity_per_class = []
    for i in range(len(np.unique(test_labels_array))):
        tn = cm.sum() - (cm[i, :].sum() + cm[:, i].sum() - cm[i, i])
        fp = cm[:, i].sum() - cm[i, i]
        fn = cm[i, :].sum() - cm[i, i]
        specificity = tn / (tn + fp) if (tn + fp) != 0 else 0
        specificity_per_class.append(specificity)

    # 计算 Matthews Correlation Coefficient
    mcc = matthews_corrcoef(test_labels_array, cbc_pred_test_array)

    # 计算 F1 Score
    f1 = f1_score(test_labels_array, cbc_pred_test_array, average='macro')

    # 计算 ROC AUC 和 PRC
    roc_auc = roc_auc_score(test_labels_array, cbc_pred_test_proba, multi_class='ovr', average='macro')

    prc = average_precision_score(test_labels_array, cbc_pred_test_proba, average='macro')

    cm = np.array2string(cm, separator=',', threshold=np.inf, max_line_width=np.inf)

    # 将指标写入文件
    with open('test_output.txt', 'a') as open_output:
        open_output.write(f'Confusion Matrix:\n{cm}\n')
        open_output.write(f'Per-class Recall (TPR):\n{recall_per_class}\n')
        open_output.write(f'Per-class Specificity (TNR):\n{specificity_per_class}\n')
        open_output.write(f'MCC: {mcc}\n')
        open_output.write(f'F1 Score: {f1}\n')
        open_output.write(f'ROC AUC: {roc_auc}\n')
        open_output.write(f'PRC: {prc}\n')


def data(root, train_preprocess, test_preprocess, test_size, batch_size):
    datasets = ImageFolder(root=root, transform=train_preprocess)
    test_val_datasets = ImageFolder(root=root, transform=test_preprocess)

    X = np.arange(len(datasets))
    y = np.array(datasets.targets)

    train_split = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=42)
    val_split = StratifiedShuffleSplit(n_splits=1, test_size=0.5, random_state=42)
    '''输入按标签顺序的索引以及标签值,输出划分出来的两个索引'''
    train_idx, temp_idx = next(iter(train_split.split(X, y)))

    temp_targets = y[temp_idx]

    val_idx, test_idx = next(iter(val_split.split(temp_idx, temp_targets)))
    val_idx = temp_idx[val_idx]
    test_idx = temp_idx[test_idx]

    '''根据索引划分数据集'''
    train_datasets = Subset(datasets, train_idx)
    test_datasets = Subset(test_val_datasets, test_idx)
    val_datasets = Subset(test_val_datasets, val_idx)

    '''训练集打破顺序有助于训练,测试和验证保持顺序益与发现问题'''
    train_loaders = DataLoader(train_datasets, batch_size=batch_size, shuffle=True, drop_last=False, num_workers=4)
    test_loaders = DataLoader(test_datasets, batch_size=batch_size, shuffle=False, drop_last=False, num_workers=4)
    val_loaders = DataLoader(val_datasets, batch_size=batch_size, shuffle=False, drop_last=False, num_workers=4)

    return train_loaders, test_loaders, val_loaders, val_datasets


def replace_relu_with_elu(module):
    for name, child in module.named_children():
        if isinstance(child, nn.ReLU):
            setattr(module, name, nn.ELU())
        else:
            replace_relu_with_elu(child)

if __name__ == '__main__':
    BATCH_SIZE = 64
    LR = 0.01
    TEST_SIZE = 0.2
    EPOCH = 250
    K_FOLD = 5

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    root = r".\datasets"
    train_preprocess = v2.Compose(
        [
            v2.ToImage(),
            v2.ToDtype(torch.uint8, scale=True),
            v2.Resize([250,250],antialias=True),
            v2.CenterCrop([224, 224]),
            v2.RandomRotation((0, 45)),
            v2.RandomVerticalFlip(0.5),
            v2.RandomHorizontalFlip(0.5),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    '''测试集和验证集不应该包含任何预处理,除了裁剪和归一化'''
    test_preprocess = v2.Compose(
        [
            v2.ToImage(),
            v2.ToDtype(torch.uint8, scale=True),
            v2.Resize([250, 250], antialias=True),
            v2.CenterCrop([224, 224]),
            #v2.Grayscale(num_output_channels=1),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )


    train_loaders, test_loaders, val_loaders, val_datasets = data(root, train_preprocess, test_preprocess, TEST_SIZE,
                                                                  BATCH_SIZE)


    '''加载模型,并微调'''
    resnet = resnet34(weights=None)
    #resnet.conv1 = nn.Conv2d(1,64,kernel_size=(7,7),stride=(2,2),padding=(3,3),bias=False)
    inputs = resnet.fc.in_features
    resnet.fc = nn.Linear(inputs, 36, True)
    replace_relu_with_elu(resnet)

    print(resnet)

    optim = torch.optim.Adam(resnet.parameters(),lr=LR)
    loss_func = nn.CrossEntropyLoss()
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optim,step_size=20, gamma=0.5)

    train_model(resnet, optim, loss_func, train_loaders, test_loaders, EPOCH, scheduler=lr_scheduler)

    resnet.load_state_dict(torch.load(r'.\model.pth'))

    labels_total, pred_total, score_total, val_accuracy = val_acc(resnet, loss_func, device, K_FOLD, BATCH_SIZE,
                                                                  val_datasets, val_loaders)

    access(labels_total, pred_total, score_total)





















