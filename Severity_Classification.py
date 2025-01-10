import os
import random
import pandas as pd
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from torchvision import datasets, transforms, models 
from torchvision.datasets import ImageFolder
from torchvision.transforms import ToTensor
from torchvision.utils import make_grid
from torch.utils.data import random_split
from torch.utils.data.dataloader import DataLoader
import matplotlib.pyplot as plt

def SCNNPNN():
    data_dir = 'The IQ-OTHNCCD lung cancer dataset\The IQ-OTHNCCD lung cancer dataset'
    Name0 = os.listdir(data_dir)
    print(Name0)
    Name1=['Normal cases', 'Bengin cases', 'Malignant cases']
    Name=sorted(Name1)
    Name
    N=list(range(len(Name)))    
    normal_mapping=dict(zip(Name,N)) 
    reverse_mapping=dict(zip(N,Name)) 
    dataset=[]
    for i in tqdm(range(len(Name))):
        path=os.path.join(data_dir,Name[i])
        for im in os.listdir(path):          
            labeli=normal_mapping[Name[i]]
            img1=cv2.imread(os.path.join(path,im))
            img2=cv2.resize(img1,dsize=(100,100),interpolation=cv2.INTER_CUBIC)
            img3=img2.astype(np.float32)
            image=torch.from_numpy(img3).permute(2,0,1) 
            dataset+=[[image,labeli]]
    dataset[100]
    img, label = dataset[100]
    print(img.shape)
    print(label)
    def show_image(img,label):
        img2=img.permute(1,2,0).numpy().astype(int)
        print(img2.shape)
        print(reverse_mapping[label])
        plt.imshow(img2)
    show_image(*dataset[20])
    torch.manual_seed(20)
    val_size = len(dataset)//10
    test_size = len(dataset)//5
    train_size = len(dataset) - val_size - test_size
    train_ds, val_ds, test_ds = random_split(dataset, [train_size, val_size, test_size])
    len(train_ds), len(val_ds), len(test_ds) 
    batch_size = 64
    train_loader = DataLoader(train_ds, batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size, num_workers=4, pin_memory=True)
    m=len(dataset)
    M=list(range(m))
    random.seed(2021)
    random.shuffle(M)
    dataset[0][0]
    fig, axs = plt.subplots(4,4,figsize=(15,15))
    for i in range(16):
        r=i//4
        c=i%4
        img, label = dataset[M[i]]
        img2=img.permute(1,2,0).numpy().astype(int)
        ax=axs[r][c].axis("off")
        ax=axs[r][c].set_title(reverse_mapping[label])
        ax=axs[r][c].imshow(img2)
    plt.show()
    def accuracy(outputs, labels):
        _, preds = torch.max(outputs, dim=1)
        return torch.tensor(torch.sum(preds == labels).item() / len(preds))
    class ImageClassificationBase(nn.Module):
        def training_step(self, batch):
            images, labels = batch 
            out = self(images)                  
            loss = F.cross_entropy(out, labels) 
            return loss
        def validation_step(self, batch):
            images, labels = batch 
            out = self(images)                    
            loss = F.cross_entropy(out, labels)   
            acc = accuracy(out, labels)           
            return {'val_loss': loss.detach(), 'val_acc': acc}
        def validation_epoch_end(self, outputs):
            batch_losses = [x['val_loss'] for x in outputs]
            epoch_loss = torch.stack(batch_losses).mean()   
            batch_accs = [x['val_acc'] for x in outputs]
            epoch_acc = torch.stack(batch_accs).mean()      
            return {'val_loss': epoch_loss.item(), 'val_acc': epoch_acc.item()}
        def epoch_end(self, epoch, result):
            print("Epoch [{}], train_loss: {:.4f}, val_loss: {:.4f}, val_acc: {:.4f}".format(
                epoch, result['train_loss'], result['val_loss'], result['val_acc']))
    def evaluate(model, val_loader):
        outputs = [model.validation_step(batch) for batch in val_loader]
        return model.validation_epoch_end(outputs)
    def fit(epochs, lr, model, train_loader, val_loader, opt_func=torch.optim.SGD):
        history = []
        optimizer = opt_func(model.parameters(), lr)
        for epoch in range(epochs):
            model.train()
            train_losses = []
            for batch in tqdm(train_loader):
                loss = model.training_step(batch)
                train_losses.append(loss)
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
            result = evaluate(model, val_loader)
            result['train_loss'] = torch.stack(train_losses).mean().item()
            model.epoch_end(epoch, result)
            history.append(result)
        return history
    torch.cuda.is_available()
    def get_default_device():
        """Pick GPU if available, else CPU"""
        if torch.cuda.is_available():
            return torch.device('cuda')
        else:
            return torch.device('cpu')
    def to_device(data, device):
        """Move tensor(s) to chosen device"""
        if isinstance(data, (list,tuple)):
            return [to_device(x, device) for x in data]
        return data.to(device, non_blocking=True)
    class DeviceDataLoader():
        """Wrap a dataloader to move data to a device"""
        def __init__(self, dl, device):
            self.dl = dl
            self.device = device
        def __iter__(self):
            """Yield a batch of data after moving it to device"""
            for b in self.dl: 
                yield to_device(b, self.device)
        def __len__(self):
            """Number of batches"""
            return len(self.dl)
    device = get_default_device()
    device
    train_loader = DeviceDataLoader(train_loader, device)
    val_loader = DeviceDataLoader(val_loader, device)
    test_loader = DeviceDataLoader(test_loader, device)
    m=len(dataset)
    M=list(range(m))
    random.seed(2021)
    random.shuffle(M)
    input_size = 3*100*100
    output_size = len(Name)
    def accuracy(outputs, labels):
        _, preds = torch.max(outputs, dim=1)
        return torch.tensor(torch.sum(preds == labels).item() / len(preds))
    class ImageClassificationBase(nn.Module):
        def training_step(self, batch):
            images, labels = batch 
            out = self(images)                   
            loss = F.cross_entropy(out, labels)  
            return loss
        def validation_step(self, batch):
            images, labels = batch 
            out = self(images)                    
            loss = F.cross_entropy(out, labels)   
            acc = accuracy(out, labels)           
            return {'val_loss': loss.detach(), 'val_acc': acc}
        def validation_epoch_end(self, outputs):
            batch_losses = [x['val_loss'] for x in outputs]
            epoch_loss = torch.stack(batch_losses).mean()   
            batch_accs = [x['val_acc'] for x in outputs]
            epoch_acc = torch.stack(batch_accs).mean()      
            return {'val_loss': epoch_loss.item(), 'val_acc': epoch_acc.item()}
        def epoch_end(self, epoch, result):
            print("Epoch [{}], train_loss: {:.4f}, val_loss: {:.4f}, val_acc: {:.4f}".format(
                epoch, result['train_loss'], result['val_loss'], result['val_acc']))
    class CnnModel(ImageClassificationBase):
        def __init__(self):
            super().__init__()
            self.network = nn.Sequential(
                nn.Conv2d(3, 100, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.Conv2d(100, 150, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2, 2),
                nn.Conv2d(150, 200, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.Conv2d(200, 200, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2, 2),
                nn.Conv2d(200, 250, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.Conv2d(250, 250, kernel_size=3, stride=1, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2, 2),
                nn.Flatten(), 
                nn.Linear(36000, 6400),  
                nn.ReLU(),            
                nn.Linear(6400, 640),  
                nn.ReLU(),
                nn.Linear(640, 64),  
                nn.ReLU(),
                nn.Linear(64, 8),  
                nn.ReLU(),
                nn.Dropout(0.25),
                nn.Linear(8, output_size))
        def forward(self, xb):
            return self.network(xb)
    model = CnnModel()
    model.cuda()
    for images, labels in train_loader:
        print('images.shape:', images.shape)    
        out = model(images)      
        print('out.shape:', out.shape)
        break
    device = get_default_device()
    device
    train_dl = DeviceDataLoader(train_loader, device)
    val_dl = DeviceDataLoader(val_loader, device)
    to_device(model, device)
    @torch.no_grad()
    def evaluate(model, val_loader):
        model.eval()
        outputs = [model.validation_step(batch) for batch in val_loader]
        return model.validation_epoch_end(outputs)
    def fit(epochs, lr, model, train_loader, val_loader, opt_func=torch.optim.SGD):
        history = []
        optimizer = opt_func(model.parameters(), lr)
        for epoch in range(epochs):
            model.train()
            train_losses = []
            for batch in tqdm(train_loader):
                loss = model.training_step(batch)
                train_losses.append(loss)
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
            result = evaluate(model, val_loader)
            result['train_loss'] = torch.stack(train_losses).mean().item()
            model.epoch_end(epoch, result)
            history.append(result)
        return history
    model = to_device(CnnModel(), device)
    history=[evaluate(model, val_loader)]
    history
    num_epochs = 50
    opt_func = torch.optim.Adam
    lr = 0.001
    history+= fit(num_epochs, lr, model, train_dl, val_dl, opt_func)
    history+= fit(num_epochs, lr/10, model, train_dl, val_dl, opt_func)
    def plot_accuracies(history):
        accuracies = [x['val_acc'] for x in history]
        plt.plot(accuracies, '-x')
        plt.xlabel('epoch')
        plt.ylabel('accuracy')
        plt.title('Accuracy vs. No. of epochs')
        plt.show()
    def plot_losses(history):
        train_losses = [x.get('train_loss') for x in history]
        val_losses = [x['val_loss'] for x in history]
        plt.plot(train_losses, '-bx')
        plt.plot(val_losses, '-rx')
        plt.xlabel('epoch')
        plt.ylabel('loss')
        plt.legend(['Training', 'Validation'])
        plt.title('Loss vs. No. of epochs')
        plt.show()
    plot_accuracies(history)
    plot_losses(history)
    evaluate(model, test_loader)
    history.save("cnnmodel.h5")
