import os, random
# import matplotlib.pyplot as plt
import numpy as np
import torch
import cv2 as cv
# import glob
# from PIL import Image
from os import listdir

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, datasets, models
from torch.nn.modules.loss import CrossEntropyLoss

from torchsummary import summary
import torch.nn as nn
import pytorch_cascaded


from collections import defaultdict
import torch.nn.functional as F
# from loss import dif_loss
from skimage import measure


seed = 1234
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cudnn.deterministic = True 
torch.backends.cudnn.benchmark = False 

# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"]='1'; 

fold = 1
run = 1
num_task = 2 # 1to 2  #bu ne anlama geliyor?
batch_size = 1
num_class = 2
feature_map_size = 16   #dogru mu diye kontrol edilebilir
#criterion = pytorch_unet.PHLoss()

model_name = 'unet_fold{}_run{}'.format(fold, run)
snapshot_path = './workingmodels'

if not os.path.exists(snapshot_path):
    os.makedirs(snapshot_path)


transforms_applied = [transforms.ToTensor()]
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

dataPath = '/userfiles/cgunduz/datasets/Henle/' #degistir burayı
savePathPrefix = './'

trInputPath = dataPath + 'images/tr/'
valInputPath = dataPath + 'images/val/'
tsInputPath = dataPath + 'images/ts/'

outputPathtr = dataPath + 'golds-binary/'
outputPathval = dataPath +  'golds-binary/'
outputPathts = dataPath +  'golds-binary/' #testleri mi elimizde yoktu neden böyle boş kaldı, basta '' böeyleydi sadece

fdmapPath = '/kuacc/users/dozkan23/hpc_run/fdmaps_outer/' #check again
imagePostfix = '.png'


class ImageDataset(Dataset):
    def __init__(self, inputPath, outputPath, transforms_=None):
        self.transform = transforms.Compose(transforms_)
        self.names = listAllOCTFiles(inputPath, imagePostfix)

        #"selfname" #std liler geliyo burdan (tr test val std yazıyor.)
        
        self.files_A = [(os.path.join(inputPath, os.path.basename(x + imagePostfix))) for x in self.names]
        if outputPath != '':#buraya giriyor

            self.files_B = [(os.path.join(outputPath, os.path.basename(x[:-3]+"hfl" + imagePostfix))) for x in self.names]

    

        else:
            
            self.files_B = []  #bossa bos kalsın demek oldu bu
               

    def __getitem__(self, index):
        image_A = cv.imread(self.files_A[index], -1) #original image values  #image'i okuyor
        image_A = (image_A - image_A.mean()) / image_A.std()   
    
        if self.files_B != []: #outputu varsa yani
            golds = []

            image_B = cv.imread(self.files_B[index], 0)  #  outputu oku
            image_B_gold = (image_B> 0).astype(np.int_) 
            golds.append(torch.tensor(image_B_gold))    #1 256 512 oldu (1i ekledik channel degeri olarak)
           
            
            for i in range(1, num_task):  #burdaki ne döngüsü bi garip geldi.
                fd_map = np.loadtxt(fdmapPath + self.names[index][:-4] + 'fdmap' + str(i))  #benim isimlendirmem neyse burayı düzelt
                fd_map = (fd_map - fd_map.mean()) / (fd_map.std()) #fd mapler normalize edildi
                golds.append(torch.tensor(fd_map).unsqueeze(0)) 
                
        else:
            golds = []

        if self.transform:
            item_A = self.transform(image_A)
        
#         return [item_A, golds]
        return [item_A, golds, self.files_A[index]]  #ne ne ne döndürdü: 1)bizim okunan resimlerimiz, golds (benm anladıgım tüm goldslar döndü), files_a'nın hepsi bence yine

    def __len__(self):
        return len(self.files_A)


def listAllOCTFiles(imageDirPath, imagePostfix):
    fileList = listdir(imageDirPath)
    postLen = len(imagePostfix)
    imageNames = []
    for i in range(len(fileList)):
        if fileList[i][-postLen::] == imagePostfix: #png dosyası ise demek istemiş yani
            imageNames.append(fileList[i][:-postLen])
    return imageNames


num_workers = 1  #ne anlama geliyor
# Training data loader
train_loader = DataLoader(
    ImageDataset(trInputPath, outputPathtr, transforms_=transforms_applied),
    batch_size=batch_size,
    shuffle=True,
    num_workers=num_workers
)
# Validation data loader
val_loader = DataLoader(
    ImageDataset(valInputPath, outputPathval, transforms_=transforms_applied),
    batch_size=batch_size,
    shuffle=True,
    num_workers=num_workers
)
# Test data loader
test_loader = DataLoader(
    ImageDataset(tsInputPath, outputPathts, transforms_=transforms_applied), #testin gold degeri mi yok elimizde neden böyle 
    batch_size=batch_size,
    shuffle=False,
    num_workers=num_workers
)

dataloaders = {
    'train': train_loader,
    'val': val_loader
}

# mean_center_difs = 
# for inputs, labels in train_loader:
#     labels[1].numpy()
    


def calc_loss(pred1,pred2, labels, metrics, device, bce_weight=0.5):
#     bce = F.binary_cross_entropy_with_logits(pred, target, weights)
#     bce = F.binary_cross_entropy_with_logits(pred, target)
#     bce = nn.BCEWithLogitsLoss()(pred, target)
    labels[1] = labels[1].float()  #hata alıyordum bu 2 satırı ben ekledim
    labels[0] = labels[0].long() 
    mse= nn.MSELoss()(pred1, labels[1]) #check again!    
    bce = CrossEntropyLoss()(pred2, labels[0]) #check again #labels shape kontrol sqe
        
#     pred = F.sigmoid(pred)
#     pred = F.softmax(pred, dim = 1)
#     dice = dice_loss(pred, target)
    
#     loss = bce * bce_weight + dice * (1 - bce_weight)
    metrics['bce'] += bce.data.cpu().numpy() * labels[0].size(0) #total not avergae
    metrics['mse'] += mse.data.cpu().numpy() * labels[1].size(0)#total not avergae
    loss= bce+mse
    metrics['loss'] += loss.data.cpu().numpy() * labels[0].size(0)
    
    return loss

"""if ph:
    ph_loss = criterion(pred, target, device)
    metrics['ph'] += ph_loss.data.cpu().numpy() * target.size(0)
    loss = ph_loss
else:
    loss = bce"""
    


def print_metrics(metrics, epoch_samples, phase):    
    outputs = []
    for k in metrics.keys():
        outputs.append("{}: {:4f}".format(k, metrics[k] / epoch_samples))
        
    print("{}: {}".format(phase, ", ".join(outputs)))   
    
    
dtype = torch.FloatTensor

def train_model(model, optimizer, scheduler, num_epochs=25):
    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = 1e15
    #mse_loss = nn.MSELoss()
    
#     f = open('thickness.txt', 'w')
    for epoch in range(1, num_epochs+1):
        print('Epoch {}/{}'.format(epoch, num_epochs))
        print('-' * 10)
        
        since = time.time()

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
#                 scheduler.step()
                for param_group in optimizer.param_groups:
                    print("LR", param_group['lr'])
                    
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            metrics = defaultdict(float)
            epoch_samples = 0
            
#             for inputs, labels in dataloaders[phase]:
            for inputs, labels, names in dataloaders[phase]:
                inputs = inputs.type(dtype).to(device)
                
                labels = [x.long().to(device) for x in labels] #list

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
#                     output, out_list = model(inputs)
                    output1,output2 = model(inputs) #2 return
                    loss = calc_loss(output1, output2, labels, metrics, device)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                epoch_samples += inputs.size(0)

            print_metrics(metrics, epoch_samples, phase)
#             epoch_loss = (metrics['loss'] + metrics['loss2']) / epoch_samples
            epoch_loss = (metrics['loss']) / epoch_samples

            # deep copy the model
            if phase == 'val' and epoch_loss <= best_loss:
                print("saving best model")
                best_loss = epoch_loss
                best_model_wts = copy.deepcopy(model.state_dict())
                torch.save(best_model_wts, './workingmodels/' + model_name)
                
            if phase == 'val':
                valid_loss = epoch_loss
                scheduler.step(epoch_loss)
                
#             model_wts = copy.deepcopy(model.state_dict())
#             torch.save(model_wts, './models/' + model_name + f'_{epoch}')

        time_elapsed = time.time() - since
        print('{:.0f}m {:.0f}s\n'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val loss: {:4f}'.format(best_loss))

    # load best model weights
    model.load_state_dict(best_model_wts)
#     f.close()
    return model


import torch.optim as optim
from torch.optim import lr_scheduler
import time
import copy


model = pytorch_cascaded.cascaded(num_class, feature_map_size=feature_map_size) #multitask , task_no=num_task-1
# summary(model, input_size=(1, 128, 1024), device="cpu")
model = model.to(device)

# if cnt:
#     model.load_state_dict(torch.load('./models/' + model_name))


# model.load_state_dict(torch.load('./models/UNet_fold4_run1_40'))


optimizer_ft = optim.Adam(model.parameters(), lr=1e-3)
# optimizer_ft = optim.Adadelta(filter(model.parameters()), lr=1e-1)

# exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=60, gamma=0.1) 
exp_lr_scheduler = lr_scheduler.ReduceLROnPlateau(optimizer_ft, mode='min', factor=0.1, patience=20) 
#         
model = train_model(model, optimizer_ft, exp_lr_scheduler, num_epochs=40)
