import os
import torch
import cv2 as cv
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import pytorch_cascaded

# Define the seed for reproducibility
seed = 1234
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.backends.cudnn.deterministic = True 
torch.backends.cudnn.benchmark = False 

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
num_task = 2
# Data paths
dataPath = '/userfiles/cgunduz/datasets/Henle/'
tsInputPath = dataPath + 'images/ts/'
outputPathts = dataPath + 'golds-binary/'  # Adjust this path if needed
fdmapPath = '/kuacc/users/dozkan23/hpc_run/fdmaps_outer/'
imagePostfix = '.png'

# Load the model
num_class = 2
feature_map_size = 16
model = pytorch_cascaded.cascaded(num_class, feature_map_size=feature_map_size)

# Use map_location to handle loading on CPU if GPU is not available
model.load_state_dict(torch.load('./workingmodels/unet_fold1_run1', map_location=torch.device('cpu')))
model = model.to(device)
model.eval()

# Dataset class
class ImageDataset(Dataset):
    def __init__(self, inputPath, outputPath, transforms_=None):
        self.transform = transforms.Compose(transforms_)
        self.names = self.list_all_files(inputPath, imagePostfix)
        self.files_A = [(os.path.join(inputPath, os.path.basename(x + imagePostfix))) for x in self.names]
        self.files_B = [(os.path.join(outputPath, os.path.basename(x[:-3] + "hfl" + imagePostfix))) for x in self.names] if outputPath else []

    def __getitem__(self, index):
        image_A = cv.imread(self.files_A[index], -1)
        image_A = (image_A - image_A.mean()) / image_A.std()
        image_A = image_A.astype(np.float32)
        golds = []

        if self.files_B:
            image_B = cv.imread(self.files_B[index], 0)
            image_B_gold = (image_B > 0).astype(np.int_)
            golds.append(torch.tensor(image_B_gold, dtype=torch.float32))
            for i in range(1, num_task):
                fd_map = np.loadtxt(fdmapPath + self.names[index][:-4] + 'fdmap' + str(i))
                fd_map = (fd_map - fd_map.mean()) / fd_map.std()
                golds.append(torch.tensor(fd_map, dtype=torch.float32).unsqueeze(0))
        if self.transform:
            item_A = self.transform(image_A)
        return [item_A, golds, self.files_A[index]]

    def __len__(self):
        return len(self.files_A)

    def list_all_files(self, imageDirPath, imagePostfix):
        fileList = os.listdir(imageDirPath)
        postLen = len(imagePostfix)
        imageNames = [file[:-postLen] for file in fileList if file.endswith(imagePostfix)]
        return imageNames

# Create DataLoader for test data
transforms_applied = [transforms.ToTensor()]
batch_size = 1
num_workers = 1
test_loader = DataLoader(
    ImageDataset(tsInputPath, outputPathts, transforms_=transforms_applied),
    batch_size=batch_size,
    shuffle=False,
    num_workers=num_workers
)

# Function to calculate Dice score
def dice_score(pred, target):
    smooth = 1e-5
    pred_flat = pred.view(-1)
    target_flat = target.view(-1)
    intersection = (pred_flat * target_flat).sum()
    return (2. * intersection + smooth) / (pred_flat.sum() + target_flat.sum() + smooth)

# Testing function
def test_model(model, test_loader, device):
    model.eval()
    dice_scores = []

    with torch.no_grad():
        for inputs, labels, names in test_loader:
            inputs = inputs.to(device)
            labels = [x.to(device) for x in labels]
            
            output1, output2 = model(inputs)
            pred = torch.argmax(output2, dim=1)
            
            dice = dice_score(pred, labels[0])
            dice_scores.append(dice.item())

            # Save predicted images
            pred_np = pred.squeeze().cpu().numpy().astype(np.uint8) * 255
            save_path = os.path.join('./predictions', os.path.basename(names[0]))
            cv.imwrite(save_path, pred_np)

    mean_dice = np.mean(dice_scores)
    print(f'Mean Dice Score: {mean_dice}')

# Run the test
if __name__ == '__main__':
    os.makedirs('./predictions', exist_ok=True)
    test_model(model, test_loader, device)
