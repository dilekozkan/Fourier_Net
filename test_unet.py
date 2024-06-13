import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
import os
import numpy as np
from Model import UNet
from heart_dataset import HeartDataset

def dice_coefficient(predicted, target):
    smooth=1.0
    predicted=predicted.contiguous().view(-1)
    target=target.contiguous().view(-1)
    intersection=(target*predicted).sum()
    return (2.*intersection+smooth)/(predicted.sum()+target.sum()+smooth)


#test the model
def test_model():
    save_path = "/kuacc/users/dozkan23/hpc_run/sixfolder/heart_unet_results"

    DATA_PATH="/kuacc/users/dozkan23/hpc_run/sixfolder"
    MODEL_SAVE_PATH ="/kuacc/users/dozkan23/hpc_run/UNET/unet_forheart.pth"
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    model = UNet(n_channels=1, n_classes=1).to(device)
    BATCH_SIZE = 16
    test_dataset=HeartDataset(DATA_PATH, tip="test")
    test_dataloader=DataLoader(dataset=test_dataset, 
                            batch_size=BATCH_SIZE,
                            shuffle=True)
    model.load_state_dict(torch.load(MODEL_SAVE_PATH)) # bunu oradan alabilir mi bilmiyorum
    model.eval()
    dice_score=0
    with torch.no_grad():
    #test the model and save the outputs
        for idx, sample_gt in enumerate(tqdm(test_dataloader)):
            sample=sample_gt[0].to(device)  #float demeye gerek var mı? bunlardan ne zaman bilicem
            gt=sample_gt[1].to(device)

            
            prediction=model(sample)
            prediction = torch.sigmoid(prediction)
    # Apply threshold to get binary prediction
            prediction = (prediction >= 0.5)
            dice_score+=dice_coefficient(prediction, gt).item()
            # Processing and saving images
            for i in range(sample.size(0)):
                pred = prediction[i].squeeze().cpu().numpy()
                gt_mask = gt[i].squeeze().cpu().numpy()
                sample_img = sample[i].squeeze().cpu().numpy()

                # to visaulize multiply with 255 and convert to image format (necessary uint8 format)
                pred_img = (pred * 255).astype(np.uint8)
                gt_img = (gt_mask * 255).astype(np.uint8)
                sample_img = (sample_img * 255).astype(np.uint8)

                # Stack images horizontally
                combined_img = np.hstack((sample_img, gt_img, pred_img))
                
                # Save image
                plt.imsave(os.path.join(save_path, f"test_{idx}_{i}.png"), combined_img, cmap='gray')

        dice_score=dice_score/(idx+1)
        print(dice_score)

        
test_model()
            

        


