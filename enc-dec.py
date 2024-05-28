import os
import glob
import cv2
import time
from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as f
from torchvision import transforms
from PIL import Image
from DoLR.model import DeepLabv3
from DoLR.metrics import mean_IoU, pixel_accuracy
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim

INPUT_SIZE=256
EPOCHS=500
model_path='DoLR/NEW_MODEL256.pt'
input_path='Final_Scripts/Final_Data_256/Data'
label_path='Final_Scripts/Final_Data_256/Labels'
model_save='DoLR/Enc_Dec.pth'

class Prediction():
    def __init__(self, model_path):
        super().__init__()
        self.model=model_path
        self.transform=transforms.Compose([transforms.ToTensor(), 
                                           transforms.Resize((INPUT_SIZE,INPUT_SIZE))])
        self.device=('cuda' if torch.cuda.is_available() else 'cpu')

    def prediction(self, image_path):
        image=Image.open(image_path).convert('RGB')
        image=self.transform(image)
        # image=torch.tensor(image).unsqueeze(0)
        image=image.unsqueeze(0)
        image=image.to(self.device)
        model=DeepLabv3()
        checkpoint=torch.load(self.model)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(self.device)
        with torch.no_grad():
            output=model(image)
            output=torch.argmax(output, dim=1)
        
        output=output.squeeze(0).cpu().numpy()
        output=output*255
        output=np.uint8(output)
        kernel=np.ones((5,5),np.uint8)
        final_image=255-cv2.dilate(255-output, kernel, iterations=1)

        return final_image
    


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        super().__init__()
        self.en_conv1=nn.Conv2d(1,16,3,padding=1)
        self.en_conv2=nn.Conv2d(16,4,3,padding=1)
        self.pool=nn.MaxPool2d(2,2)
        self.de_conv1=nn.ConvTranspose2d(4,16,2,stride=2)
        self.de_conv2=nn.ConvTranspose2d(16,1,2,stride=2)

    def forward(self,x):
        x=self.en_conv1(x)
        x=f.relu(x)
        x=self.pool(x)
        x=self.en_conv2(x)
        x=f.relu(x)
        x=self.pool(x)
        x=self.de_conv1(x)
        x=f.relu(x)
        x=self.de_conv2(x)
        x=f.sigmoid(x)
        return x
    


# class MyDataset(Dataset):
#     def __init__(self, input_path, label_path, prediction):
#         super().__init__()
#         self.input_path=input_path
#         self.label_path=label_path
#         image_list=os.listdir(self.input_path)
#         self.data=[]
#         for f in image_list:
#             img_path=os.path.join(self.input_path, f)
#             labels_path=os.path.join(self.label_path, f)
#             self.data.append([img_path, labels_path])
#         self.img_dim=(256, 256)
#     def __len__(self):
#         return len(self.data)
    
#     def __getitem__(self, index):
#         img_path, label_path=self.data[index]
#         image=prediction.prediction(img_path)
#         label=cv2.imread(label_path)
#         label=cv2.resize(self.img_dim)
#         image=torch.from_numpy(image)
#         label=torch.from_numpy(image)
#         # image=image.permute(2,0,1)
#         # label=label.permute(2,0,1)
#         return image, label




class MyDataset(Dataset):
    def __init__(self, input_path, label_path, prediction):
        super().__init__()
        self.input_path=input_path
        self.label_path=label_path
        image_list=os.listdir(self.input_path)
        self.data=[]
        for f in image_list:
            img_path=os.path.join(self.input_path, f)
            labels_path=os.path.join(self.label_path, f)
            self.data.append([img_path, labels_path])
        self.img_dim=(256, 256)
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        img_path, label_path=self.data[index]
        image=cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)  # Convert to grayscale
        label=cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)  # Convert to grayscale
        label=cv2.resize(label, self.img_dim)  # Resize label

        image=torch.from_numpy(image)
        label=torch.from_numpy(label)

        # Expand dimensions to match the expected input shape of the model
        image = image.unsqueeze(0).float()
        label = label.unsqueeze(0).float()

        return image, label
    
# def train(model, optimizer, criterion, n_epoch, data_loaders: dict, device):
    
#     train_losses = np.zeros(n_epoch)
#     val_losses = np.zeros(n_epoch)
#     model.to(device)

#     since = time.time()

#     for epoch in range(n_epoch):
#         train_loss = 0.0
#         train_accuracy = 0.0
#         train_iou = 0.0

#         model.train()
#         for inputs, targets in tqdm(data_loaders, desc=f'Training... Epoch: {epoch + 1}/{EPOCHS}'):
#             inputs, targets = inputs.to(device).float(), targets.to(device).float()

#             optimizer.zero_grad()

#             outputs = model(inputs)
#             loss = criterion(outputs, targets)
#             train_loss += loss.item()
#             train_accuracy += pixel_accuracy(outputs, targets)
#             train_iou += mean_IoU(outputs, targets)

#             loss.backward()
#             optimizer.step()

#         train_loss = train_loss / len(data_loaders)
#         train_accuracy = train_accuracy / len(data_loaders)
#         train_iou = train_iou / len(data_loaders)



#         # save epoch losses
#         train_losses[epoch] = train_loss

#         print(f"Epoch [{epoch+1}/{n_epoch}]:")
#         print(
#             f"Train Loss: {train_loss:.4f}, Train Pixel Accuracy: {train_accuracy:.4f}, Train IOU: {train_iou:.4f}")
#         print('-'*20)

    
#     # 
#     time_elapsed = time.time() - since
#     print('Training completed in {:.0f}m {:.0f}s'.format(
#         time_elapsed // 60, time_elapsed % 60))






def train(model, optimizer, criterion, n_epoch, data_loader, device):
    model.train()
    for epoch in range(n_epoch):
        epoch_loss = 0
        for batch_idx, (inputs, targets) in enumerate(data_loader):
            inputs, targets = inputs.to(device), targets.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        print(f"Epoch {epoch+1}/{n_epoch}, Loss: {epoch_loss/len(data_loader)}")



if __name__== '__main__':

    dataset=MyDataset
    prediction=Prediction(model_path=model_path)

    dataloader=DataLoader(dataset=MyDataset(label_path=label_path, input_path=input_path, prediction=prediction), 
                          batch_size=4, shuffle=True, drop_last=True)

    model=Model()
    criterion=nn.HuberLoss()
    optimizer=optim.Adam(model.parameters(), lr=1e-2)

    device='cuda'
    model.to(device)

    train(model, optimizer=optimizer, criterion=criterion, n_epoch=EPOCHS, 
          data_loader=dataloader, device=device)

    torch.save(model.state_dict(), model_save)



