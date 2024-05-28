import cv2
import numpy as np
import os
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torchvision import transforms
from DoLR.model import DeepLabv3


# Assuming you have the true labels and segmented masks as numpy arrays
# true_labels.shape = (num_samples, height, width)
# segmented_masks.shape = (num_samples, height, width)
model_path='DoLR/NEW_MODEL256.pt'
input_path='Final_Scripts/Final_Data_256/Data'
label_path='Final_Scripts/Final_Data_256/Labels'
model_save='DoLR/Enc_Dec.pth'

class Prediction():
    def __init__(self, model_path):
        super().__init__()
        self.model=model_path
        self.transform=transforms.Compose([transforms.ToTensor(), 
                                           transforms.Resize((256,256))])
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
pred=Prediction(model_path=model_path)
    
true_labels=[]
segmented_masks=[]

for f in [f for f in os.listdir(label_path) if f.endswith('.tif')]:
    image=cv2.imread((os.path.join(label_path, f)))
    true_labels.append(image)

for f in [f for f in os.listdir(input_path) if f.endswith('.tif')]:
    path=os.path.join(input_path, f)
    image=pred(path)
    segmented_masks.append(image)
    


# Convert numpy arrays to PyTorch tensors
true_labels_tensor = torch.from_numpy(true_labels).float()
segmented_masks_tensor = torch.from_numpy(segmented_masks).float()

# Define a simple neural network for refinement
class RefinementModel(nn.Module):
    def __init__(self):
        super(RefinementModel, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(16, 1, kernel_size=3, padding=1)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        return x

# Create DataLoader for batch processing
dataset = TensorDataset(segmented_masks_tensor.unsqueeze(1), true_labels_tensor.unsqueeze(1))
dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

# Instantiate the model and define loss and optimizer
model = RefinementModel()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
num_epochs = 10
for epoch in range(num_epochs):
    for segmented_masks_batch, true_labels_batch in dataloader:
        optimizer.zero_grad()
        predicted_masks_batch = model(segmented_masks_batch)
        loss = criterion(predicted_masks_batch, true_labels_batch)
        loss.backward()
        optimizer.step()
    
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss.item()}")

# Now, you can use the trained model to refine your segmented masks
refined_masks = model(segmented_masks_tensor.unsqueeze(1)).squeeze(1).detach().numpy()
