

import tarfile
from torchvision.datasets.utils import download_url
import torchvision.transforms as tt
import torchvision
from torch.utils.data import DataLoader
import torchvision.transforms as T
import matplotlib.pyplot as plt
from torch.utils.data import random_split
import torch.nn.functional as F
from sklearn.metrics import f1_score,confusion_matrix,accuracy_score
import os
import torch
from PIL import Image
from torchvision import transforms
from tqdm import tqdm
from torch.utils.data import Dataset
from sklearn import preprocessing
import torch.nn as nn
import scipy.io
import pandas as pd
import numpy as np
from tqdm.notebook import tqdm
from efficientnet_pytorch import EfficientNet
import warnings
warnings.filterwarnings("ignore")

Data = '../Data'
modelname = 'efficientNet.pth'

print("###############################################################")
print("#############Testing is Started on " + modelname,"#############")
print("###############################################################")
class CustomDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.classes = sorted(os.listdir(root_dir))
        self.class_to_idx = {cls_name[:3]: idx for idx, cls_name in enumerate(self.classes)}
        self.images = self._get_images()

    def _get_images(self):
        images = []
        # print(self.class_to_idx)
        for class_folder in self.classes:
            class_path = os.path.join(self.root_dir, class_folder)
            class_idx = self.class_to_idx[class_folder[:3]]
            for img_name in os.listdir(class_path):
                img_path = os.path.join(class_path, img_name)
                images.append((img_path, class_idx))
        return images

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path, class_idx = self.images[idx]
        image = Image.open(img_path)
        if self.transform:
            image = self.transform(image)
        return image, class_idx


trans = tt.Compose([
    tt.ToTensor(),tt.Resize((224,224)),tt.Normalize((0.485,0.456,0.406),(0.229,0.224,0.225))   
])
test_dir = 'D:\\Project\\SML\\CUB-200-2011-dataset-main\\CUB-200-2011-dataset-main\\test'
data = CustomDataset(test_dir, transform=trans)
data[0]
batch = 128

test_dl = DataLoader( data,batch, shuffle=True)


def accuracy(outputs, labels):
    _, preds = torch.max(outputs, dim=1)
    return torch.tensor(torch.sum(preds == (labels)).item() / len(preds))


class Model(nn.Module):
  def __init__(self ):
    super().__init__()
    model = EfficientNet.from_pretrained('efficientnet-b4')
    model._fc = nn.Linear(model._fc.in_features, 200)
    self.network =model

  def forward(self, x):
    return self.network(x)
  
  def train_step(self,batch):
      img , label = batch
      pred = self(img)
      loss = F.cross_entropy(pred,label)
      return loss
  
  def valid_step(self,batch):
      img , label = batch
      pred = self(img)
      loss = F.cross_entropy(pred, label)  
      acc = accuracy(pred, label)
      return {'acc':acc , 'loss':loss.item()}

  def valid_metric_cals(self,output):
    loss = [i['loss'] for i in output]
    acc = [i['acc'] for i in output]
    valid_loss = np.mean(loss)
    valid_acc = np.mean(acc)
    return {'val_loss':valid_loss,'valid_acc':valid_acc}
  
  def epoch_end(self, epoch_no , result):
    #wandb.log({ 'epoch' : epoch_no+1, 'training loss': result["train_loss"], 'validation loss': result["val_loss"], 'accuracy': result["valid_acc"]})
    print(f'epoch :{epoch_no+1}, training loss: {result["train_loss"]}, validation loss: {result["val_loss"]}, accuracy: {result["valid_acc"]}')


@torch.no_grad()
def evalu(model, val_dl):
  model.eval()
  temp = [ model.valid_step(j) for j in val_dl ]
  return model.valid_metric_cals(temp)

try:
  model = Model()
  model.load_state_dict(torch.load("../Model/"+modelname,map_location=torch.device('cpu')))

  @torch.no_grad()
  def test_accuracy_f1(test_dl):
    model.eval()
    acc=0
    f1=0
    list_label = []
    list_pred = []
    

    for num,(img,label) in enumerate(test_dl):
      outputs = model(img)
      batchs_report = []
      _, pred = torch.max(outputs, dim=1)
      predx,labelx = (pred.cpu().clone().tolist(),label.cpu().clone().tolist())
      
      list_label+=labelx
      list_pred+=predx
    f1 =f1_score(list_label,list_pred,average='macro')
    acc =accuracy_score(list_label,list_pred)

    return f'f1 Score: {f1}', f'Accuracy: {acc}'
  score_rate = test_accuracy_f1(test_dl)
  print(score_rate)
  print("###############################################################")
  print("#############Testing is End #############")
  print("###############################################################")
except Exception as e:
   print("Something Went Wrong ")
   import traceback
   print(traceback.format_exc())
   print(e)








