"""
Step 0: Set up Experiment Object that keeps track of average l2, cos, pearsoncc
Step 1: Load dataset
Step 2: For a specific file hash in the real images, get all the corresponding adversarial samples for all attack classes from the 15.0 distortion category
Step 3: If not all are present as adversarial samples, pass this iteration
Step 4: Compute delta by subtracting the original image from the adversarial sample
Step 5: Normalize all distortions to unit length
Step 6: for all attacks used by Fezza19, compute the l2, cos, pearson to all the other attacks, add values to average l2, cos, pearsoncc
Step 7: Repeat for all of our attacks, make sure you store the avgl2, avg cos, pearsoncc in another object obviously

! You'll need to (1) normalize the cosine similarity and (2) subtract it from 1 (so 1- cos(x,y)), since
0 is complete negative correlation and  1 is complete correlation (0.5 is orthogonality)
"""

import torch
import torch.nn as nn
import torchvision.transforms as T
import torch.nn.functional as F
import os
import pandas as pd

from tqdm import tqdm
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import InterpolationMode


from src.datasets.subsets import Nips17Subset



class Nips17Subset(Dataset):
    
    def __init__(self):
        super().__init__()
        path_test = './data/nips17/'
        path_labels = path_test + 'images.csv'
        path_images = path_test + 'images/'
        self.labels = pd.read_csv(path_labels)
        self.img_dir = path_images
        self.transform = T.Compose([T.Resize(256,  interpolation=InterpolationMode.BILINEAR),
                                    T.CenterCrop(224),
                                    T.ToTensor()])

    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.labels.iloc[idx, 0])
        image = Image.open(img_path + '.png')
        image = self.transform(image)
        label = self.labels.iloc[idx, 6] - 1
        return image, label, img_path.split('/')[-1]

class Stats:

    def __init__(self, data, attacks):
        """
        pass attacks as list with the names corresponding to the naming in the dataset like this ['pgdl2',...,'vmifgsm']
        """
        len_attacks = len(attacks)
        self.data = data
        self.adv_data_path = './data/perc_eval/4.0'
        self.total_l2 = torch.zeros((len_attacks, len_attacks),dtype=torch.float32)
        self.total_cos = torch.zeros((len_attacks, len_attacks),dtype=torch.float32)
        self.total_pearson = torch.zeros((len_attacks, len_attacks),dtype=torch.float32)
        self.cos = nn.CosineSimilarity(dim=(1,2,3), eps=1e-6)
        self.attacks = attacks
        self.attacks_to_id = {attack:e for e,attack in enumerate(attacks)}
        self.n = 0
        self.to_tensor = T.ToTensor()
    
    def process_data(self):
        for image, label, filename in tqdm(self.data):
            self.n += 1
            for attack in self.attacks:
                id = self.attacks_to_id[attack]
                adv_image1 = self.load_adv_sample(attack, filename)
                pert1 = self.to_unit_length(image - adv_image1)
                for sub_attack in self.attacks:
                    if sub_attack != attack:
                        # don't compute for identical attacks
                        sub_id = self.attacks_to_id[sub_attack]
                        adv_image2 = self.load_adv_sample(sub_attack, filename)
                        pert2 = self.to_unit_length(image - adv_image2)

                        # compute l2, cos, pearson
                        cos_result = self.norm_cosine_sim(pert1, pert2)
                        l2_result = self.l2(pert1, pert2)
                        pearson_result = self.norm_pearson_corr(pert1, pert2)

                        # fill matrices with result
                        self.total_cos[id, sub_id] += cos_result
                        self.total_l2[id, sub_id] += l2_result
                        self.total_pearson[id, sub_id] += pearson_result
    
    def get_results(self):
        avg_l2 = self.total_l2 / self.n
        avg_cos = self.total_cos / self.n
        avg_pearson = self.total_pearson / self.n
        
        print('Average l2 distances:')
        print(avg_l2)
        print('Average Inverted Cosine Similarities:')
        print(avg_cos)
        print('Average Inverted Pearson Correlation Coefficient:')
        print(avg_pearson)

        return avg_l2, avg_cos, avg_pearson

    def load_adv_sample(self, attack, filename):
        img_path = f'{self.adv_data_path}/{attack}/{filename}.png'
        image = Image.open(img_path)
        image = self.to_tensor(image)
        return image
    
    def to_unit_length(self, x):
        return x / x.norm(p=2)

    def norm_cosine_sim(self, x, y):
        """
        deltas are centered around 0.
        To get cos values between 0,1, we normalize.

        1 -> same directionality
        0.5 -> orthoginality
        0 -> negative directionality

        invert for this computation
        """
        x = x.flatten()
        y = y.flatten()
        return 1 - ((1 + F.cosine_similarity(x, y, dim=0) ) / 2).item()
    
    def norm_pearson_corr(self, x, y):
        """
        Compute the Pearson correlation coefficient between two images.
        
        Args:
            img1, img2: torch.Tensor of shape (C, H, W) or (H, W).
            
        Returns:
            Pearson correlation coefficient (scalar tensor).

        1 -> perfect correlation
        0.5 -> not correlated
        0 -> perfect negative correlation

        invert for this computation
        """
        x = x.flatten()
        y = y.flatten()
        
        mean1, mean2 = x.mean(), y.mean()
        std1, std2 = x.std(), y.std()
        covariance = ((x - mean1) * (y - mean2)).mean()      
        pearson_corr = covariance / (std1 * std2 + 1e-8)  # Add small epsilon to avoid division by zero
        
        return  1 - ((1 + pearson_corr.item()) / 2)

    def l2(self, x, y):
        #distance = (x - y).pow(2).sum(dim=(1,2,3)).sqrt()
        distance = (x - y).pow(2).sum().sqrt()
        return distance.item()

if __name__ == '__main__':

    data = Nips17Subset()
    
    attacks=[
        'dim',
        'hpf_fgsm'
    ]

    stats = Stats(data, attacks)
    stats.process_data()
    stats.get_results()
    print('done')


    