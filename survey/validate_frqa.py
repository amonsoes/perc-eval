import os
import pyiqa
import torch
import pandas as pd
import torchvision
import torchvision.transforms as T


from pathlib import Path
from tqdm import tqdm

class FRQA:

    def __init__(self, device):

        # rule based
        self.mad = pyiqa.create_metric('mad', device=device)
        self.psnr = pyiqa.create_metric('psnr', device=device)
        self.ssim = pyiqa.create_metric('ssim', device=device)
        self.ms_ssim = pyiqa.create_metric('ms_ssim', device=device)
        self.vif = pyiqa.create_metric('vif', device=device)

        # model based
        self.dists = pyiqa.create_metric('dists', device=device)
        self.lpips = pyiqa.create_metric('lpips', device=device)
        self.ahiq = pyiqa.create_metric('ahiq', device=device)
        self.topiq = pyiqa.create_metric('topiq_fr', device=device)

        self.device = device
        self.columns = ['file_id', 'l2', 'mad', 'psnr', 'ssim', 'ms_ssim', 'vif', 'dists', 'lpips', 'ahiq', 'topiq_fr']

        self.score_dict = {
            "file_id": [],
            "l2" : [],
            "mad" : [],
            "psnr" : [],
            "ssim" : [],
            "ms_ssim" : [],
            "vif" : [],
            "dists" : [],
            "lpips" : [],
            "ahiq" : [],
            "topiq_fr" : [],
        }
    
    def compute_iqa(self, target, reference, designator):
        """
        target: adversarial/distorted images
        reference: clean reference images
        paths: str to 
        """

        assert reference.ndim == 4
        assert target.ndim == 4

        self.score_dict[self.columns[0]].append(designator) 

        l2_score = self.get_l2(reference, target)

        # pyiqa expects distorted/target first, reference second.
        mad_score = self.mad(target, reference)
        psnr_score = self.psnr(target, reference)
        ssim_score = self.ssim(target, reference)
        ms_ssim_score = self.ms_ssim(target, reference)
        vif_score = self.vif(target, reference)

        dists_score = self.dists(target, reference)
        lpips_score = self.lpips(target, reference)
        ahiq_score = self.ahiq(target, reference)
        topiq_score = self.topiq(target, reference)

        iqa_list = [
            l2_score,
            mad_score,
            psnr_score,
            ssim_score,
            ms_ssim_score,
            vif_score,
            dists_score,
            lpips_score,
            ahiq_score,
            topiq_score
        ]

        for iqa_score, col in zip(iqa_list, self.columns[1:]):
            self.score_dict[col].append(iqa_score.item())

    def get_l2(self, reference, target):
        if reference.max() > 1:
            raise ValueError('original image is not 0 < x < 1')
        if target.max() > 1:
            raise ValueError('adv image is not 0 < x < 1')
        distance = (reference - target).pow(2).sum(dim=(1,2,3)).sqrt()
        return distance / reference.max()

    def get_dataframe(self):
        
        score_df = pd.DataFrame(self.score_dict, columns=self.columns)
        score_df.to_csv("./iqa_scores.csv")

        print("IQA scores for survey samples:\n")
        print(score_df)
        return score_df

    def load_dataframe(self):
        return pd.read_csv("./iqa_scores.csv", index_col=0)

if __name__ == '__main__':

    iqa_root = Path('./data/perc_eval')
    convert_dtype = T.ConvertImageDtype(torch.float32)
    frqa = FRQA(device='cpu')

    dist_dict = {
        'percal':{'4.0':12,
                    '8.0':18,
                    '15.0':10,
                    '23.0':3,
                    '38.0':5},
        'square_attack':{'4.0':20,
                    '8.0':27,
                    '15.0':28,
                    '23.0':32,
                    '38.0':36},
        'ppba':{'4.0':40,
                    '8.0':44,
                    '15.0':49,
                    '23.0':53,
                    '38.0':56},
        'rcw':{'4.0':77,
                    '8.0':66,
                    '15.0':71,
                    '23.0':74,
                    '38.0':63},
        'cgnc':{'4.0':83,
                    '8.0':85,
                    '15.0':88,
                    '23.0':93,
                    '38.0':97},
        'cgsp':{'4.0':102,
                    '8.0':107,
                    '15.0':109,
                    '23.0':112,
                    '38.0':117},
        'sparsesigmaattack':{'4.0':120,
                    '8.0':127,
                    '15.0':131,
                    '23.0':134,
                    '38.0':139},
        'hpf_fgsm':{'4.0':142,
                    '8.0':145,
                    '15.0':150,
                    '23.0':153,
                    '38.0':157,},
        'dim':{'4.0':161,
                    '8.0':165,
                    '15.0':168,
                    '23.0':175,
                    '38.0':179},
        'bsr':{'4.0': 180,
                    '8.0':184,
                    '15.0':188,
                    '23.0':192,
                    '38.0':196},
        'vmifgsm':{'4.0':202,
                    '8.0':205,
                    '15.0':209,
                    '23.0':214,
                    '38.0':219},
        'pgdl2':{'4.0':220,
                    '8.0':224,
                    '15.0':228,
                    '23.0':235,
                    '38.0':238},
        'sgd_uap':{'4.0':241,
                    '8.0':245,
                    '15.0':248,
                    '23.0':253,
                    '38.0':258},
        'sga_uap':{'4.0':263,
                    '8.0':265,
                    '15.0':271,
                    '23.0':275,
                    '38.0':279},
        }

    file_2_info = {}
    for attack, pert_dict in dist_dict.items():
        for pert, file in pert_dict.items():
            file_2_info[file] = (attack, pert)
        
    if not os.path.exists("./iqa_scores.csv"):
    
        file_2_inf_keys = file_2_info.keys()
        for file in tqdm(range(280)):
            if file in file_2_inf_keys:
                attack, pert_deg = file_2_info[file]
                file_path_adv = iqa_root / pert_deg / attack / f"{file}.png"
                file_path_ref = iqa_root / "images_new" / f"{file}.png"

                target = convert_dtype(torchvision.io.read_image(file_path_adv)).unsqueeze(0)
                reference = convert_dtype(torchvision.io.read_image(file_path_ref)).unsqueeze(0)

                frqa.compute_iqa(target, reference, f"{attack}-{pert_deg}")

        frqa.get_dataframe()

    else:
        print(f"IQA score file already exsists at ./iqa_scores.csv :\n")
        iqa_score_df = pd.read_csv("./iqa_scores.csv", index_col=0)
        print(iqa_score_df)