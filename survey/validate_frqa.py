
import pyiqa
import pandas as pd

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
        return score_df

    def load_dataframe(self):
        return pd.read_csv("./iqa_scores.csv", index_col=0)
