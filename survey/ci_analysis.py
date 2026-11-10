
import csv
import matplotlib.pyplot as plt
import math
import numpy as np
import seaborn as sns
import pandas as pd

from scipy.stats import t
from pathlib import Path

from utils import build_votes_from_ratings



RANDOM_SEED = 42
RNG = np.random.default_rng(RANDOM_SEED)
FONT_SCALE= 1.6
TITLE_SIZE = 16

BS_PART_CHECKPOINT_PATH = Path(
    "./saves/participant_bootstrap_iqa_checkpoint.pkl"
)

BS_STIM_CHECKPOINT_PATH = Path(
    "./saves/stimuli_bootstrap_iqa_checkpoint.pkl"
)

BS_CRS_CHECKPOINT_PATH = Path(
    "./saves/crossed_bootstrap_iqa_checkpoint.pkl"
)

CHECKPOINT_EVERY = 25
N_BOOTSTRAPS = 5
BEST_METRIC_HYPOTHESIS = 'topiq_fr'

SIGNIFICANCE_LEVEL = 0.05

# 1 if metric quantifies visual degredation (the higher the more perturbed) -1 if metric quantifies visual fidelity (inverted)
METRIC_DIRECTION = {
    "l2":        1,
    "mad":       1,
    "psnr":     -1,
    "ssim":     -1,
    "ms_ssim":  -1,
    "vif":      -1,
    "dists":     1,
    "lpips":     1,
    "ahiq":     -1,
    "topiq_fr": -1,
}


class CIAnalysis:

    def __init__(self, ratings, l2_distortions, file_dict, alpha=SIGNIFICANCE_LEVEL):
        self.ratings_dict = build_votes_from_ratings(ratings)
        self.c_level = 1.0 - alpha
        self.ci_dict = self.process_ratings()
        self.l2_distortions = l2_distortions
        self.file_dict = file_dict
    
    def get_ci_tuple(self, ratings):
        ci_entry = ConfidenceInterval(ratings, c_level=self.c_level)
        lower, upper = ci_entry.ci()
        return ci_entry.sample_mu, lower, upper
    
    def process_ratings(self):
        print(f'====== performing participant bootstrapping computing confidence on the original MOS intervals for {self.c_level} confidence...====')
        ci_dict = self.ratings_dict.copy()
        for attack_name, v_dict in self.ratings_dict.items():
            for delta, ratings in v_dict.items():
                ci_dict[attack_name][delta] = self.get_ci_tuple(ratings)
        return ci_dict
    
    def transform(self):
        file_id, file, id, means, ci_lower, ci_upper = [], [], [], [], [], []
        for attack_name, v_dict in self.ci_dict.items():
            for delta, intervals in v_dict.items():
                #id.append(f'{attack_name} : {round(self.l2_distortions[attack_name][delta], 2)}')
                id.append(round(self.l2_distortions[attack_name][delta], 2))
                file.append(f'{self.file_dict[attack_name][delta]}.png')
                mean, lower, upper = intervals
                means.append(mean)
                ci_lower.append(lower)
                ci_upper.append(upper)
                file_id.append(f"{attack_name}-{delta}")
        self.ci_df = pd.DataFrame([file_id, means, ci_lower, ci_upper], index=["file_id", "mos", "ci_lower", "ci_upper"]).T
        print('done')

                
        zipped = sorted(zip(id, file, means, ci_lower, ci_upper), key=lambda x: x[0])
        id, file, means, ci_lower, ci_upper = zip(*zipped)
        return [f'{fi} : {str(i)}' for fi,i in zip(file,id)], means, ci_lower, ci_upper

    def plot(self, visualize=False):

        sns.set_theme(style="white", context="paper", font_scale=FONT_SCALE)
        id, means, ci_lower, ci_upper = self.transform()
        plt.figure(figsize=(14,6))
        plt.bar(id, means, yerr=[np.array(means)-np.array(ci_lower), np.array(ci_upper)-np.array(means)], 
                capsize=4, color='skyblue')
        plt.xlabel("Image Index : L2-norm")
        plt.xticks(rotation=90)
        plt.ylabel("Mean Opinion Score (DSIS)")
        plt.title("Confidence Intervals of the Mean Opinion of the Perceptual Evaluation of Adversarial Samples (95% CI)", fontweight='bold', fontsize=TITLE_SIZE)
        plt.tight_layout()
        if visualize:
            plt.show()
        else:
            plt.savefig('./visualizations/participant_confidence_intervals.png')

    def extend_results_with_intervals(self, dp_obj):
        stimuli_df = dp_obj.stimuli_df


    
class ConfidenceInterval:

    def __init__(self, ratings, c_level=0.95):
        self.ratings = ratings
        self.n_ratings = len(ratings)
        self.dof = self.n_ratings - 1 # degrees of freedom
        self.t_alpha = (1 - c_level) # alpha value determined by confidence level
        self.t_val = t.ppf(1 - (1 - c_level) / 2, self.dof)
        self.sample_mu = sum(ratings) / self.n_ratings
        self.sample_std = self.get_std()

    def ci(self):
        lower = self.sample_mu - (self.t_val * (self.sample_std/math.sqrt(self.n_ratings)))
        upper = self.sample_mu + (self.t_val * (self.sample_std/math.sqrt(self.n_ratings)))
        return lower, upper

    def bootstrap_ci(self, data, n_bootstrap=10000, c_level=0.95, statistic=np.mean):
        """Compute a bootstrap confidence interval for Likert-scale or any nonparametric data."""
        data = np.array(data)
        n = len(data)
        boot_means = np.empty(n_bootstrap)

        for i in range(n_bootstrap):
            sample = np.random.choice(data, size=n, replace=True)
            boot_means[i] = statistic(sample)

        lower = np.percentile(boot_means, ((1 - c_level) / 2) * 100)
        upper = np.percentile(boot_means, (1 - (1 - c_level) / 2) * 100)
        return lower, upper
    
    def get_std(self):
        return math.sqrt(sum([(rating - self.sample_mu)**2 for rating in self.ratings])/ self.dof)