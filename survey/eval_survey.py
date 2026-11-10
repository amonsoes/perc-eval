import argparse
import csv
import torch
import matplotlib.pyplot as plt
import os
import math
import torchvision
import torchvision.transforms as T
import numpy as np
import pandas as pd
import copy
import seaborn as sns

from scipy.stats import pearsonr, spearmanr
from statsmodels.stats.multitest import multipletests
from tqdm import tqdm

from process import DatasetConstructor
from PIL import Image
from sklearn.model_selection import GroupKFold
from pathlib import Path

from validate_frqa import FRQA
from metric_collection import MetricCollection
from utils import FivePLFitError, build_votes_from_ratings
from ci_analysis import CIAnalysis

from bootstrap_checkpoint import save_bootstrap_checkpoint, load_bootstrap_checkpoint
from advanced_survey_stats import repeated_split_half_reliability, summarize_split_half, participant_leave_one_out_consistency, summarize_participant_consistency



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

CHECKPOINT_EVERY = 20
N_BOOTSTRAPS = 5000
BEST_METRIC_HYPOTHESIS = 'topiq_fr' # post-hoc-hypothesis based on point estimates

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

##########################################################################################


# DATACLASS SETUP


##########################################################################################


class DataProcessor:

    def __init__(self, device):
        self.n = 0
        self.iqa_root = Path('./data/perc_eval')
        self.attacks = ['pgdl2', 'vmifgsm', 'percal', 'sgd_uap', 'sparsesigmaattack', 'hpf_fgsm', 'square_attack', 'ppba', 'rcw', 'cgnc', 'cgsp', 'dim', 'bsr', 'sga_uap']
        self.votes = self.set_up_votes()
        self.to_tensor = T.ToTensor()
        self.dataset_contructor = DatasetConstructor()
        self.frqa = FRQA(device=device)
        self.convert_dtype = T.ConvertImageDtype(torch.float32)
        self.answer_to_value = {
            '1': 0, # -> no perceptible distortion
            '2': 0.25,
            '3': 0.5,
            '5': 0.75,
            '6': 1 # -> very perturbed
        }

        self.stat_2_int = {
            'plcc':0,
            'srocc':1,
            'or':3
        }

        self.display_names = {
                    "l2": r"$L_2$",
                    "mad": "MAD",
                    "ssim": "SSIM",
                    "psnr": "PSNR",
                    "ssim.1": "MS-SSIM",
                    "vif": "VIF",
                    "dists": "DISTS",
                    "lpips": "LPIPS",
                    "ahiq": "AHIQ",
                    "topiq_fr": "TOPIQ-FR",
                }
        
        self.dist_dict = {
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

        self.file_2_info = {}
        for attack, pert_dict in self.dist_dict.items():
            for pert, file in pert_dict.items():
                self.file_2_info[file] = (attack, pert)

        self.l2_dict = copy.deepcopy(self.dist_dict)
        self.f_dict = copy.deepcopy(self.dist_dict)


        self.field_map = {
            'A031':'A045_01',
            'A084':'A046_01',
            'A033':'A049_01',
            'A034':'A050_01',
            'A035':'A053_01',
            'A036':'A054_01',
            'A037':'A051_01',
            'A038':'A052_01',
            'A039':'A048_01',
            'A040':'A055_01',
            'A041':'A047_01',
            'A042':'A059_01',
            'A043':'A056_01',
            'A044':'A057_01',
            'A115':'A129_01',
            'A116':'A130_01',
            'A117':'A133_01',
            'A118':'A134_01',
            'A119':'A141_01',
            'A120':'A142_01',
            'A121':'A135_01',
            'A122':'A136_01',
            'A123':'A132_01',
            'A124':'A140_01',
            'A125':'A131_01',
            'A126':'A137_01',
            'A127':'A138_01',
            'A128':'A139_01',
        }
        

        self.file_2_inf = {}
        self.file_2_nl2 = {}
        self.file_2_mos = {}

        #frequency dicts
        self.file_2_mfreq = {}

        #color and luminance dicts
        self.file_2_mlum = {}
        self.file_2_mcb = {}
        self.file_2_mcr = {}

        #ig stat dicts

        self.create_file_csv()
        self.eps = 1e-05
        self.maximum_l2 = math.log(38.0 + self.eps)
        self.minimum_l2 = math.log(0 + self.eps)


    ##########################################################################################


    # TOOLS, FUNCTIONS AND METRICS


    ##########################################################################################

    def convert_attack_l2_to_file(self, attack_l2):
        attack, dist = attack_l2.split("-")
        return self.dist_dict[attack][dist]

    def log_minmax_l2(self, l2):
        l2 = math.log(l2 + self.eps)
        return (l2 - self.minimum_l2) / (self.maximum_l2 - self.minimum_l2)

    def get_mos(self, file):
        attack, pert_deg = self.file_2_inf[file.split('.')[0]]
        path = self.iqa_root / pert_deg / attack / "iqa_report.csv"
        with open(path, 'r') as f:
            csv_obj = csv.reader(f)
            for line in csv_obj:
                if line[0] == f'./data/survey/images/{str(file)}':
                    return float(line[-1])

    def get_actual_distortion(self, file):
        attack, pert_deg = self.file_2_inf[file.split('.')[0]]
        path = self.iqa_root / pert_deg / attack / "iqa_report.csv"
        with open(path, 'r') as f:
            csv_obj = csv.reader(f)
            for line in csv_obj:
                if line[0] == f'./data/survey/images/{str(file)}':
                    return line[1]

    def remove_outliers(self):
        self.participants_raw = self.participants.copy()

        self.participants_flagged = self.identify_outlier_ratings(
            self.participants_raw,
            ol_threshold=3.0,
        )

        self.participants_clean = (
            self.participants_flagged.loc[
                ~self.participants_flagged["is_outlier"]
            ]
            .copy()
            .reset_index(drop=True)
        )

    def identify_outlier_ratings(
        self,
        ratings,
        image_col="file_id",
        rating_col="rating",
        ol_threshold=3.0,
    ):
        ratings = ratings.copy().reset_index(drop=True)

        ratings["z_score"] = np.nan

        for _, indices in ratings.groupby(
            image_col,
            sort=False,
        ).groups.items():

            values = ratings.loc[
                indices,
                rating_col,
            ].to_numpy(dtype=float)

            z_scores = np.asarray(
                self.compute_z(values),
                dtype=float,
            )

            if len(z_scores) != len(values):
                raise ValueError(
                    "compute_z returned an unexpected number of values."
                )

            ratings.loc[indices, "z_score"] = z_scores

        ratings["is_outlier"] = (
            ratings["z_score"] > ol_threshold
        )

        return ratings
    
    def compute_z(self, votes):
        n = len(votes)
        mu = sum(votes) / n
        sigma = math.sqrt(sum([(vote - mu)**2 for vote in votes])/ (n-1))
        z_scores = [abs(((vote - mu) / (sigma + self.eps))) for vote in votes]
        return z_scores
    
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
        pearson_corr = covariance / (std1 * std2 + 1e-8)
        return pearson_corr


    ##########################################################################################


    # PARTICIPANT CONSISTENCY, SPLIT-HALF SUMMARY


    ##########################################################################################

    def survey_statistics(self):
        print("\n================= survey statistics ===================\n")

        print('(1) number of rated images per participant, (2) number of participants:')
        print(
            self.participants
            .groupby("participant_id")
            .size()
            .value_counts()
        )


        total = 0
        n = 0
        n_votes = []
        filepaths = []
        attacks = []
        mags = []
        medians = []
        means = []
        stds = []

        votes = build_votes_from_ratings(self.participants_clean)

        for attack, perts in votes.items():
            for pert, votes in perts.items():
                votes = np.array(votes)
                filepath = self.dist_dict[attack][pert]
                filepaths.append(filepath)
                attacks.append(attack)
                mags.append(pert)
                print(f'n ratings for image with index: {filepath}, magnitude: {pert} by attack: {attack} -> {len(votes)}')
                total += len(votes)
                medians.append(np.median(votes))
                means.append(votes.mean())
                stds.append(votes.std())
                n_votes.append(len(votes))
                n += 1

        n_votes = np.array(n_votes)
        filepaths = np.array(filepaths)
        attacks = np.array(attacks)
        mags = np.array(mags)

        survey_stat_df = pd.DataFrame(zip(filepaths, attacks, mags, n_votes, medians, means, stds), columns=['filepaths', 'attacks', 'magnitudes', 'n_votes', 'medians', 'means', 'STD'])
        survey_stat_df = survey_stat_df.sort_values(by='filepaths', ascending=True)

        print(f'\n(4) average rating per image: {n_votes.mean()}')
        print(f'\n(5) rating number standard deviation: {n_votes.std()}')
        print(f'\n(6) total number of ratings: {survey_stat_df["n_votes"].sum()}')

        print("Summarized survey statistics per Image:\n")

        print(survey_stat_df)
        survey_stat_df.to_csv("./survey_stats/survey_stat_summary_image.csv")
        survey_stat_df.to_latex("./survey_stats/survey_stat_summary_image.tex")


        split_half_results = (
            repeated_split_half_reliability(
                self.participants_clean,
                participant_col="participant_id",
                image_col="file_id",
                rating_col="rating",
                n_splits=5000,
                min_ratings_per_image_per_half=2,
                random_seed=RANDOM_SEED,
            )
        )

        summarized_split_half_results = summarize_split_half(split_half_results)
        summarized_split_half_results.to_csv("./survey_stats/summarized_split_half_results.csv")
        summarized_split_half_results.to_latex("./survey_stats/summarized_split_half_results.tex")

        print("Summary of Repeated Split-Half Reliability computation:\n")
        print(summarized_split_half_results)

        participant_consistency = (
            participant_leave_one_out_consistency(
                self.participants,
                participant_col="participant_id",
                image_col="file_id",
                rating_col="rating",
                minimum_images=17,
            )
        )

        participant_consistency_summary = (
            summarize_participant_consistency(
                participant_consistency
            )
        )
        participant_consistency_summary.to_csv("./survey_stats/participant_consistency_summary.csv")
        participant_consistency_summary.to_latex("./survey_stats/participant_consistency_summary.tex")

        print("Summary of Leave-One-Out Participant Consistency:\n")
        print(participant_consistency_summary)
                    

        print("\n====================================\n")


    ##########################################################################################


    # VOTE ANALYSIS AND MOS COMPUTATION


    ##########################################################################################
    
    
    def create_file_csv(self):
        #if not os.path.exists('./data/perc_eval/images/images.csv'):
        with open ('./data/perc_eval/images/images.csv', 'w') as f:
            csv_obj = csv.writer(f)
            csv_obj.writerow(['filename', 'attack', 'pert_bound'])
            for key, val in self.dist_dict.items():
                for skey, sval in val.items():
                    csv_obj.writerow([sval, key, skey])
                    self.file_2_inf[str(sval)] = (key, skey)

    def set_up_votes(self):
        votes = {}
        for attack in self.attacks:
            votes[attack] = {}
            for pert in ['4.0', '8.0', '15.0', '23.0', '38.0']:
                votes[attack][pert] = []
        return votes
    
    def set_up_results(self):
        votes = {}
        for attack in self.attacks:
            votes[attack] = {}
            for pert in ['4.0', '8.0', '15.0', '23.0', '38.0']:
                votes[attack][pert] = 0
        return votes

    def process_data(self, data_path):
        if not os.path.exists("./participants.csv"):
            self.participants = []
            with open(data_path, 'r') as f:
                csv_obj = csv.reader(f)
                fields = next(csv_obj)
                results_fields = fields[35:49] + fields[90:104]
                pertmag_fields = fields[49:63] + fields[104:118]
                for line in csv_obj:
                    is_desktop = True if int(line[63]) == 1 else False
                    if is_desktop:
                        self.n += 1
                        participant_id = line[0]
                        results = line[35:49] + line[90:104]
                        results_dict = {field: result for field, result in zip(results_fields, results)}
                        if all(results):
                            pert_mags_files = line[49:63] + line[104:118]
                            pertmag_dict = {field: pertmag for field, pertmag in zip(pertmag_fields, pert_mags_files)}
                            for e, (res_field, result)  in enumerate(results_dict.items()):
                                pertmag_field = self.field_map[res_field]
                                pert_mag_lst = pertmag_dict[pertmag_field].split(',')
                                if all([x != 'undefined' for x in pert_mag_lst]):
                                    if len(pert_mag_lst) > 3:
                                        pert_mag, _ , file, _, attack = pert_mag_lst
                                    else:
                                        pert_mag, file, attack = pert_mag_lst
                                    self.votes[attack][pert_mag.split('.')[0] + '.0'].append(self.answer_to_value[result])
                                    self.participants.append((participant_id, '-'.join([attack, pert_mag.split('.')[0] + '.0']), self.answer_to_value[result]))
            self.participants = pd.DataFrame(self.participants, columns=('participant_id', 'file_id', 'rating'))
            self.participants.to_csv("./participants.csv")
        else:
            self.participants = pd.read_csv("./participants.csv")

    def ratings_to_long(self, data):
        data = data.copy()

        # Convert NumPy arrays or similar vectors to Python lists
        for column in ['results', 'perts_mags_files']:
            data[column] = data[column].map(
                lambda value: (
                    value.tolist()
                    if hasattr(value, "tolist")
                    else list(value)
                )
            )

        rating_lengths = data['results'].map(len)
        image_lengths = data['perts_mags_files'].map(len)

        if not rating_lengths.equals(image_lengths):
            bad_rows = data.index[
                rating_lengths != image_lengths
            ].tolist()

            raise ValueError(
                "Ratings and images have different lengths "
                f"in rows: {bad_rows}"
            )

        if not (rating_lengths == 28).all():
            raise ValueError(
                "Not every participant has exactly 28 ratings. "
                f"Observed lengths: {sorted(rating_lengths.unique())}"
            )

        # Explode both columns simultaneously
        long_data = (
            data
            .explode(
                ['results', 'perts_mags_files'],
                ignore_index=True,
            )
            .rename(
                columns={
                    'results': "result",
                    'perts_mags_files': "pert_mag_file",
                }
            )
        )

        long_data["result"] = pd.to_numeric(
            long_data["result"],
            errors="raise",
        )

        if long_data[
            ["participant_id", "pert_mag_file", "result"]
        ].isna().any().any():
            raise ValueError(
                "Missing participant IDs, image IDs, or ratings."
            )

        return long_data[
            ["participant_id", "pert_mag_file", "result"]
        ]

    def l2_to_mos(self, l2):
        if l2 < 4.0:
            return 0.0
        elif l2 < 8.0:
            return 0.25
        elif l2 < 15.0:
            return 0.5
        elif l2 < 23.0:
            return 0.75
        else:
            return 1

    def mos(self, votes):
        if votes:
            mos = sum(votes) / len(votes)
            return mos
        else:
            return 0

    def get_iqa_and_mos(self):

        print('Compute IQA for stimuli...')

        if not os.path.exists("./iqa_scores.csv"):
            
            file_2_inf_keys = self.file_2_inf.keys()
            for file in tqdm(range(280)):
                if str(file) in file_2_inf_keys:
                    attack, pert_deg = self.file_2_inf[str(file)]
                    file_path_adv = self.iqa_root / pert_deg / attack / f"{file}.png"
                    file_path_ref = self.iqa_root / "images_new" / f"{file}.png"

                    target = self.convert_dtype(torchvision.io.read_image(file_path_adv)).unsqueeze(0)
                    reference = self.convert_dtype(torchvision.io.read_image(file_path_ref)).unsqueeze(0)

                    self.frqa.compute_iqa(target, reference, f"{attack}-{pert_deg}")

            self.results_df = self.frqa.get_dataframe()
                
        else:
            print("found IQA score csv file in root. Passing computation....")

            self.results_df = self.frqa.load_dataframe()
        
        print('IQA retrieved.')

        print('Compute MOS...')

        self.remove_outliers()

        self.original_subjective_statistics = self.original_subjective_statistics(
            self.participants_clean
        )

        self.results_df.sort_values(by='file_id', inplace=True)

        # read IQA-MOS tabel computed before
        self.stimuli_df = pd.merge(self.results_df, self.original_subjective_statistics, on='file_id')
        self.metric_collection = MetricCollection(image_stats=self.original_subjective_statistics)
        self.files_df = self.results_df['file_id']

        print('done')

    
    ##########################################################################################


    # SUBJECTIVE RESAMPLING, STIMULI RESAMPLING, TWO-WAY CROSSED RESAMPLING PROCEDURES


    ##########################################################################################

    def verify_mos_consistency(
        self,
        ratings, #individual ratings
        stimuli, # iqa results
        file_col="file_id",
        rating_col="rating",
        mos_col="mos",
        atol=1e-10,
    ):
        reconstructed = (
            ratings.groupby(file_col)[rating_col]
            .mean()
            .rename("reconstructed_MOS")
        )

        comparison = (
            stimuli[[file_col, mos_col]]
            .set_index(file_col)
            .join(reconstructed)
        )

        comparison["difference"] = (
            comparison[mos_col]
            - comparison["reconstructed_MOS"]
        )

        inconsistent = comparison.loc[
            ~np.isclose(
                comparison[mos_col],
                comparison["reconstructed_MOS"],
                atol=atol,
            )
        ]

        if not inconsistent.empty:
            raise ValueError(
                "Participant ratings do not reproduce the original MOS:\n"
                + inconsistent.to_string()
            )

    def check_stats(self, point_estimates, bootstrap_results, checks):

        for metric, statistic in checks:
            values = np.asarray(
                bootstrap_results[metric][statistic],
                dtype=float,
            )

            point = point_estimates[metric][self.stat_2_int[statistic]]
            lower, upper = np.percentile(values, [2.5, 97.5])

            print(
                metric,
                statistic,
                {
                    "point": point,
                    "bootstrap_mean": values.mean(),
                    "bootstrap_median": np.median(values),
                    "bias": values.mean() - point,
                    "ci": (lower, upper),
                    "point_percentile": np.mean(values <= point),
                    "minimum": values.min(),
                    "maximum": values.max(),
                },
            )

    def original_subjective_statistics(
        self,
        ratings,
        image_col="file_id",
        rating_col="rating",
    ):
        return (
            ratings
            .groupby(image_col)[rating_col]
            .agg(
                mos="mean",
                rating_std="std",
                n_ratings="count",
            )
            .reset_index()
        )

    def resample_subjective_statistics(
        self,
        ratings,
        rng,
        participant_col="participant_id",
        image_col="image_id",
        rating_col="rating",
    ):
        participants = ratings[participant_col].unique()

        participant_draw = rng.choice(
            participants,
            size=len(participants),
            replace=True,
        )

        # Number of times each participant was selected
        weights = (
            pd.Series(participant_draw)
            .value_counts()
            .rename("bootstrap_weight")
        )

        resampled_mos = self.compute_weighted_mos(ratings, weights, participant_col, rating_col, image_col)

        return resampled_mos

    def resample_stimuli_statistics(
        self,
        stimuli,
        rng,
        mos_col="MOS",
        image_col="file",
    ):
        if stimuli[image_col].isna().any():
            raise ValueError(f"{image_col} contains missing values.")

        if stimuli[image_col].duplicated().any():
            raise ValueError(
                "Stimulus dataframe must contain exactly one row per stimulus."
            )

        if stimuli[mos_col].isna().any():
            raise ValueError(f"{mos_col} contains missing values.")

        n_stimuli = len(stimuli)

        # Sample row positions with replacement.
        draw_indices = rng.integers(
            low=0,
            high=n_stimuli,
            size=n_stimuli,
        )

        # This retains the original column dtypes.
        stimuli_draw_df = (
            stimuli.iloc[draw_indices]
            .copy()
            .reset_index(drop=True)
        )

        weights = (
            stimuli_draw_df[image_col]
            .value_counts(sort=False)
            .rename("bootstrap_weight")
        )

        return stimuli_draw_df, weights

    
    def compute_weighted_mos(self, ratings, weights, participant_col, rating_col, image_col):

        sampled = ratings.merge(
            weights,
            left_on=participant_col,
            right_index=True,
            how="inner",
        )

        sampled["weighted_rating"] = (
            sampled[rating_col]
            * sampled["bootstrap_weight"]
        )

        sampled["weighted_squared_rating"] = (
            sampled[rating_col] ** 2
            * sampled["bootstrap_weight"]
        )

        statistics = (
            sampled
            .groupby(image_col)
            .agg(
                n_ratings=("bootstrap_weight", "sum"),
                weighted_sum=("weighted_rating", "sum"),
                weighted_squared_sum=(
                    "weighted_squared_rating",
                    "sum",
                ),
            )
            .reset_index()
        )

        statistics["mos"] = (
            statistics["weighted_sum"]
            / statistics["n_ratings"]
        )

        # Sample variance of the resampled ratings
        variance_numerator = (
            statistics["weighted_squared_sum"]
            - statistics["weighted_sum"] ** 2
            / statistics["n_ratings"]
        )

        statistics["rating_variance"] = (
            variance_numerator
            / (statistics["n_ratings"] - 1)
        ).clip(lower=0)

        statistics["rating_std"] = np.sqrt(
            statistics["rating_variance"]
        )

        return statistics[
            [image_col, "mos", "rating_std", "n_ratings"]
        ]


    def resample_stimuli_statistics(
            self,
            ratings,
            rng,
            participant_col="participant_id",
            image_col="image_id",
            rating_col="rating",
        ):
            participants = ratings[participant_col].unique()

            participant_draw = rng.choice(
                participants,
                size=len(participants),
                replace=True,
            )

            # Number of times each participant was selected
            weights = (
                pd.Series(participant_draw)
                .value_counts()
                .rename("bootstrap_weight")
            )

            sampled = ratings.merge(
                weights,
                left_on=participant_col,
                right_index=True,
                how="inner",
            )

            sampled["weighted_rating"] = (
                sampled[rating_col]
                * sampled["bootstrap_weight"]
            )

            sampled["weighted_squared_rating"] = (
                sampled[rating_col] ** 2
                * sampled["bootstrap_weight"]
            )

            statistics = (
                sampled
                .groupby(image_col)
                .agg(
                    n_ratings=("bootstrap_weight", "sum"),
                    weighted_sum=("weighted_rating", "sum"),
                    weighted_squared_sum=(
                        "weighted_squared_rating",
                        "sum",
                    ),
                )
                .reset_index()
            )

            statistics["mos"] = (
                statistics["weighted_sum"]
                / statistics["n_ratings"]
            )

            # Sample variance of the resampled ratings
            variance_numerator = (
                statistics["weighted_squared_sum"]
                - statistics["weighted_sum"] ** 2
                / statistics["n_ratings"]
            )

            statistics["rating_variance"] = (
                variance_numerator
                / (statistics["n_ratings"] - 1)
            ).clip(lower=0)

            statistics["rating_std"] = np.sqrt(
                statistics["rating_variance"]
            )

            return statistics[
                [image_col, "mos", "rating_std", "n_ratings"]
            ]

    def resample_subjective_statistics(
        self,
        ratings,
        rng,
        participant_col="participant_id",
        image_col="image_id",
        rating_col="rating",
    ):
        participants = ratings[participant_col].unique()

        participant_draw = rng.choice(
            participants,
            size=len(participants),
            replace=True,
        )

        # Number of times each participant was selected
        weights = (
            pd.Series(participant_draw)
            .value_counts()
            .rename("bootstrap_weight")
        )

        sampled = ratings.merge(
            weights,
            left_on=participant_col,
            right_index=True,
            how="inner",
        )

        sampled["weighted_rating"] = (
            sampled[rating_col]
            * sampled["bootstrap_weight"]
        )

        sampled["weighted_squared_rating"] = (
            sampled[rating_col] ** 2
            * sampled["bootstrap_weight"]
        )

        statistics = (
            sampled
            .groupby(image_col)
            .agg(
                n_ratings=("bootstrap_weight", "sum"),
                weighted_sum=("weighted_rating", "sum"),
                weighted_squared_sum=(
                    "weighted_squared_rating",
                    "sum",
                ),
            )
            .reset_index()
        )

        statistics["mos"] = (
            statistics["weighted_sum"]
            / statistics["n_ratings"]
        )

        # Sample variance of the resampled ratings
        variance_numerator = (
            statistics["weighted_squared_sum"]
            - statistics["weighted_sum"] ** 2
            / statistics["n_ratings"]
        )

        statistics["rating_variance"] = (
            variance_numerator
            / (statistics["n_ratings"] - 1)
        ).clip(lower=0)

        statistics["rating_std"] = np.sqrt(
            statistics["rating_variance"]
        )

        return statistics[
            [image_col, "mos", "rating_std", "n_ratings"]
        ]

    def resample_crossed_statistics(
        self,
        ratings,
        stimuli,
        rng,
        participant_col="participant_id",
        file_col="file_id",
        rating_col="rating",
        mos_col="mos",
    ):
        """
        Independently resample participants and stimuli.

        ratings:
            Cleaned participant-level table with one row per observed
            participant-stimulus rating.

        stimuli:
            One row per stimulus containing the IQA scores and original MOS.

        Returns:
            An expanded stimulus-level bootstrap sample. Repeated rows
            represent stimulus bootstrap multiplicities, and MOS is
            recomputed using participant bootstrap multiplicities.
        """

        if stimuli[file_col].duplicated().any():
            raise ValueError("Expected one row per stimulus.")

        participants = ratings[participant_col].unique()
        stimulus_ids = stimuli[file_col].to_numpy()

        n_participants = len(participants)
        n_stimuli = len(stimulus_ids)

        # ---------------------------------------------------------
        # 1. Resample participants as clusters
        # ---------------------------------------------------------
        participant_draw = rng.choice(
            participants,
            size=n_participants,
            replace=True,
        )

        participant_weights = (
            pd.Series(participant_draw)
            .value_counts(sort=False)
            .rename("participant_weight")
            .rename_axis(participant_col)
            .reset_index()
        )

        # A participant drawn k times contributes all their ratings k times.
        weighted_ratings = ratings.merge(
            participant_weights,
            on=participant_col,
            how="inner",
            validate="many_to_one",
        )

        weighted_ratings["weighted_rating"] = (
            weighted_ratings[rating_col]
            * weighted_ratings["participant_weight"]
        )

        # ---------------------------------------------------------
        # 2. Recompute MOS from resampled participants
        # ---------------------------------------------------------
        bootstrap_mos = (
            weighted_ratings
            .groupby(file_col)
            .agg(
                weighted_rating_sum=("weighted_rating", "sum"),
                participant_weight_sum=("participant_weight", "sum"),
            )
        )

        bootstrap_mos[mos_col] = (
            bootstrap_mos["weighted_rating_sum"]
            / bootstrap_mos["participant_weight_sum"]
        )

        bootstrap_mos = bootstrap_mos[mos_col]

        # ---------------------------------------------------------
        # 3. Independently resample stimuli
        # ---------------------------------------------------------
        stimulus_draw = rng.choice(
            stimulus_ids,
            size=n_stimuli,
            replace=True,
        )

        # Expanding the dataframe makes repeated stimuli function as
        # bootstrap weights.
        bootstrap_sample = (
            stimuli
            .set_index(file_col)
            .loc[stimulus_draw]
            .reset_index()
            .copy()
        )

        # Preserve the fixed original MOS for diagnostics.
        bootstrap_sample["original_MOS"] = bootstrap_sample[mos_col]

        # Replace it with the participant-resampled MOS.
        bootstrap_sample[mos_col] = (
            bootstrap_sample[file_col]
            .map(bootstrap_mos)
        )

        if bootstrap_sample[mos_col].isna().any():
            missing = bootstrap_sample.loc[
                bootstrap_sample[mos_col].isna(),
                file_col,
            ].unique()

            raise RuntimeError(
                "No selected participant rated the following selected "
                f"stimuli: {missing.tolist()}"
            )

        # Unique identifier for each of the 70 bootstrap draws.
        bootstrap_sample["bootstrap_draw"] = np.arange(n_stimuli)

        stimulus_weights = (
            pd.Series(stimulus_draw)
            .value_counts(sort=False)
            .rename("stimulus_weight")
        )

        return (bootstrap_sample, participant_weights, stimulus_weights)

    def identity_bootstrap_test(self, ratings_long, original_mos, participant_col, image_col, rating_col):
        participant_ids = ratings_long["participant_id"].unique()

        identity_weights = pd.Series(
            1,
            index=participant_ids,
            name="bootstrap_weight",
        )

        identity_mos = self.compute_weighted_mos(
            ratings_long,
            identity_weights,
            participant_col=participant_col,
            rating_col=rating_col,
            image_col=image_col
        ).sort_index()

        original_mos = original_mos.sort_index()

        is_identity = np.testing.assert_allclose(
            identity_mos['mos'].to_numpy(),
            original_mos['mos'].to_numpy(),
            rtol=0,
            atol=1e-12,
        )

        return True if is_identity is None else False


    def compute_pvalue_from_bs(
            self,
            point_estimates,
            bs_delta_arrays,
    ):
        p_values_iqa_difference = {metric: {criterion: 0.0 for criterion in ['plcc', 'srocc', 'or']} for metric in bs_delta_arrays.keys()}

        for metric, bootstrap_p_metric in bs_delta_arrays.items():
            for criterion in ['plcc', 'srocc', 'or']:
                bootstrap_list = bootstrap_p_metric[criterion]
                observed_val = point_estimates[metric].loc[self.stat_2_int[criterion]]
                p_value = DataProcessor.bootstrap_pvalue(observed=observed_val, bootstrap_estimates=bootstrap_list)
                p_values_iqa_difference[metric][criterion] = p_value

        return pd.DataFrame(p_values_iqa_difference)


    @staticmethod
    def bootstrap_pvalue(
        observed,
        bootstrap_estimates,
        null_value=0.0,
        alternative="two-sided",
    ):
        bootstrap_estimates = np.asarray(bootstrap_estimates)

        # Approximate sampling distribution under H0.
        null_distribution = bootstrap_estimates - observed
        observed_distance = observed - null_value

        if alternative == "two-sided":
            extreme = (
                np.abs(null_distribution)
                >= abs(observed_distance)
            )

        elif alternative == "greater":
            extreme = null_distribution >= observed_distance

        elif alternative == "less":
            extreme = null_distribution <= observed_distance

        else:
            raise ValueError(
                "alternative must be 'two-sided', 'greater', or 'less'"
            )

        return (extreme.sum() + 1) / (len(bootstrap_estimates) + 1)

    def get_original_statistics(
            self,
            visualize,

    ):
        # get original MOS, IG-STD, POINT ESTIMATES
        point_estimates = self.compare_to_iqa(
            original_mos=self.original_subjective_statistics, 
            visualize=visualize
            )

        
        print('\nOBTAINED POINT ESTIMATES ON ORIGINAL MOS DATA:\n')
        print(point_estimates)

        self.verify_mos_consistency(ratings=self.participants_clean, stimuli=self.stimuli_df)

        # get Freq, Lum Pearson on original MOS, Point Esimates
        frequency_point_estimates = self.get_freq_correlations(
            self.results_df.copy(deep=True), 
            self.original_subjective_statistics, 
            is_original_mos=True
            )
        
        lum_point_estimates = self.get_color_lum_correlations(
            self.results_df.copy(deep=True), 
            self.original_subjective_statistics, 
            is_original_mos=True
            )

        metric_order = sorted(
            point_estimates,
            key=lambda metric: point_estimates[
                metric
            ].iloc[1],
            reverse=True,
        )

        return {
            "point_estimates": point_estimates,
            "frequency_point_estimates": frequency_point_estimates,
            "lum_point_estimates": lum_point_estimates,
            "metric_order": metric_order
        }

            

    def participant_wise_bootstrap_iqa_comparison(
        self,
        n_bootstrap=N_BOOTSTRAPS,
        participant_col="participant_id",
        image_col="file_id",
        rating_col="rating",
        alpha_value = SIGNIFICANCE_LEVEL,
        visualize=False
    ):

        print("\n=========== computing CI for correlation stats PLCC, SROCC and OR with participant bootstrapping ==========\n")

        checkpoint_metadata = {
            "random_seed": RANDOM_SEED,
            "n_splits": 5,
            "metric_names": list(self.results_df.columns),
            "participant_column": participant_col,
            "image_column": image_col,
            "rating_column": rating_col,
        }

        # lower and upper percentile according to the p-value
        p_percentile = [(alpha_value*100) / 2, 100 - (alpha_value*100) / 2]
        
        # get original POINT ESTIMATES
        original_stats = self.get_original_statistics(visualize=visualize)
        point_estimates = original_stats["point_estimates"]
        point_estimate_plcc_freq, point_estimate_srocc_freq, freq_comps, freq_diffs = original_stats["frequency_point_estimates"]
        point_estimate_plcc_lum, point_estimate_srocc_lum, y_comps, y_diffs = original_stats["frequency_point_estimates"]
        metric_order = original_stats["metric_order"]

        print('\nOBTAINED POINT ESTIMATES ON ORIGINAL MOS DATA:\n')
        print(point_estimates)

        if BS_PART_CHECKPOINT_PATH.exists():
            state = load_bootstrap_checkpoint(
                BS_PART_CHECKPOINT_PATH,
                RNG,
            )

            if state["n_bootstrap"] != n_bootstrap:
                raise RuntimeError(
                    "The requested number of replicates differs from "
                    "the checkpoint configuration."
                )

            completed = state["completed"]
            attempts = state["attempts"]
            bootstrap_results = state["bootstrap_results"]
            bootstrap_ig_stat_results = (
                state["bootstrap_ig_stat_results"]
            )
            failure_log = state["failure_log"]

            print(
                f"Resuming from {completed}/{n_bootstrap} "
                f"completed replicates."
            )

        else:


            bootstrap_results = {
                metric: {
                    "plcc": [],
                    "srocc": [],
                    "or": [],
                }
                for metric in self.results_df.columns[1:]
            }

            bootstrap_ig_stat_results = {
                "freq":{
                    'plcc':[],
                    'srocc':[]
                },
                "lum":{
                    'plcc':[],
                    'srocc':[]
                },
            }

            print('\nIdentity check of participant weighting in resampling algorithm...\n')

            is_identity = self.identity_bootstrap_test(self.participants, 
                                        self.original_subjective_statistics, 
                                        participant_col=participant_col, 
                                        image_col=image_col, 
                                        rating_col=rating_col,)

            print(f'\nIdentity testing of MOS resampling strategy: {is_identity}')

        
            completed = 0
            attempts = 0
            failure_log = []

        # checkpoint loading logic end


        maximum_attempts = n_bootstrap * 3

        while completed < n_bootstrap:
            attempts += 1
            print(f'BOOTSTRAPPING ITERATION :{attempts}\n')

            if attempts > maximum_attempts:
                raise RuntimeError(
                    f"Only {completed}/{n_bootstrap} valid replicates "
                    f"after {attempts - 1} attempts. "
                    f"Failure rate: "
                    f"{len(failure_log) / (attempts - 1):.2%}"
                )

            try:
                # resample dataset, get MOS, IG-STD
                bootstrapped_mos_data = self.resample_subjective_statistics(
                    ratings=self.participants,
                    rng=RNG,
                    participant_col=participant_col,
                    image_col=image_col,
                    rating_col=rating_col,
                )

                # transform metric with resampled MOS data and compute PLCC,SROCC,OR

                metric_df_for_bs_mos = self.results_df.copy(deep=True)
                stats_on_transformed_metrics, folds = self.transform_iqa_with_bs_mos(
                    metric_df_for_bs_mos, 
                    bootstrapped_mos_data, 
                    visualize, 
                    is_original_mos=False, 
                    kfold_grouping=False
                )

                for e, metric in enumerate(bootstrap_results.keys()):
                    bootstrap_results[metric]['plcc'].append(stats_on_transformed_metrics.iloc[0,e])
                    bootstrap_results[metric]['srocc'].append(stats_on_transformed_metrics.iloc[1,e])
                    bootstrap_results[metric]['or'].append(stats_on_transformed_metrics.iloc[3,e])

                # compute frequency correlation with resampled MOS

                bootstrap_plcc_freq, bootstrap_srocc_freq = self.get_freq_correlations(metric_df_for_bs_mos, bootstrapped_mos_data)
                bootstrap_plcc_lum, bootstrap_srocc_lum  = self.get_color_lum_correlations(metric_df_for_bs_mos, bootstrapped_mos_data)

                bootstrap_ig_stat_results['freq']['plcc'].append(bootstrap_plcc_freq)
                bootstrap_ig_stat_results['lum']['plcc'].append(bootstrap_plcc_lum)
                bootstrap_ig_stat_results['freq']['srocc'].append(bootstrap_srocc_freq)
                bootstrap_ig_stat_results['lum']['srocc'].append(bootstrap_srocc_lum)

                completed += 1
                print(f"\nFinished Bootstrapping Iteration {attempts}.\n")


                if completed % CHECKPOINT_EVERY == 0:
                    save_bootstrap_checkpoint(
                        BS_PART_CHECKPOINT_PATH,
                        completed=completed,
                        attempts=attempts,
                        n_bootstrap=n_bootstrap,
                        bootstrap_results=bootstrap_results,
                        bootstrap_ig_stat_results=(
                            bootstrap_ig_stat_results
                        ),
                        failure_log=failure_log,
                        rng=RNG,
                        metadata=checkpoint_metadata,
                    )

                    print(
                        f"Saved checkpoint after "
                        f"{completed} valid replicates."
                    )



            except FivePLFitError as exc:
                failure_log.append({
                    "attempt": attempts,
                    "error": str(exc),
                })

                print(
                    f"Rejected bootstrap attempt {attempts}: {exc}"
                )
                continue

        print('\nComputing alpha-Level quantile from collected bootstraps for every metric..\n')


        p_values_iqa_difference = self.compute_pvalue_from_bs(point_estimates=point_estimates, bootstrap_results=bootstrap_results)


        ci_at_percentile = {metric: {stat:((np.round(np.percentile(iter_vals, p_percentile), decimals=3))) for stat, iter_vals in iter_stats.items()} for metric, iter_stats in bootstrap_results.items()}
        bootstrap_results_df = pd.DataFrame(bootstrap_results)
        ci_at_percentile = pd.DataFrame(ci_at_percentile)
        print('Metric confidence intervals:\n')
        print(ci_at_percentile)
        (ci_at_percentile.T).to_csv("./survey_stats/CI_metrics.csv")
        print('Done.\n')

        ######################

        print('\nChecking suspicious point-estimate to bootstraping ci metric-statistics combinations...\n')

        checks = [
            ("topiq_fr", "plcc"),
            ("lpips", "srocc"),
        ]

        self.check_stats(point_estimates, bootstrap_results, checks)

        print('Done. \n')


        ######################


        print('\nVisualizing CI for all stats...\n')

        self.plot_metric_confidence_intervals(point_estimates=point_estimates, confidence_intervals=ci_at_percentile, metric_order=metric_order)

        print('Done. \n')


        #######################


        print(f'Computing Delta Intervals from bootstrap iterations for Reference {BEST_METRIC_HYPOTHESIS} and targets')

        self.plot_metric_difference_ci(BEST_METRIC_HYPOTHESIS, list(bootstrap_results.keys()), bootstrap_results_df, point_estimates, metric_order)

        print('Done. \n')


        ######################


        print('Create Graph and Confidence Intervals for Color and Lum Correlation Point estimates...\n')

        self.compute_freq_ci_and_plot(
            point_estimate_plcc_freq, 
            point_estimate_srocc_freq, 
            bootstrap_ig_stat_results['freq']['plcc'], 
            bootstrap_ig_stat_results['freq']['srocc'], 
            freq_comps, 
            freq_diffs)
        
        self.compute_color_ci_and_plot(
            point_estimate_plcc_lum, 
            point_estimate_srocc_lum, 
            bootstrap_ig_stat_results['lum']['plcc'],
            bootstrap_ig_stat_results['lum']['srocc'], 
            y_comps, 
            y_diffs)

        print('Done. \n')


    def stimuli_wise_bootstrap_iqa_comparison(
            self,
            n_bootstrap=N_BOOTSTRAPS,
            image_col="file",
            mos_col="MOS",
            alpha_value = SIGNIFICANCE_LEVEL,
            visualize=False
        ):

        print("\n=========== computing CI for correlation stats PLCC, SROCC and OR with stimuli resampling ==========\n")

        checkpoint_metadata = {
            "random_seed": RANDOM_SEED,
            "n_splits": 5,
            "metric_names": list(self.results_df.columns),
            "mos_column": mos_col,
            "image_column": image_col,
        }

        # lower and upper percentile according to the p-value
        p_percentile = [(alpha_value*100) / 2, 100 - (alpha_value*100) / 2]
        
        # get original POINT ESTIMATES
        original_stats = self.get_original_statistics(visualize=visualize)
        point_estimates = original_stats["point_estimates"]
        point_estimate_plcc_freq, point_estimate_srocc_freq, freq_comps, freq_diffs = original_stats["frequency_point_estimates"]
        point_estimate_plcc_lum, point_estimate_srocc_lum, y_comps, y_diffs = original_stats["frequency_point_estimates"]
        metric_order = original_stats["metric_order"]
        
        print('\nOBTAINED POINT ESTIMATES ON ORIGINAL MOS DATA:\n')
        print(point_estimates)

        if BS_STIM_CHECKPOINT_PATH.exists():
            state = load_bootstrap_checkpoint(
                BS_STIM_CHECKPOINT_PATH,
                RNG,
            )

            if state["n_bootstrap"] != n_bootstrap:
                raise RuntimeError(
                    "The requested number of replicates differs from "
                    "the checkpoint configuration."
                )

            completed = state["completed"]
            attempts = state["attempts"]
            bootstrap_results = state["bootstrap_results"]
            bootstrap_ig_stat_results = (
                state["bootstrap_ig_stat_results"]
            )
            failure_log = state["failure_log"]

            print(
                f"Resuming from {completed}/{n_bootstrap} "
                f"completed replicates."
            )

        else:


            bootstrap_results = {
                metric: {
                    "plcc": [],
                    "srocc": [],
                    "or": [],
                }
                for metric in self.results_df.columns[1:]
            }

            bootstrap_ig_stat_results = {
                "freq":{
                    'plcc':[],
                    'srocc':[]
                },
                "lum":{
                    'plcc':[],
                    'srocc':[]
                },
            }

            completed = 0
            attempts = 0
            failure_log = []

        # checkpoint loading logic end


        maximum_attempts = n_bootstrap * 3

        while completed < n_bootstrap:
            attempts += 1
            print(f'BOOTSTRAPPING ITERATION :{attempts}\n')

            if attempts > maximum_attempts:
                raise RuntimeError(
                    f"Only {completed}/{n_bootstrap} valid replicates "
                    f"after {attempts - 1} attempts. "
                    f"Failure rate: "
                    f"{len(failure_log) / (attempts - 1):.2%}"
                )

            try:
                # resample stimuli, get resampled file-iqa-mos dataframe
                bootstrapped_stimuli_df, weight = self.resample_stimuli_statistics(
                    stimuli=self.stimuli_df,
                    rng=RNG,
                    mos_col=mos_col,
                    image_col=image_col,
                )

                # transform metric with resampled MOS data and compute PLCC,SROCC,OR

                bootstrapped_mos_data = bootstrapped_stimuli_df[['file', 'MOS', 'STD']]
                bootstrapped_mos_data = bootstrapped_mos_data.rename({"file":"file_id", "MOS":"mos", "STD":"rating_std"}, axis=1)

                _ = bootstrapped_stimuli_df.pop('MOS')
                _ = bootstrapped_stimuli_df.pop('STD')

                stats_on_transformed_metrics, folds = self.transform_iqa_with_bs_mos(
                    bootstrapped_stimuli_df, 
                    bootstrapped_mos_data, 
                    visualize, 
                    is_original_mos=False, 
                    kfold_grouping=True
                )

                for e, metric in enumerate(bootstrap_results.keys()):
                    bootstrap_results[metric]['plcc'].append(stats_on_transformed_metrics.iloc[0,e])
                    bootstrap_results[metric]['srocc'].append(stats_on_transformed_metrics.iloc[1,e])
                    bootstrap_results[metric]['or'].append(stats_on_transformed_metrics.iloc[3,e])

                # compute frequency correlation with resampled MOS

                bootstrap_plcc_freq, bootstrap_srocc_freq = self.get_freq_correlations(bootstrapped_stimuli_df, bootstrapped_mos_data, folds=folds)
                bootstrap_plcc_lum, bootstrap_srocc_lum  = self.get_color_lum_correlations(bootstrapped_stimuli_df, bootstrapped_mos_data, folds=folds)

                bootstrap_ig_stat_results['freq']['plcc'].append(bootstrap_plcc_freq)
                bootstrap_ig_stat_results['lum']['plcc'].append(bootstrap_plcc_lum)
                bootstrap_ig_stat_results['freq']['srocc'].append(bootstrap_srocc_freq)
                bootstrap_ig_stat_results['lum']['srocc'].append(bootstrap_srocc_lum)

                completed += 1
                print(f"\nFinished Bootstrapping Iteration {attempts}.\n")


                if completed % CHECKPOINT_EVERY == 0:
                    save_bootstrap_checkpoint(
                        BS_STIM_CHECKPOINT_PATH,
                        completed=completed,
                        attempts=attempts,
                        n_bootstrap=n_bootstrap,
                        bootstrap_results=bootstrap_results,
                        bootstrap_ig_stat_results=(
                            bootstrap_ig_stat_results
                        ),
                        failure_log=failure_log,
                        rng=RNG,
                        metadata=checkpoint_metadata,
                    )

                    print(
                        f"Saved checkpoint after "
                        f"{completed} valid replicates."
                    )



            except FivePLFitError as exc:
                failure_log.append({
                    "attempt": attempts,
                    "error": str(exc),
                })

                print(
                    f"Rejected bootstrap attempt {attempts}: {exc}"
                )
                continue

        

        print('\nComputing alpha-Level quantile from collected bootstraps for every metric..\n')

        p_values_iqa_difference = self.compute_pvalue_from_bs(point_estimates=point_estimates, bootstrap_results=bootstrap_results)

        ci_at_percentile = {metric: {stat:((np.round(np.percentile(iter_vals, p_percentile), decimals=3))) for stat, iter_vals in iter_stats.items()} for metric, iter_stats in bootstrap_results.items()}
        bootstrap_results_df = pd.DataFrame(bootstrap_results)
        ci_at_percentile = pd.DataFrame(ci_at_percentile)
        print('Metric confidence intervals:\n')
        print(ci_at_percentile)
        (ci_at_percentile.T).to_csv("./survey_stats/CI_metrics.csv")
        print('Done.\n')

        ######################

        print('\nChecking suspicious point-estimate to bootstraping ci metric-statistics combinations...\n')

        checks = [
            ("topiq_fr", "plcc"),
            ("lpips", "srocc"),
        ]

        self.check_stats(point_estimates, bootstrap_results, checks)

        print('Done. \n')


        ######################


        print('\nVisualizing CI for all stats...\n')

        self.plot_metric_confidence_intervals(point_estimates=point_estimates, confidence_intervals=ci_at_percentile, metric_order=metric_order)

        print('Done. \n')


        #######################


        print(f'Computing Delta Intervals from bootstrap iterations for Reference {BEST_METRIC_HYPOTHESIS} and targets')

        self.plot_metric_difference_ci(BEST_METRIC_HYPOTHESIS, list(bootstrap_results.keys()), bootstrap_results_df, point_estimates, metric_order)

        print('Done. \n')


        ######################


        print('Create Graph and Confidence Intervals for Color and Lum Correlation Point estimates...\n')

        self.compute_freq_ci_and_plot(
            point_estimate_plcc_freq, 
            point_estimate_srocc_freq, 
            bootstrap_ig_stat_results['freq']['plcc'], 
            bootstrap_ig_stat_results['freq']['srocc'], 
            freq_comps, 
            freq_diffs)
        
        self.compute_color_ci_and_plot(
            point_estimate_plcc_lum, 
            point_estimate_srocc_lum, 
            bootstrap_ig_stat_results['lum']['plcc'],
            bootstrap_ig_stat_results['lum']['srocc'], 
            y_comps, 
            y_diffs)

        print('Done. \n')


    def crossed_bootstrap_iqa_comparison(
            self,
            n_bootstrap=N_BOOTSTRAPS,
            image_col="file_id",
            mos_col="MOS",
            alpha_value = SIGNIFICANCE_LEVEL,
            visualize=False
        ):

        print("\n=========== computing CI for correlation stats PLCC, SROCC and OR with two-way crossed resampling ==========\n")


        checkpoint_metadata = {
            "random_seed": RANDOM_SEED,
            "n_splits": 5,
            "metric_names": list(self.results_df.columns),
            "mos_column": mos_col,
            "image_column": image_col,
        }

        # lower and upper percentile according to the p-value
        p_percentile = [(alpha_value*100) / 2, 100 - (alpha_value*100) / 2]

        # get original POINT ESTIMATES
        original_stats = self.get_original_statistics(visualize=visualize)
        point_estimates = original_stats["point_estimates"]
        point_estimate_plcc_freq, point_estimate_srocc_freq, freq_comps, freq_diffs = original_stats["frequency_point_estimates"]
        point_estimate_plcc_lum, point_estimate_srocc_lum, y_comps, y_diffs = original_stats["frequency_point_estimates"]
        metric_order = original_stats["metric_order"]
        
        print('\nOBTAINED POINT ESTIMATES ON ORIGINAL MOS DATA:\n')
        print(point_estimates)
        

        if BS_CRS_CHECKPOINT_PATH.exists():
            state = load_bootstrap_checkpoint(
                BS_CRS_CHECKPOINT_PATH,
                RNG,
            )

            if state["n_bootstrap"] != n_bootstrap:
                raise RuntimeError(
                    "The requested number of replicates differs from "
                    "the checkpoint configuration."
                )

            completed = state["completed"]
            attempts = state["attempts"]
            bootstrap_results = state["bootstrap_results"]
            bootstrap_ig_stat_results = (
                state["bootstrap_ig_stat_results"]
            )
            failure_log = state["failure_log"]

            print(
                f"Resuming from {completed}/{n_bootstrap} "
                f"completed replicates."
            )

        else:


            bootstrap_results = {
                metric: {
                    "plcc": [],
                    "srocc": [],
                    "or": [],
                }
                for metric in self.results_df.columns[1:]
            }

            bootstrap_ig_stat_results = {
                "freq":{
                    'plcc':[],
                    'srocc':[]
                },
                "lum":{
                    'plcc':[],
                    'srocc':[]
                },
            }

            completed = 0
            attempts = 0
            failure_log = []

        # checkpoint loading logic end


        maximum_attempts = n_bootstrap * 3

        while completed < n_bootstrap:
            attempts += 1
            print(f'BOOTSTRAPPING ITERATION :{attempts}\n')

            if attempts > maximum_attempts:
                raise RuntimeError(
                    f"Only {completed}/{n_bootstrap} valid replicates "
                    f"after {attempts - 1} attempts. "
                    f"Failure rate: "
                    f"{len(failure_log) / (attempts - 1):.2%}"
                )

            try:
                # resample stimuli, get resampled file-iqa-mos dataframe
                bootstrapped_df, part_weight, stim_weight = self.resample_crossed_statistics(
                    ratings=self.participants_clean,
                    stimuli=self.stimuli_df,
                    rng=RNG
                )

                # transform metric with resampled MOS data and compute PLCC, SROCC, OR
                


                bootstrapped_mos_data = bootstrapped_df[['file_id', 'mos', 'rating_std', 'original_MOS', 'bootstrap_draw', 'n_ratings']]

                bootstrapped_df.drop(['mos', 'rating_std', 'original_MOS', 'bootstrap_draw', 'n_ratings'], axis=1, inplace=True)


                stats_on_transformed_metrics, folds = self.transform_iqa_with_bs_mos(
                    bootstrapped_df, 
                    bootstrapped_mos_data, 
                    visualize, 
                    is_original_mos=False, 
                    kfold_grouping=True
                )

                for e, metric in enumerate(bootstrap_results.keys()):
                    bootstrap_results[metric]['plcc'].append(stats_on_transformed_metrics.iloc[0,e])
                    bootstrap_results[metric]['srocc'].append(stats_on_transformed_metrics.iloc[1,e])
                    bootstrap_results[metric]['or'].append(stats_on_transformed_metrics.iloc[3,e])

                # compute frequency correlation with resampled MOS

                bootstrap_plcc_freq, bootstrap_srocc_freq = self.get_freq_correlations(bootstrapped_df, bootstrapped_mos_data, folds=folds)
                bootstrap_plcc_lum, bootstrap_srocc_lum  = self.get_color_lum_correlations(bootstrapped_df, bootstrapped_mos_data, folds=folds)

                bootstrap_ig_stat_results['freq']['plcc'].append(bootstrap_plcc_freq)
                bootstrap_ig_stat_results['lum']['plcc'].append(bootstrap_plcc_lum)
                bootstrap_ig_stat_results['freq']['srocc'].append(bootstrap_srocc_freq)
                bootstrap_ig_stat_results['lum']['srocc'].append(bootstrap_srocc_lum)

                completed += 1
                print(f"\nFinished Bootstrapping Iteration {attempts}.\n")


                if completed % CHECKPOINT_EVERY == 0:
                    save_bootstrap_checkpoint(
                        BS_CRS_CHECKPOINT_PATH,
                        completed=completed,
                        attempts=attempts,
                        n_bootstrap=n_bootstrap,
                        bootstrap_results=bootstrap_results,
                        bootstrap_ig_stat_results=(
                            bootstrap_ig_stat_results
                        ),
                        failure_log=failure_log,
                        rng=RNG,
                        metadata=checkpoint_metadata,
                    )

                    print(
                        f"Saved checkpoint after "
                        f"{completed} valid replicates."
                    )


            except FivePLFitError as exc:
                failure_log.append({
                    "attempt": attempts,
                    "error": str(exc),
                })

                print(
                    f"Rejected bootstrap attempt {attempts}: {exc}"
                )
                continue


        print('\nComputing alpha-Level quantile from collected bootstraps for every metric..\n')



        ci_at_percentile = {metric: {stat:((np.round(np.percentile(iter_vals, p_percentile), decimals=3))) for stat, iter_vals in iter_stats.items()} for metric, iter_stats in bootstrap_results.items()}
        bootstrap_results_df = pd.DataFrame(bootstrap_results)
        ci_at_percentile = pd.DataFrame(ci_at_percentile)
        print('Metric confidence intervals:\n')
        print(ci_at_percentile)
        (ci_at_percentile.T).to_csv("./survey_stats/CI_metrics.csv")
        print('Done.\n')

        ######################

        print('\nChecking suspicious point-estimate to bootstraping ci metric-statistics combinations...\n')

        checks = [
            ("topiq_fr", "plcc"),
            ("lpips", "srocc"),
        ]

        self.check_stats(point_estimates, bootstrap_results, checks)

        print('Done. \n')


        ######################


        print('\nVisualizing CI for all stats...\n')

        self.plot_metric_confidence_intervals(point_estimates=point_estimates, confidence_intervals=ci_at_percentile, metric_order=metric_order)

        print('Done. \n')


        #######################


        print(f'Computing Delta Intervals from bootstrap iterations for Reference {BEST_METRIC_HYPOTHESIS} and targets')

        self.plot_metric_difference_ci(BEST_METRIC_HYPOTHESIS, list(bootstrap_results.keys()), bootstrap_results_df, point_estimates, metric_order)

        print('Done. \n')


        ######################


        print('Create Graph and Confidence Intervals for Color and Lum Correlation Point estimates...\n')

        self.compute_freq_ci_and_plot(
            point_estimate_plcc_freq, 
            point_estimate_srocc_freq, 
            bootstrap_ig_stat_results['freq']['plcc'], 
            bootstrap_ig_stat_results['freq']['srocc'], 
            freq_comps, 
            freq_diffs)
        
        self.compute_color_ci_and_plot(
            point_estimate_plcc_lum, 
            point_estimate_srocc_lum, 
            bootstrap_ig_stat_results['lum']['plcc'],
            bootstrap_ig_stat_results['lum']['srocc'], 
            y_comps, 
            y_diffs)

        print('Done. \n')




    ##########################################################################################


    # IQA-MOS AGREEMENT ANALYSIS


    ##########################################################################################


    def compare_to_iqa(self, original_mos, visualize=False):
        results_df = self.results_df.copy(deep=True)
        _ = results_df.pop('file_id') 

        rating_std = self.metric_collection.ig_statistics['rating_std'].to_numpy()

        metrics_df = results_df.apply(self.metric_collection, args=(original_mos['mos'], rating_std), visualize=visualize, is_original_mos=True, folds=None, axis=0)
        return metrics_df
                    
    def get_iqa_for_file(self, file, pert_mag, attack):
        iqa_path = self.iqa_root + f'{pert_mag}/{attack}/iqa_report.csv'
        with open(iqa_path, 'r') as f:
            csv_obj = csv.reader(f)
            for line in csv_obj:
                if line[0].endswith(str(file)):
                    return [float(i) for i in line[1:]]

    def calculate_statistics(
        self,
        data,
        metric,
        mos_col="mos",
        std_col="rating_std",
    ):
        predicted_mos = data[metric].to_numpy(dtype=float)
        mos = data[mos_col].to_numpy(dtype=float)
        rating_std = data[std_col].to_numpy(dtype=float)

        if not (
            predicted_mos.shape
            == mos.shape
            == rating_std.shape
        ):
            raise ValueError("Input arrays have different shapes.")

        plcc = pearsonr(predicted_mos, mos).statistic
        srocc = spearmanr(predicted_mos, mos).statistic

        # Fezza/legacy IQA definition
        is_outlier = (
            np.abs(predicted_mos - mos)
            > 2.0 * rating_std
        )

        outlier_ratio = is_outlier.mean()

        return plcc, srocc, outlier_ratio


    def plot_metric_difference_ci(self, metric, targets, b_results_df, point_estimates, metric_order):

        metric_order = metric_order[1:] # remove AHIQ 
        

        point_estimate_delta_plcc = point_estimates[BEST_METRIC_HYPOTHESIS][0] - point_estimates[:].iloc[0]
        point_estimate_delta_srocc = point_estimates[BEST_METRIC_HYPOTHESIS][1] - point_estimates[:].iloc[1]
        point_estimate_delta_or = point_estimates[:].iloc[3] - point_estimates[BEST_METRIC_HYPOTHESIS][3]
        point_estimate_delta = pd.concat((point_estimate_delta_plcc, point_estimate_delta_srocc, point_estimate_delta_or), axis=1).T
        delta_interval_dict = {}
        delta_array_dict = {}

        for target in targets:
            if target != metric:
                delta_intervals, bs_delta_arrays = self.metric_ci_comparison(b_results_df, metric, target)
                delta_interval_dict[target] = delta_intervals
                delta_array_dict[target] = bs_delta_arrays


        delta_interval_df = pd.DataFrame(delta_interval_dict)

        delta_interval_df.drop('l2', axis=1)
        point_estimates.drop('l2', axis=1)

        p_values_iqa_difference = self.compute_pvalue_from_bs(point_estimates=point_estimate_delta, bs_delta_arrays=delta_array_dict)

        p_holm, p_raw, reject_h0 = holm_correction(p_values_iqa_difference) 

        stat_2_int = {
                    'plcc':0,
                    'srocc':1,
                    'or':3
                }

        sns.set_theme(
            style="whitegrid",
            context="paper",
            font_scale=FONT_SCALE
        )

        y_positions = np.arange(len(metric_order))

        figure, axes = plt.subplots(
            1,
            3,
            figsize=(11, 5.2),
            sharey=True,
            constrained_layout=True,
            gridspec_kw={
                "width_ratios": [1, 1, 0.9],
            },
        )

        panel_configuration = [
            ("plcc", r"$\Delta$ PLCC", 1.0),
            ("srocc", r"$\Delta$ SROCC", 1.0),
            ("or", r"$\Delta$ OR", 1.0),
        ]

        for axis, (
            statistic,
            title,
            scale,
        ) in zip(axes, panel_configuration):

            estimates = np.array([
                point_estimate_delta[metric][stat_2_int[statistic]]
                * scale
                for metric in metric_order
            ])

            lower = np.array([
                delta_interval_df[
                    metric
                ][statistic][0] * scale
                for metric in metric_order
            ])


            below_0 = lower <= 0.0

            upper = np.array([
                delta_interval_df[
                    metric
                ][statistic][1] * scale
                for metric in metric_order
            ])

            colors = ["0.55" if not below_0[e] else "#D55E00" for e, _ in enumerate(metric_order)]

            # Confidence interval lines
            axis.hlines(
                y=y_positions,
                xmin=lower,
                xmax=upper,
                color=colors,
                linewidth=1.8,
                zorder=1,
            )

            # Small end caps
            cap_height = 0.12

            axis.vlines(
                lower,
                y_positions - cap_height,
                y_positions + cap_height,
                color="0.55",
                linewidth=1,
            )

            axis.vlines(
                upper,
                y_positions - cap_height,
                y_positions + cap_height,
                color="0.55",
                linewidth=1,
            )

            # Other metrics
            regular_indices = [
                index
                for index, metric in enumerate(metric_order)
                if metric != "ahiq"
            ]

            axis.scatter(
                estimates[regular_indices],
                y_positions[regular_indices],
                s=32,
                color="0.25",
                zorder=2,
            )

            axis.set_title(title)
            axis.grid(axis="y", visible=False)
            axis.grid(
                axis="x",
                color="0.88",
                linewidth=0.7,
            )

            # If CI crosses 0 we cannot reject the 0-hypothesis
            axis.axvline(
                0,
                color="0.4",
                linestyle="--",
                linewidth=1,
                zorder=0,
            )

            axis.set_xlim(-0.05, 0.4)

        axes[0].set_yticks(y_positions)
        axes[0].set_yticklabels([
            self.display_names.get(metric, metric)
            for metric in metric_order
        ])

        axes[0].invert_yaxis()

        for axis in axes[1:]:
            axis.tick_params(
                axis="y",
                labelleft=False,
            )

        sns.despine(figure)

        figure.suptitle(
            "95% CI of Correlation Metric Differences (Delta) of AHIQ and remaining FR-IQA metrics",
            fontsize=TITLE_SIZE,
            fontweight='bold'
        )

        plt.savefig('./visualizations/delta_confidence_intervals.png')
        return figure, axes


    def metric_ci_comparison(self, bootstrap_results_df, metric, target, alpha_value=SIGNIFICANCE_LEVEL):

        p_percentile = [(alpha_value*100) / 2, 100 - (alpha_value*100) / 2]

        metric_array = np.vstack(bootstrap_results_df[metric].to_numpy()).astype(np.float64)
        target_array = np.vstack(bootstrap_results_df[target].to_numpy()).astype(np.float64)

        plcc_delta_array = metric_array[0] - target_array[0]
        srocc_delta_array = metric_array[1] - target_array[1]
        or_delta_array = target_array[2] - metric_array[2]

        plcc_delta_intervals = np.percentile(plcc_delta_array, p_percentile)
        srocc_delta_intervals = np.percentile(srocc_delta_array, p_percentile)
        or_delta_intervals = np.percentile(or_delta_array, p_percentile)

        print(f'\n\tConfidence Intervals at confidence level: {100 - (alpha_value*100)} for reference {metric.upper()} and target {target.upper()}\n')

        print('\t\tDelta PLCC: ' , plcc_delta_intervals)
        print('\t\tDelta SROCC: ' , srocc_delta_intervals)
        print('\t\tDelta OR: ' , or_delta_intervals)

        return {'plcc': plcc_delta_intervals, 'srocc' : srocc_delta_intervals, 'or': or_delta_intervals}, {'plcc': plcc_delta_array, 'srocc' : srocc_delta_array, 'or': or_delta_array}

    def plot_metric_confidence_intervals(
        self,
        point_estimates,
        confidence_intervals,
        metric_order
    ):

        confidence_intervals.drop('l2', axis=1)
        point_estimates.drop('l2', axis=1)
        metric_order.remove('l2')

        stat_2_int = {
            'plcc':0,
            'srocc':1,
            'or':3
        }

        sns.set_theme(
            style="whitegrid",
            context="paper",
            font_scale=FONT_SCALE,
        )

        y_positions = np.arange(len(metric_order))

        figure, axes = plt.subplots(
            1,
            3,
            figsize=(11, 5.2),
            sharey=True,
            constrained_layout=True,
            gridspec_kw={
                "width_ratios": [1, 1, 0.9],
            },
        )

        panel_configuration = [
            ("plcc", "PLCC ↑", 1.0),
            ("srocc", "SROCC ↑", 1.0),
            ("or", "Outlier ratio (%) ↓", 100.0),
        ]

        for axis, (
            statistic,
            title,
            scale,
        ) in zip(axes, panel_configuration):

            estimates = np.array([
                point_estimates[metric][stat_2_int[statistic]]
                * scale
                for metric in metric_order
            ])

            lower = np.array([
                confidence_intervals[
                    metric
                ][statistic][0] * scale
                for metric in metric_order
            ])

            

            upper = np.array([
                confidence_intervals[
                    metric
                ][statistic][1] * scale
                for metric in metric_order
            ])

            # Confidence interval lines
            axis.hlines(
                y=y_positions,
                xmin=lower,
                xmax=upper,
                color="0.55",
                linewidth=1.8,
                zorder=1,
            )

            # Small end caps
            cap_height = 0.12

            axis.vlines(
                lower,
                y_positions - cap_height,
                y_positions + cap_height,
                color="0.55",
                linewidth=1,
            )

            axis.vlines(
                upper,
                y_positions - cap_height,
                y_positions + cap_height,
                color="0.55",
                linewidth=1,
            )

            # Other metrics
            regular_indices = [
                index
                for index, metric in enumerate(metric_order)
                if metric != "ahiq"
            ]

            axis.scatter(
                estimates[regular_indices],
                y_positions[regular_indices],
                s=32,
                color="0.25",
                zorder=2,
            )

            # Highlight AHIQ
            if "ahiq" in metric_order:
                ahiq_index = metric_order.index("ahiq")

                axis.scatter(
                    estimates[ahiq_index],
                    y_positions[ahiq_index],
                    s=48,
                    color="#D55E00",
                    edgecolor="white",
                    linewidth=0.6,
                    zorder=3,
                    label="AHIQ",
                )

            axis.set_title(title)
            axis.grid(axis="y", visible=False)
            axis.grid(
                axis="x",
                color="0.88",
                linewidth=0.7,
            )

        axes[0].set_yticks(y_positions)
        axes[0].set_yticklabels([
            self.display_names.get(metric, metric)
            for metric in metric_order
        ])

        axes[0].invert_yaxis()

        for axis in axes[1:]:
            axis.tick_params(
                axis="y",
                labelleft=False,
            )

        sns.despine(figure)

        figure.suptitle(
            "Agreement of FR-IQA metrics with subjective MOS",
            fontsize=TITLE_SIZE,
            fontweight='bold'
        )

        plt.savefig('./visualizations/metric_confidence_intervals.png')
        return figure, axes

    def paired_metric_difference(
        self,
        bootstrap_results,
        reference_metric,
        competitor,
        statistic,
    ):
        reference = np.asarray(
            bootstrap_results[
                reference_metric
            ][statistic]
        )

        comparison = np.asarray(
            bootstrap_results[
                competitor
            ][statistic]
        )

        if statistic == "or":
            # Positive means lower OR for reference metric
            differences = comparison - reference
        else:
            # Positive means higher correlation for reference
            differences = reference - comparison

        estimate = differences.mean()

        lower, upper = np.percentile(
            differences,
            [2.5, 97.5],
        )

        return estimate, lower, upper

    def transform_iqa_with_bs_mos(
            self, 
            metric_df, 
            mos_df, 
            visualize=False, 
            is_original_mos=False,
            kfold_grouping=False):

        #align MOS,STD and Metrics
        mos_df = mos_df.sort_values(by='file_id')
        metric_df = metric_df.sort_values(by='file_id')

        # extract MOS and STD
        mos_scores = mos_df['mos']#.to_numpy()
        rating_std = mos_df['rating_std']#.to_numpy()

        if kfold_grouping:
            # to prevent training data leakage for stimuli-based resamplings
            group_cv = GroupKFold(n_splits=5)
            folds = list(
                group_cv.split(
                    X=metric_df,
                    groups=metric_df["file_id"],
                )
            )
        else:
            folds = None

        _ = metric_df.pop('file_id') # remove the file column from the metric df

        metrics_df = metric_df.apply(self.metric_collection, args=(mos_scores, rating_std), visualize=visualize, is_original_mos=False, folds=folds, axis=0)
        return metrics_df, folds
    
    # Step 3: Compare MOS to (1) frequency features (radial profiles) and (2) luminance and color features (Y,Cb,CR)


    ## (1) Frequency

    def compute_freq_ci_and_plot(self, point_estimate_plcc, point_estimate_srocc, bootstrap_results_plcc, bootstrap_results_srocc, freq_comps, diffs, alpha_value=SIGNIFICANCE_LEVEL):

        sns.set_theme(
            style="whitegrid",
            context="paper",
            font_scale=FONT_SCALE
        )

        p_percentile = [(alpha_value*100) / 2, 100 - (alpha_value*100) / 2]

        plcc = point_estimate_plcc.statistic
        plcc_lower, plcc_upper = np.percentile(
            bootstrap_results_plcc,
            p_percentile
        )


        srocc = point_estimate_srocc.statistic
        srocc_lower, srocc_upper = np.percentile(
            bootstrap_results_srocc,
            p_percentile
        )

        fig, ax = plt.subplots(figsize=(8.5,5.0), constrained_layout=True)

        
        df = pd.DataFrame({'Mean Frequency': freq_comps, 'Delta':diffs})

        ax.scatter(
            freq_comps,
            diffs,
            marker="o",
            s=48,
            facecolors="white",
            edgecolors="#0072B2",
            linewidths=1.3,
            label="No",
            zorder=3,
        )

        sns.regplot(
            data=df,
            x="Mean Frequency",
            y="Delta",
            scatter=False,
            robust=True,
            ci=95,
            color="0.25",
            line_kws={"linewidth": 1.4},
            ax=ax,
        )

        ax.text(
            0.98,
            0.04,
            (
                rf"$\rho={srocc:.2f}$, "
                rf"95% CI [{srocc_lower:.2f}, {srocc_upper:.2f}], "
                rf"$n={len(bootstrap_results_srocc)}$"
            ),
            transform=ax.transAxes,
            ha="right",
            va="bottom",
        )

        print('Correlation stats result for Frequency-Delta relationship:\n')
        print(f'\tplcc:{plcc}, 95% CI [{plcc_lower},{plcc_upper}]')
        print(f'\trho:{srocc}, 95% CI [{srocc_lower},{srocc_upper}]\n')

        plt.title("Mean Frequency vs Difference between Actual and Perceived Distortion",fontweight='bold', fontsize=TITLE_SIZE)

        plt.savefig('./visualizations/frequency_percdelta.png')



    def compute_color_ci_and_plot(self, point_estimate_plcc,point_estimate_srocc,  bootstrap_results_plcc, bootstrap_results_srocc, y_comps, y_diffs, alpha_value=SIGNIFICANCE_LEVEL):

        sns.set_theme(
            style="whitegrid",
            context="paper",
            font_scale=FONT_SCALE
        )

        p_percentile = [(alpha_value*100) / 2, 100 - (alpha_value*100) / 2]

        plcc = point_estimate_plcc.statistic
        plcc_lower, plcc_upper = np.percentile(
            bootstrap_results_plcc,
            p_percentile
        )

        srocc = point_estimate_srocc.statistic
        srocc_lower, srocc_upper = np.percentile(
            bootstrap_results_srocc,
            p_percentile
        )

        fig, ax = plt.subplots(figsize=(8.5,5.0), constrained_layout=True)

        
        df = pd.DataFrame({'Mean Luminance': y_comps, 'Delta':y_diffs})

        ax.scatter(
            y_comps,
            y_diffs,
            marker="o",
            s=48,
            facecolors="white",
            edgecolors="#0072B2",
            linewidths=1.3,
            label="No",
            zorder=3,
        )

        sns.regplot(
            data=df,
            x="Mean Luminance",
            y="Delta",
            scatter=False,
            robust=True,
            ci=95,
            color="0.25",
            line_kws={"linewidth": 1.4},
            ax=ax,
        )

        ax.text(
            0.98,
            0.04,
            (
                rf"$\rho={srocc:.2f}$, "
                rf"95% CI [{srocc_lower:.2f}, {srocc_upper:.2f}], "
                rf"$n={len(bootstrap_results_srocc)}$"
            ),
            transform=ax.transAxes,
            ha="right",
            va="bottom",
        )

        print('Correlation stats result for Luminance-Delta relationship:\n')
        print(f'\tplcc:{plcc}, 95% CI [{plcc_lower},{plcc_upper}]')
        print(f'\trho:{srocc}, 95% CI [{srocc_lower},{srocc_upper}]')

        plt.title("Mean Luminance vs Difference between Actual and Perceived Distortion",fontweight='bold', fontsize=TITLE_SIZE)

        plt.savefig('./visualizations/luminance_percdelta.png')


    ##########################################################################################


    # IMAGE CHARACTERISTIC - PERCEPTUAL DISTORTION - RELATIONSHIP ANALYSIS


    ##########################################################################################

    
    def get_freq_correlations(self, metric_df, mos_df, is_original_mos=False, folds=None):
        # compute mean frequency per image
        # compute difference bewteen MOS result and distortion class

        #align MOS,STD and Metrics
        mos_df = mos_df.sort_values(by='file_id')
        metric_df = metric_df.sort_values(by='file_id')

        # extract MOS and STD
        mos_scores = mos_df['mos'].to_numpy()
        rating_std = mos_df['rating_std'].to_numpy()

        files_df = metric_df.pop('file_id') # remove the file column from the metric df

        l2_results = metric_df['l2']
        l2_results = self.metric_collection.fit_and_transform(l2_results, mos_scores, visualize=False, is_original_mos=is_original_mos, folds=folds) # transformed iqm
        frequency_components_df = files_df.apply(self.get_frequency_components)

        if is_original_mos:
            plcc_result, srocc_result, freq_comps, diffs = self.compute_freq_correlations_lsq(mos_scores, l2_results, frequency_components_df, is_original_mos)
            return plcc_result, srocc_result, freq_comps, diffs 
        else:
            plcc_result, srocc_result = self.compute_freq_correlations_lsq(mos_scores, l2_results, frequency_components_df, is_original_mos)
            return plcc_result, srocc_result


    def get_color_lum_correlations(self, metric_df, mos_df, is_original_mos=False, folds=None):
        # compute mean frequency per image
        # compute difference bewteen MOS result and distortion class

        #align MOS,STD and Metrics
        mos_df = mos_df.sort_values(by='file_id')
        metric_df = metric_df.sort_values(by='file_id')

        # extract MOS and STD
        mos_scores = mos_df['mos'].to_numpy()
        rating_std = mos_df['rating_std'].to_numpy()

        files_df = metric_df.pop('file_id') # remove the file column from the metric df


        l2_results = metric_df['l2']
        l2_results = self.metric_collection.fit_and_transform(l2_results, mos_scores, visualize=False, is_original_mos=is_original_mos, folds=folds) # transformed iqm
        clr_components_df = files_df.apply(self.get_color_lum_components)

        if is_original_mos:
            plcc_results, srocc_result, y_comps, y_diffs = self.compute_clr_lum_correlations_lsq(mos_scores, l2_results, clr_components_df, is_original_mos)
            return plcc_results, srocc_result, y_comps, y_diffs 
        else:
            plcc_results, srocc_result = self.compute_clr_lum_correlations_lsq(mos_scores, l2_results, clr_components_df, is_original_mos)
            return plcc_results, srocc_result
    
    def get_color_lum_components(self, file):
        attack, pert = file.split('-')
        file = self.dist_dict[attack][pert]
        path_to_ori = self.iqa_root / ('images/' + str(file) + '.png')
        img = Image.open(path_to_ori)
        img = img.convert('YCbCr')
        img = self.to_tensor(img)
        color_lum = img.mean(dim=(1,2))
        return color_lum.tolist()

    def compute_freq_correlations_lsq(self, mos_scores, l2_results, frequency_components_df, is_original_mos=False):
        frequency_components = frequency_components_df.to_numpy()
        differences = l2_results - mos_scores
        plcc_res = pearsonr(frequency_components, differences)
        srocc_res = spearmanr(frequency_components, differences)

        # plot freq correlations
        # x - frequency component y - difference
        # hopefully, as the mean freq increases, so will teh gap between l2 and mos

        zipped = zip(frequency_components, differences)
        # sort by mean frequency
        sorted_zipped = sorted(zipped, key=lambda x: x[0])

        freq_comps = [x[0] for x in sorted_zipped]
        diffs = [x[1] for x in sorted_zipped]

        if is_original_mos:
            return plcc_res, srocc_res, freq_comps, diffs
        else:
            return plcc_res, srocc_res

    def get_frequency_components(self, file):
        attack, pert = file.split('-')
        file = self.dist_dict[attack][pert]
        path_to_ori = self.iqa_root / ('images/' + str(file) + '.png')
        img = Image.open(path_to_ori)
        img = self.to_tensor(img)
        rad_profile = self.dataset_contructor.compute_rad_profile(img)
        denom = rad_profile.sum()
        weighter = torch.arange(0,len(rad_profile))
        counter = rad_profile * weighter
        counter = counter.mean()
        return (counter / denom).item()
    
    ## (2) Luminance and color features
    
    def compute_clr_lum_correlations_lsq(self, mos_scores, l2_results, clr_components_df, is_original_mos):
        clr_components = clr_components_df.to_list()

        y_components = np.array([x[0] for x in clr_components])

        differences = l2_results - mos_scores

        plcc_res = pearsonr(y_components, differences)
        srocc_res = pearsonr(y_components, differences)

        y_zipped = zip(y_components, differences)

        # sort by mean color
        y_sorted_zipped = sorted(y_zipped, key=lambda x: x[0])


        y_comps = [x[0] for x in y_sorted_zipped]
        y_diffs = [x[1] for x in y_sorted_zipped]


        if is_original_mos:
            return plcc_res, srocc_res, y_comps, y_diffs
        else:
            return plcc_res, srocc_res

def holm_correction(p_value_df, alpha=SIGNIFICANCE_LEVEL):

    stacked = p_value_df.stack()
    
    labels = (
        stacked.index
        .to_flat_index()
        .to_numpy()
    )

    raw_pvalues_array = stacked.to_numpy()

    rejected, adjusted_pvalues, _, _ = multipletests(
        raw_pvalues_array,
        alpha=alpha,
        method="holm",
    )

    results  =  pd.DataFrame({
        "comparison": labels,
        "p_raw": raw_pvalues_array,
        "p_holm": adjusted_pvalues,
        "reject_h0": rejected,
    }, index=stacked.index)

    p_holm, p_raw, reject_h0 = results["p_holm"].unstack(), results["p_raw"].unstack(), results["reject_h0"].unstack()

    print("###################################################################################\n")

    print("Metric difference hypothesis testing using Holm-adjusted p-values from two-way crossed participant / stimuli resampling\n")

    print("###################################################################################\n")


    print("###################################################################################\n")

    print("Raw p-values:\n")
    print(p_raw, "\n")

    print("###################################################################################\n")



    print("###################################################################################\n")

    print("Holm-adjusted p-values:\n")
    print(p_holm, "\n")

    print("###################################################################################\n")



    print("###################################################################################\n")

    print("Reject 0-hypothesis (H-0: no statistically resolvable difference between AHIQ and remaining metric):\n")
    print(reject_h0, "\n")

    print("###################################################################################\n")



    return p_holm, p_raw, reject_h0


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='data.csv', help='set the path to data csv')
    parser.add_argument('--resampling_target', type=str, default='crossed', help='set one of ["participants", "stimuli", "crossed"]')
    parser.add_argument('--device', type=str, default='cpu', help='set the device')

    args = parser.parse_args()

    data_processor = DataProcessor(args.device)

    # process survey results, compute MOS, compute survey statistics
    data_processor.process_data(args.data_path)
    data_processor.get_iqa_and_mos()
    data_processor.survey_statistics()

    ci_analysis = CIAnalysis(
        ratings=data_processor.participants_clean,
        l2_distortions=data_processor.l2_dict,
        file_dict=data_processor.f_dict
    )
    ci_analysis.plot(visualize=False)
    ci_analysis.extend_results_with_intervals(dp_obj=data_processor)

    
    ci_analysis.ci_df.drop("mos", axis=1, inplace=True)
    results_df = pd.merge(data_processor.stimuli_df, ci_analysis.ci_df, on='file_id')
    results_df.to_csv("./data/perc_eval/scores_and_mos.csv", index=False)



    print("\n====================================\n")

    #fitting_stability_results = data_processor.metric_collection.quantify_fitting_stability(data_processor.stimuli_df, RNG, n_repeats=100)


    if args.resampling_target == 'participants':
        correlation_ci = data_processor.participant_wise_bootstrap_iqa_comparison()
    elif args.resampling_target == 'stimuli':
        correlation_ci = data_processor.stimuli_wise_bootstrap_iqa_comparison()
    elif args.resampling_target == 'crossed':
        correlation_ci = data_processor.crossed_bootstrap_iqa_comparison()
    else:
        print('resampling target [{args.resampling_target}] not implemented.')

    print('done')


