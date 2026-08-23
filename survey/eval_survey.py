import argparse
import csv
import torch
import matplotlib.pyplot as plt
import os
import math
import torchvision.transforms as T
import numpy as np
import pandas as pd
import copy
import seaborn as sns

from scipy.optimize import curve_fit
from scipy.stats import pearsonr, spearmanr, t
from process import DatasetConstructor
from PIL import Image
from sklearn.model_selection import KFold
from scipy.special import expit
from pathlib import Path

from bootstrap_checkpoint import save_bootstrap_checkpoint, load_bootstrap_checkpoint
from advanced_survey_stats import repeated_split_half_reliability, summarize_split_half, participant_leave_one_out_consistency, summarize_participant_consistency


RANDOM_SEED = 42
RNG = np.random.default_rng(RANDOM_SEED)
FONT_SCALE= 1.6
TITLE_SIZE = 16
CHECKPOINT_PATH = Path(
    "./saves/bootstrap_iqa_checkpoint.pkl"
)
CHECKPOINT_EVERY = 25
N_BOOTSTRAPS = 5000

class FivePLFitError(RuntimeError):
    """The 5PL model could not be fitted for a metric/fold."""

class DataProcessor:

    def __init__(self):
        self.n = 0
        self.iqa_root = './data/perc_eval/'
        self.results_csv = self.iqa_root + 'mos_results.csv'
        if os.path.exists(self.results_csv):
            os.remove(self.results_csv)
        with open(self.results_csv, 'w') as n_f:
            csv_obj_new = csv.writer(n_f)
            csv_obj_new.writerow(['file','l2',	'mad',	'ssim',	'psnr',	'ssim',	'vif',	'dists',	'lpips',	'ahiq',	'topiq_fr', 'MOS'])
        self.attacks = ['pgdl2', 'vmifgsm', 'percal', 'sgd_uap', 'sparsesigmaattack', 'hpf_fgsm', 'square_attack', 'ppba', 'rcw', 'cgnc', 'cgsp', 'dim', 'bsr', 'sga_uap']
        self.votes = self.set_up_votes()
        self.to_tensor = T.ToTensor()
        self.dataset_contructor = DatasetConstructor()
        self.answer_to_value = {
            '1': 0,
            '2': 0.25,
            '3': 0.5,
            '5': 0.75,
            '6': 1
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

        for attack, perts in self.votes.items():
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
        print(f'\n(6) total number of ratings: {survey_stat_df['n_votes'].sum()}')

        print("Summarized survey statistics per Image:\n")

        print(survey_stat_df)
        survey_stat_df.to_csv("./survey_stats/survey_stat_summary_image.csv")
        survey_stat_df.to_latex("./survey_stats/survey_stat_summary_image.tex")


        split_half_results = (
            repeated_split_half_reliability(
                self.participants,
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
    
    ###### Step 1 : Process surveys into votes CSV, Calculate MOS and put the results in the iqa_reports of the dataset
    
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
    
    def compute_mos_per_sample(self):
        # TODO: change to MOS per image
        file_2_id = {}
        ig_stats_list = []
        self.rm_outliers()
        file_2_inf_keys = self.file_2_inf.keys()
        with open(self.results_csv, 'a') as n_f:
            csv_obj_new = csv.writer(n_f)
            for file in range(280):
                if str(file) in file_2_inf_keys:
                    attack, pert_deg = self.file_2_inf[str(file)]
                    file_path = f'./data/survey/images/{file}'
                    file_2_id[file_path] = "-".join([attack,pert_deg])
                    path = f'{self.iqa_root}{pert_deg}/{attack}/iqa_report.csv'
                    with open(path, 'r+') as f:
                        csv_obj = csv.reader(f)
                        for line in csv_obj:
                            if line[0] == file_path:
                                self.l2_dict[attack][pert_deg] = float(line[1])
                                mos = self.mos(self.votes[attack][pert_deg])
                                ig_sigma = np.array(self.votes[attack][pert_deg]).std()
                                if mos == -1:
                                    mos = self.l2_to_mos(float(line[1]))
                                ig_stats_list.append((file, mos, ig_sigma))
                                line.append(mos)
                                csv_obj_new.writerow(line)
                                break
        print('- MOS computed...')
        self.results_df = pd.read_csv(self.results_csv)
        mos_np = self.results_df.pop('MOS').to_numpy()
        self.ig_stats_df = pd.DataFrame(ig_stats_list, columns=['File', 'MOS', 'STD'])
        self.metric_collection = MetricCollection(mos_np, ig_statistics=self.ig_stats_df)
        self.files_df = self.results_df['file']
        self.results_df['file'] = self.results_df['file'].replace(file_2_id)
        print('done')


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


    # Step 2: Compare MOS (which is now in the iqa report file) to the IQA methods. Use SROCC, PLCC, RMSE, OL

    def compare_to_iqa(self, mos_scores, visualize=False):
        results_df = self.results_df.copy(deep=True)
        _ = results_df.pop('file') 

        rating_std = self.metric_collection.ig_statistics['STD'].to_numpy()

        results_df_transformed = results_df.apply(self.metric_collection.fit_and_transform, args=(mos_scores, visualize, True), raw=True, axis=0, result_type='broadcast')
        metrics_df = results_df_transformed.apply(self.metric_collection, args=(mos_scores, rating_std), axis=0)
        return metrics_df
                    
    def get_iqa_for_file(self, file, pert_mag, attack):
        iqa_path = self.iqa_root + f'{pert_mag}/{attack}/iqa_report.csv'
        with open(iqa_path, 'r') as f:
            csv_obj = csv.reader(f)
            for line in csv_obj:
                if line[0].endswith(str(file)):
                    return [float(i) for i in line[1:]]

    def check_stats(self, point_estimates, bootstrap_results, checks):

        stat_2_int = {
            'plcc':0,
            'srocc':1,
            'or':3
        }

        for metric, statistic in checks:
            values = np.asarray(
                bootstrap_results[metric][statistic],
                dtype=float,
            )

            point = point_estimates[metric][stat_2_int[statistic]]
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

    def bootstrap_iqa_comparison(
        self,
        n_bootstrap=N_BOOTSTRAPS,
        participant_col="participant_id",
        image_col="file_id",
        rating_col="rating",
        p_value = 0.05,
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
        p_percentile = [(p_value*100) / 2, 100 - (p_value*100) / 2]
        
        # get original MOS, IG-STD, POINT ESTIMATES
        original_mos = self.original_subjective_statistics(self.participants)
        point_estimates = self.transform_iqa_with_bs_mos(self.results_df.copy(deep=True), original_mos, visualize=visualize, is_original_mos=True)
        print('\nOBTAINED POINT ESTIMATES ON ORIGINAL MOS DATA:\n')
        print(point_estimates)

        # get Freq, Lum Pearson on original MOS
        point_estimate_plcc_freq, point_estimate_srocc_freq, freq_comps, freq_diffs = self.get_freq_correlations(
            self.results_df.copy(deep=True), 
            original_mos, 
            is_original_mos=True
            )
        
        
        point_estimate_plcc_lum, point_estimate_srocc_lum, y_comps, y_diffs = self.get_color_lum_correlations(
            self.results_df.copy(deep=True), 
            original_mos, 
            is_original_mos=True
            )

        #point_estimates = pd.DataFrame({'l2': {0: 0.86245, 1: 0.83273, 2: 1.17704, 3: 0.08571, 4: 0.0, 5: 0.0}, 'mad': {0: 0.85303, 1: 0.8571, 2: 1.21233, 3: 0.08571, 4: 0.0, 5: 0.0}, 'ssim': {0: 0.86746, 1: 0.83976, 2: 1.15637, 3: 0.05714, 4: 0.0, 5: 0.0}, 'psnr': {0: 0.82563, 1: 0.78801, 2: 1.3115, 3: 0.08571, 4: 0.0, 5: 0.0}, 'ssim.1': {0: 0.87268, 1: 0.82489, 2: 1.13454, 3: 0.04286, 4: 0.0, 5: 0.0}, 'vif': {0: 0.8527, 1: 0.84423, 2: 1.21413, 3: 0.08571, 4: 0.0, 5: 0.0}, 'dists': {0: 0.87622, 1: 0.87765, 2: 1.11966, 3: 0.07143, 4: 0.0, 5: 0.0}, 'lpips': {0: 0.61888, 1: 0.59191, 2: 1.83419, 3: 0.18571, 4: 0.0, 5: 0.0}, 'ahiq': {0: 0.89362, 1: 0.89898, 2: 1.04497, 3: 0.05714, 4: 0.0, 5: 0.0}, 'topiq_fr': {0: 0.77125, 1: 0.78449, 2: 1.48062, 3: 0.11429, 4: 0.0, 5: 0.0}})

        metric_order = sorted(
            point_estimates,
            key=lambda metric: point_estimates[
                metric
            ].iloc[1],
            reverse=True,
        )

        if CHECKPOINT_PATH.exists():
            state = load_bootstrap_checkpoint(
                CHECKPOINT_PATH,
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
                                        original_mos, 
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
                stats_on_transformed_metrics = self.transform_iqa_with_bs_mos(metric_df_for_bs_mos, bootstrapped_mos_data, visualize, is_original_mos=False)

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
                        CHECKPOINT_PATH,
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

        print('\nComputing p-Level quantile from collected bootstraps for every metric..\n')

        ci_at_percentile = {metric: {stat:((np.round(np.percentile(iter_vals, p_percentile), decimals=3))) for stat, iter_vals in iter_stats.items()} for metric, iter_stats in bootstrap_results.items()}
        #bootstrap_results = {'l2': {'plcc': [np.float64(0.86245), np.float64(0.85137), np.float64(0.86743), np.float64(0.86915), np.float64(0.86094), np.float64(0.85756), np.float64(0.83796), np.float64(0.85577), np.float64(0.85753), np.float64(0.84995), np.float64(0.84582)], 'srocc': [np.float64(0.83273), np.float64(0.82545), np.float64(0.84332), np.float64(0.83679), np.float64(0.83005), np.float64(0.82366), np.float64(0.82364), np.float64(0.82975), np.float64(0.8271), np.float64(0.80992), np.float64(0.8179)], 'or': [np.float64(0.08571), np.float64(0.11429), np.float64(0.11429), np.float64(0.07143), np.float64(0.05714), np.float64(0.1), np.float64(0.07143), np.float64(0.14286), np.float64(0.04286), np.float64(0.07143), np.float64(0.07143)]}, 'mad': {'plcc': [np.float64(0.85303), np.float64(0.84296), np.float64(0.83845), np.float64(0.8395), np.float64(0.84451), np.float64(0.84076), np.float64(0.82629), np.float64(0.84814), np.float64(0.84612), np.float64(0.82016), np.float64(0.82195)], 'srocc': [np.float64(0.8571), np.float64(0.85255), np.float64(0.8579), np.float64(0.85199), np.float64(0.85494), np.float64(0.84104), np.float64(0.83312), np.float64(0.85867), np.float64(0.85978), np.float64(0.82161), np.float64(0.83314)], 'or': [np.float64(0.08571), np.float64(0.12857), np.float64(0.11429), np.float64(0.12857), np.float64(0.07143), np.float64(0.1), np.float64(0.1), np.float64(0.12857), np.float64(0.07143), np.float64(0.07143), np.float64(0.05714)]}, 'ssim': {'plcc': [np.float64(0.86746), np.float64(0.86351), np.float64(0.87499), np.float64(0.87667), np.float64(0.87128), np.float64(0.82997), np.float64(0.81464), np.float64(0.84227), np.float64(0.86873), np.float64(0.82299), np.float64(0.82843)], 'srocc': [np.float64(0.83976), np.float64(0.84893), np.float64(0.85553), np.float64(0.85114), np.float64(0.84629), np.float64(0.8317), np.float64(0.83045), np.float64(0.84391), np.float64(0.84283), np.float64(0.81946), np.float64(0.82728)], 'or': [np.float64(0.05714), np.float64(0.08571), np.float64(0.1), np.float64(0.05714), np.float64(0.05714), np.float64(0.14286), np.float64(0.11429), np.float64(0.11429), np.float64(0.05714), np.float64(0.1), np.float64(0.1)]}, 'psnr': {'plcc': [np.float64(0.82563), np.float64(0.80299), np.float64(0.83303), np.float64(0.82602), np.float64(0.82009), np.float64(0.81434), np.float64(0.81251), np.float64(0.82311), np.float64(0.82842), np.float64(0.79849), np.float64(0.80025)], 'srocc': [np.float64(0.78801), np.float64(0.78069), np.float64(0.81159), np.float64(0.79263), np.float64(0.80408), np.float64(0.79244), np.float64(0.7613), np.float64(0.80795), np.float64(0.7934), np.float64(0.76403), np.float64(0.76912)], 'or': [np.float64(0.08571), np.float64(0.11429), np.float64(0.12857), np.float64(0.1), np.float64(0.07143), np.float64(0.1), np.float64(0.08571), np.float64(0.12857), np.float64(0.07143), np.float64(0.11429), np.float64(0.1)]}, 'ssim.1': {'plcc': [np.float64(0.87268), np.float64(0.87812), np.float64(0.87329), np.float64(0.87947), np.float64(0.87518), np.float64(0.87346), np.float64(0.87047), np.float64(0.86789), np.float64(0.87371), np.float64(0.86998), np.float64(0.86117)], 'srocc': [np.float64(0.82489), np.float64(0.85096), np.float64(0.83361), np.float64(0.84308), np.float64(0.8355), np.float64(0.84179), np.float64(0.83558), np.float64(0.83835), np.float64(0.82466), np.float64(0.82246), np.float64(0.8178)], 'or': [np.float64(0.04286), np.float64(0.05714), np.float64(0.08571), np.float64(0.07143), np.float64(0.04286), np.float64(0.07143), np.float64(0.07143), np.float64(0.11429), np.float64(0.05714), np.float64(0.05714), np.float64(0.05714)]}, 'vif': {'plcc': [np.float64(0.8527), np.float64(0.81735), np.float64(0.82883), np.float64(0.8278), np.float64(0.82871), np.float64(0.83729), np.float64(0.82205), np.float64(0.82606), np.float64(0.84167), np.float64(0.80906), np.float64(0.80678)], 'srocc': [np.float64(0.84423), np.float64(0.81908), np.float64(0.83131), np.float64(0.82144), np.float64(0.83237), np.float64(0.83331), np.float64(0.81545), np.float64(0.82855), np.float64(0.83066), np.float64(0.80957), np.float64(0.80372)], 'or': [np.float64(0.08571), np.float64(0.12857), np.float64(0.1), np.float64(0.08571), np.float64(0.05714), np.float64(0.1), np.float64(0.08571), np.float64(0.11429), np.float64(0.08571), np.float64(0.11429), np.float64(0.11429)]}, 'dists': {'plcc': [np.float64(0.87622), np.float64(0.85814), np.float64(0.87757), np.float64(0.87554), np.float64(0.87651), np.float64(0.87128), np.float64(0.86386), np.float64(0.87234), np.float64(0.8783), np.float64(0.86485), np.float64(0.85798)], 'srocc': [np.float64(0.87765), np.float64(0.86867), np.float64(0.88541), np.float64(0.86977), np.float64(0.88319), np.float64(0.88001), np.float64(0.86059), np.float64(0.87961), np.float64(0.87443), np.float64(0.85975), np.float64(0.86208)], 'or': [np.float64(0.07143), np.float64(0.07143), np.float64(0.11429), np.float64(0.07143), np.float64(0.05714), np.float64(0.07143), np.float64(0.04286), np.float64(0.11429), np.float64(0.07143), np.float64(0.07143), np.float64(0.07143)]}, 'lpips': {'plcc': [np.float64(0.61888), np.float64(0.57186), np.float64(0.62854), np.float64(0.61131), np.float64(0.60668), np.float64(0.59676), np.float64(0.59152), np.float64(0.60636), np.float64(0.59669), np.float64(0.57847), np.float64(0.57968)], 'srocc': [np.float64(0.59191), np.float64(0.53198), np.float64(0.58044), np.float64(0.55505), np.float64(0.56047), np.float64(0.54865), np.float64(0.53599), np.float64(0.56174), np.float64(0.55009), np.float64(0.52005), np.float64(0.52501)], 'or': [np.float64(0.18571), np.float64(0.24286), np.float64(0.22857), np.float64(0.17143), np.float64(0.18571), np.float64(0.2), np.float64(0.21429), np.float64(0.21429), np.float64(0.2), np.float64(0.21429), np.float64(0.21429)]}, 'ahiq': {'plcc': [np.float64(0.89362), np.float64(0.89879), np.float64(0.90088), np.float64(0.89589), np.float64(0.9115), np.float64(0.89861), np.float64(0.89634), np.float64(0.90147), np.float64(0.91056), np.float64(0.89587), np.float64(0.88922)], 'srocc': [np.float64(0.89898), np.float64(0.90306), np.float64(0.9015), np.float64(0.89674), np.float64(0.91072), np.float64(0.89763), np.float64(0.89285), np.float64(0.90185), np.float64(0.90277), np.float64(0.88423), np.float64(0.88938)], 'or': [np.float64(0.05714), np.float64(0.11429), np.float64(0.08571), np.float64(0.08571), np.float64(0.02857), np.float64(0.05714), np.float64(0.02857), np.float64(0.12857), np.float64(0.04286), np.float64(0.04286), np.float64(0.04286)]}, 'topiq_fr': {'plcc': [np.float64(0.77125), np.float64(0.74547), np.float64(0.65439), np.float64(0.6633), np.float64(0.67909), np.float64(0.64533), np.float64(0.6424), np.float64(0.64718), np.float64(0.64823), np.float64(0.65742), np.float64(0.57353)], 'srocc': [np.float64(0.78449), np.float64(0.82692), np.float64(0.80748), np.float64(0.79756), np.float64(0.80351), np.float64(0.80882), np.float64(0.77266), np.float64(0.80214), np.float64(0.77872), np.float64(0.78828), np.float64(0.75419)], 'or': [np.float64(0.11429), np.float64(0.14286), np.float64(0.17143), np.float64(0.14286), np.float64(0.11429), np.float64(0.14286), np.float64(0.1), np.float64(0.17143), np.float64(0.14286), np.float64(0.14286), np.float64(0.1)]}}
        bootstrap_results_df = pd.DataFrame(bootstrap_results)
        #ci_at_percentile = {'l2': {'plcc': np.array([0.8397285, 0.868763 ]), 'srocc': np.array([0.8117155 , 0.84185075]), 'or': np.array([0.046073  , 0.13643175])}, 'mad': {'plcc': np.array([0.82056275, 0.8476855 ]), 'srocc': np.array([0.82419975, 0.85953025]), 'or': np.array([0.06035525, 0.12857   ])}, 'ssim': {'plcc': np.array([0.81651875, 0.876292  ]), 'srocc': np.array([0.8212195 , 0.85454225]), 'or': np.array([0.05714   , 0.13643175])}, 'psnr': {'plcc': np.array([0.798886  , 0.83199275]), 'srocc': np.array([0.76191425, 0.810771  ]), 'or': np.array([0.07143, 0.12857])}, 'ssim.1': {'plcc': np.array([0.862682  , 0.87916625]), 'srocc': np.array([0.8188485, 0.849187 ]), 'or': np.array([0.046073 , 0.1078595])}, 'vif': {'plcc': np.array([0.807293 , 0.8406845]), 'srocc': np.array([0.80503625, 0.8330985 ]), 'or': np.array([0.06356825, 0.125357  ])}, 'dists': {'plcc': np.array([0.858016  , 0.87813575]), 'srocc': np.array([0.859939 , 0.8849105]), 'or': np.array([0.046073, 0.11429 ])}, 'lpips': {'plcc': np.array([0.57334725, 0.62466325]), 'srocc': np.array([0.521166 , 0.5762325]), 'or': np.array([0.174643  , 0.23964475])}, 'ahiq': {'plcc': np.array([0.89071625, 0.9112885 ]), 'srocc': np.array([0.88538875, 0.9089965 ]), 'or': np.array([0.02857 , 0.125357])}, 'topiq_fr': {'plcc': np.array([0.58902575, 0.7305345 ]), 'srocc': np.array([0.75834575, 0.8228475 ]), 'or': np.array([0.1    , 0.17143])}}
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


        print('Computing Delta Intervals from bootstrap iterations for Reference AHIQ and targets')

        self.plot_metric_difference_ci('ahiq', list(bootstrap_results.keys()), bootstrap_results_df, point_estimates, metric_order)

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


    def plot_metric_difference_ci(self, metric, targets, b_results_df, point_estimates, metric_order):

        metric_order = metric_order[1:] # remove AHIQ 
        #metric_order.remove('l2')

        point_estimate_delta_plcc = point_estimates['ahiq'][0] - point_estimates[:].iloc[0]
        point_estimate_delta_srocc = point_estimates['ahiq'][1] - point_estimates[:].iloc[1]
        point_estimate_delta_or = point_estimates[:].iloc[3] - point_estimates['ahiq'][3]
        point_estimate_delta = pd.concat((point_estimate_delta_plcc, point_estimate_delta_srocc, point_estimate_delta_or), axis=1).T
        delta_interval_dict = {}

        for target in targets:
            if target != metric:
                delta_intervals = self.metric_ci_comparison(b_results_df, metric, target)
                delta_interval_dict[target] = delta_intervals
                #delta_interval_dict[target] = {k:(round(point_estimate_delta[target].iloc[n],3), v.round(3)) for (k,v), n in zip(delta_intervals.items(), [0, 1, 2])}

        delta_interval_df = pd.DataFrame(delta_interval_dict)
        delta_interval_df.drop('l2', axis=1)
        point_estimates.drop('l2', axis=1)
        

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
    

    def identity_bootstrap_test(self, ratings_long, original_mos, participant_col, image_col,rating_col):
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


    def metric_ci_comparison(self, bootstrap_results_df, metric, target, p_value=0.05):

        p_percentile = [(p_value*100) / 2, 100 - (p_value*100) / 2]

        metric_array = np.vstack(bootstrap_results_df[metric].to_numpy()).astype(np.float64)
        target_array = np.vstack(bootstrap_results_df[target].to_numpy()).astype(np.float64)

        plcc_delta_array = metric_array[0] - target_array[0]
        srocc_delta_array = metric_array[1] - target_array[1]
        or_delta_array = target_array[2] - metric_array[2]

        plcc_delta_intervals = np.percentile(plcc_delta_array, p_percentile)
        srocc_delta_intervals = np.percentile(srocc_delta_array, p_percentile)
        or_delta_intervals = np.percentile(or_delta_array, p_percentile)

        print(f'\n\tConfidence Intervals at confidence level: {100 - (p_value*100)} for reference {metric.upper()} and target {target.upper()}\n')

        print('\t\tDelta PLCC: ' , plcc_delta_intervals)
        print('\t\tDelta SROCC: ' , srocc_delta_intervals)
        print('\t\tDelta OR: ' , or_delta_intervals)

        return {'plcc': plcc_delta_intervals, 'srocc' : srocc_delta_intervals, 'or': or_delta_intervals}
        

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




    def transform_iqa_with_bs_mos(self, metric_df, mos_df, visualize=False, is_original_mos=False):

        #align MOS,STD and Metrics
        mos_df = mos_df.sort_values(by='file_id')
        metric_df = metric_df.sort_values(by='file')

        # extract MOS and STD
        mos_scores = mos_df['mos'].to_numpy()
        rating_std = mos_df['rating_std'].to_numpy()

        _ = metric_df.pop('file') # remove the file column from the metric df

        metric_df_transformed = metric_df.apply(self.metric_collection.fit_and_transform, args=(mos_scores, visualize, is_original_mos), raw=True, axis=0, result_type='broadcast')
        metrics_df = metric_df_transformed.apply(self.metric_collection, args=(mos_scores, rating_std), axis=0)
        return metrics_df   
    
    # Step 3: Compare MOS to (1) frequency features (radial profiles) and (2) luminance and color features (Y,Cb,CR)


    def log_minmax_l2(self, l2):
        l2 = math.log(l2 + self.eps)
        return (l2 - self.minimum_l2) / (self.maximum_l2 - self.minimum_l2)

    def get_mos(self, file):
        attack, pert_deg = self.file_2_inf[file.split('.')[0]]
        path = f'{self.iqa_root}{pert_deg}/{attack}/iqa_report.csv'
        with open(path, 'r') as f:
            csv_obj = csv.reader(f)
            for line in csv_obj:
                if line[0] == f'./data/survey/images/{str(file)}':
                    return float(line[-1])


    def get_actual_distortion(self, file):
        attack, pert_deg = self.file_2_inf[file.split('.')[0]]
        path = f'{self.iqa_root}{pert_deg}/{attack}/iqa_report.csv'
        with open(path, 'r') as f:
            csv_obj = csv.reader(f)
            for line in csv_obj:
                if line[0] == f'./data/survey/images/{str(file)}':
                    return line[1]

    ## (1) Frequency

    def compute_freq_ci_and_plot(self, point_estimate_plcc, point_estimate_srocc, bootstrap_results_plcc, bootstrap_results_srocc, freq_comps, diffs, p_value=0.05):

        sns.set_theme(
            style="whitegrid",
            context="paper",
            font_scale=FONT_SCALE
        )

        p_percentile = [(p_value*100) / 2, 100 - (p_value*100) / 2]

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



    def compute_color_ci_and_plot(self, point_estimate_plcc,point_estimate_srocc,  bootstrap_results_plcc, bootstrap_results_srocc, y_comps, y_diffs, p_value=0.05):

        sns.set_theme(
            style="whitegrid",
            context="paper",
            font_scale=FONT_SCALE
        )

        p_percentile = [(p_value*100) / 2, 100 - (p_value*100) / 2]

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

    
    def get_freq_correlations(self, metric_df, mos_df, is_original_mos=False):
        # compute mean frequency per image
        # compute difference bewteen MOS result and distortion class

        #align MOS,STD and Metrics
        mos_df = mos_df.sort_values(by='file_id')
        metric_df = metric_df.sort_values(by='file')

        # extract MOS and STD
        mos_scores = mos_df['mos'].to_numpy()
        rating_std = mos_df['rating_std'].to_numpy()

        files_df = metric_df.pop('file') # remove the file column from the metric df

        l2_results = metric_df['l2']
        l2_results = self.metric_collection.fit_and_transform(l2_results, mos_scores) # transformed iqm
        frequency_components_df = files_df.apply(self.get_frequency_components)

        if is_original_mos:
            plcc_result, srocc_result, freq_comps, diffs = self.compute_freq_correlations_lsq(mos_scores, l2_results, frequency_components_df, is_original_mos)
            return plcc_result, srocc_result, freq_comps, diffs 
        else:
            plcc_result, srocc_result = self.compute_freq_correlations_lsq(mos_scores, l2_results, frequency_components_df, is_original_mos)
            return plcc_result, srocc_result

    def get_color_lum_correlations(self, metric_df, mos_df, is_original_mos=False):
        # compute mean frequency per image
        # compute difference bewteen MOS result and distortion class

        #align MOS,STD and Metrics
        mos_df = mos_df.sort_values(by='file_id')
        metric_df = metric_df.sort_values(by='file')

        # extract MOS and STD
        mos_scores = mos_df['mos'].to_numpy()
        rating_std = mos_df['rating_std'].to_numpy()

        files_df = metric_df.pop('file') # remove the file column from the metric df


        l2_results = metric_df['l2']
        l2_results = self.metric_collection.fit_and_transform(l2_results, mos_scores)
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
        path_to_ori = self.iqa_root+ 'images/' + str(file) + '.png'
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
        path_to_ori = self.iqa_root+ 'images/' + str(file) + '.png'
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
    
    def rm_outliers(self, ol_threshold=3.0):
        # compute z-score & remove evals that are 3x sigma away from the mean
        filtered_votes = self.votes.copy()
        for attack, d_dict in self.votes.items():
            for dist, v_dict in d_dict.items():
                votes = self.votes[attack][dist]
                z_scores = self.compute_z(votes)
                remove = [z_score > ol_threshold for z_score in z_scores]
                filtered_votes[attack][dist] = [vote for vote, remove_ind in zip(votes, remove) if not remove_ind]
        self.votes = filtered_votes
    
    def compute_z(self, votes):
        n = len(votes)
        mu = sum(votes) / n
        sigma = math.sqrt(sum([(vote - mu)**2 for vote in votes])/ (n-1))
        z_scores = [abs(((vote - mu) / (sigma + self.eps))) for vote in votes]
        return z_scores
    

    # Metrics

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

class MetricCollection:

    def __init__(self, mos_np, ig_statistics): 
        self.mos_scores = mos_np
        self.ig_statistics = ig_statistics

    def cross_validated_5pl(
        self,
        raw_scores,
        mos_scores,
        n_splits=5,
    ):
        metric_name = raw_scores.name

        raw_scores = np.asarray(raw_scores, dtype=float)
        mos = np.asarray(mos_scores, dtype=float)

        p0 = [1.0, 5.0, np.median(raw_scores), 0.0, np.mean(mos)]

        mapped_predictions = np.full(
            len(raw_scores),
            np.nan,
        )

        folds = KFold(
            n_splits=n_splits,
            shuffle=True,
            random_state=RANDOM_SEED,
        )

        for train_indices, test_indices in folds.split(raw_scores):
            try:
                parameters, _ = curve_fit(
                    MetricCollection.logistic_5pl,
                    raw_scores[train_indices],
                    mos[train_indices],
                    p0,
                    maxfev=1000000,
                    method="trf",
                    x_scale="jac",

                )

                mapped_predictions[test_indices] = MetricCollection.logistic_5pl(raw_scores[test_indices], *parameters)

            except (
                RuntimeError,
                ValueError,
                FloatingPointError,
                np.linalg.LinAlgError,
            ) as exc:
                raise FivePLFitError(
                    f"5PL fit failed for metric={metric_name}. "
                ) from exc

        return mapped_predictions, parameters

    """@staticmethod
    def logistic_5pl(x, a, b, c, d, g):
        return a * (0.5 - (1/(1+np.exp(b*(x - c))))) + d*x + g"""

    @staticmethod
    def logistic_5pl(x, a, b, c, d, g):
        return (
            a * (0.5 - expit(-b * (x - c)))
            + d * x
            + g
        )

    @staticmethod
    def hyperplane(x, slope, intercept):
        return slope * x + intercept
    
    def fit_and_transform(self, iqm_scores, mos_scores, visualize=False, is_original_mos=False):

        sns.set_theme(
            style="whitegrid",
            context="paper",
            font_scale=FONT_SCALE
        )

        print(f"\n\t cross-validation logistic curve fit for metric {iqm_scores.name.upper()}")

        #iqm_scores = self.minmax(iqm_scores)
        #params = MetricCollection.fit_data(iqm_scores, self.mos_scores)
        #transformed_iqm_scores = MetricCollection.logistic_5pl(iqm_scores, *params)
        transformed_iqm_scores, params= self.cross_validated_5pl(raw_scores=iqm_scores, mos_scores=mos_scores, n_splits=5)


        x_fit = np.linspace(min(iqm_scores), max(iqm_scores), 200)
        y_fit = self.logistic_5pl(x_fit, *params)

        plt.figure(figsize=(8, 5))
        plt.scatter(
            iqm_scores, 
            mos_scores,
            marker="o",
            s=48,
            facecolors="white",
            edgecolors="#0072B2",
            linewidths=1.3,
            label="IQA vs. MOS",
            zorder=3)
        plt.plot(x_fit, y_fit, color='grey', label='Fitted 5PL Curve', linewidth=2)
        plt.xlabel(f'{iqm_scores.name.upper()} score')
        plt.ylabel('MOS')
        plt.title(f'5-Parameter Logistic Fit to {(iqm_scores.name.upper())} vs. MOS', fontweight='bold', fontsize=TITLE_SIZE)
        plt.legend()
        plt.grid(True)
        plt.tight_layout()

        if visualize:
            plt.show()
        if is_original_mos:
            plt.savefig(f'./visualizations/{iqm_scores.name.upper()}.png')

        return transformed_iqm_scores
    
    def minmax(self, iqm_scores):
        maxi = iqm_scores.max()
        mini = iqm_scores.min()
        return (iqm_scores - mini) / (maxi - mini)

    @staticmethod
    def fit_data(iqm_scores, mos_scores):

        p0 = [1.0, 5.0, np.median(iqm_scores), 0.0, np.mean(mos_scores)]

        params, _ = curve_fit(MetricCollection.logistic_5pl, iqm_scores, mos_scores, p0, maxfev=1000000)
        return params

    @staticmethod
    def fit_data_lsq(x, y):

        x = np.asarray(x)
        y = np.asarray(y)
        n = x.size

        Sx = x.sum()
        Sy = y.sum()
        Sxy = (x * y).sum()
        Sxx = (x * x).sum()  # Σ x^2

        # numerator and denominator with correct parentheses
        numerator = n * Sxy - Sx * Sy
        denominator = n * Sxx - Sx * Sx

        if denominator == 0:
            raise ValueError("Denominator is zero — all x values are identical.")
        slope = numerator / denominator
        intercept = (Sy - slope * Sx) / n
        return slope, intercept

    def __call__(self, iqm_scores, mos_scores, rating_std, transform_iqm=False):
        
        # map by logistic curve
        if transform_iqm:
            iqm_scores = MetricCollection.fit_and_transform(iqm_scores, mos_scores)

        # pearson
        r_pearson, p_value_pearson = pearsonr(iqm_scores, mos_scores)
    
        #rsme
        r_rsme = MetricCollection.rmse(iqm_scores, mos_scores)

        # spearman rank order correlation
        r_spearman, p_value_spearman = spearmanr(iqm_scores, mos_scores)

        # outlier ratio
        r_or = MetricCollection.outlier_ratio(iqm_scores, mos_scores, rating_std=rating_std)


        results =  [r_pearson, r_spearman, r_rsme, r_or, p_value_pearson, p_value_spearman]
        return [round(i, 5) for i in results]

    @staticmethod
    def rmse(iqm_scores, mos_scores):
        return np.sqrt(((iqm_scores - mos_scores)**2).sum())


    @staticmethod
    def outlier_ratio(iqm_scores, mos, rating_std):
        iqm_scores = np.asarray(
            iqm_scores, dtype=float
        ).reshape(-1)

        mos = np.asarray(
            mos, dtype=float
        ).reshape(-1)

        rating_std = np.asarray(
            rating_std, dtype=float
        ).reshape(-1)

        if not (
            iqm_scores.shape
            == mos.shape
            == rating_std.shape
        ):
            raise ValueError(
                "Expected one predicted MOS, MOS, and rating "
                "standard deviation per image. Received "
                f"{iqm_scores.shape}, {mos.shape}, "
                f"{rating_std.shape}."
            )

        absolute_error = np.abs(iqm_scores - mos)

        is_outlier = absolute_error > 2.0 * rating_std

        return is_outlier.mean()

class CIAnalysis:

    def __init__(self, ratings_dict, l2_distortions, file_dict, c_level=0.95):
        self.ratings_dict = ratings_dict
        self.c_level = c_level
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
        file, id, means, ci_lower, ci_upper = [],[],[],[], []
        for attack_name, v_dict in self.ci_dict.items():
            for delta, intervals in v_dict.items():
                #id.append(f'{attack_name} : {round(self.l2_distortions[attack_name][delta], 2)}')
                id.append(round(self.l2_distortions[attack_name][delta], 2))
                file.append(f'{self.file_dict[attack_name][delta]}.png')
                mean, lower, upper = intervals
                means.append(mean)
                ci_lower.append(lower)
                ci_upper.append(upper)
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


    def extend_results_with_intervals(self, dp_obj, table_path='./data/perc_eval/mos_results.csv'):
        target_path = './data/perc_eval/mos_results_ci.csv'
        with open(table_path, 'r') as source_f:
            with open(target_path, 'w') as target_f:
                source_csv = csv.reader(source_f)
                target_csv = csv.writer(target_f)
                initial_line = next(source_csv)
                initial_line.append('ci_lower')
                initial_line.append('ci_upper')
                target_csv.writerow(initial_line)
                for line in source_csv:
                    file = int(line[0].split('/')[-1])
                    attack, pert = dp_obj.file_2_info[file]
                    mean, ci_lower, ci_upper = self.ci_dict[attack][pert]
                    line.append(ci_lower)
                    line.append(ci_upper)
                    target_csv.writerow(line)

            
    
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


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='data.csv', help='set the path to data csv')
    args = parser.parse_args()

    data_processor = DataProcessor()

    # process survey results and compute MOS
    data_processor.process_data(args.data_path)
    data_processor.compute_mos_per_sample()
    #data_processor.rm_outliers()

    data_processor.survey_statistics()

    
    
    ratings_dict = data_processor.votes
    l2_distortions = data_processor.l2_dict
    file_dict = data_processor.f_dict

    ci_analysis = CIAnalysis(ratings_dict, l2_distortions, file_dict)
    ci_analysis.plot(visualize=False)
    ci_analysis.extend_results_with_intervals(dp_obj=data_processor)

    print("\n====================================\n")

    correlation_ci = data_processor.bootstrap_iqa_comparison()

    print('done')


