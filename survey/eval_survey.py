import argparse
import csv
import torch
import matplotlib.pyplot as plt
import os
import math
import torchvision.transforms as T
import numpy as np
import pandas as pd

from scipy.optimize import curve_fit
from scipy.stats import pearsonr, spearmanr, t
from process import DatasetConstructor
from PIL import Image

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
                      '23.0':73,
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

        self.create_file_csv()
        self.eps = 1e-05
        self.maximum_l2 = math.log(38.0 + self.eps)
        self.minimum_l2 = math.log(0 + self.eps)
    
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
        with open(data_path, 'r') as f:
            csv_obj = csv.reader(f)
            fields = next(csv_obj)
            results_fields = fields[35:49] + fields[90:104]
            pertmag_fields = fields[49:63] + fields[104:118]
            for line in csv_obj:
                is_desktop = True if int(line[63]) == 1 else False
                if is_desktop:
                    self.n += 1
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


                        """for e, (answer, pert_mag_file) in enumerate(zip(results, pert_mags_files)):
                            pert_mag, file, attack = pert_mag_file.split(',')
                            if all([x != 'undefined' for x in pert_mag_file.split(',')]):
                                self.votes[attack][pert_mag.split('.')[0] + '.0'].append(self.answer_to_value[answer]) """
    
    def field_map(self, eval_field):
        #maps an eval field (holding the human eval) to the appropriate info field (holding attack, pert and file info)
        pass
    
    def compute_mos_per_sample(self):
        # TODO: change to MOS per image
        self.rm_outliers()
        file_2_inf_keys = self.file_2_inf.keys()
        with open(self.results_csv, 'a') as n_f:
            csv_obj_new = csv.writer(n_f)
            for file in range(280):
                if str(file) in file_2_inf_keys:
                    attack, pert_deg = self.file_2_inf[str(file)]
                    path = f'{self.iqa_root}{pert_deg}/{attack}/iqa_report.csv'
                    with open(path, 'r+') as f:
                        csv_obj = csv.reader(f)
                        for line in csv_obj:
                            if line[0] == f'./data/survey/images/{file}':
                                mos = self.mos(self.votes[attack][pert_deg])
                                if mos == -1:
                                    mos = self.l2_to_mos(float(line[1]))
                                line.append(mos)
                                csv_obj_new.writerow(line)
                                break
        print('- MOS computed...')
        self.results_df = pd.read_csv(self.results_csv)
        mos_np = self.results_df.pop('MOS').to_numpy()
        self.metric_collection = MetricCollection(mos_np)
        self.files_df = self.results_df.pop('file')

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

    def compare_to_iqa(self):
        results_df_transformed = self.results_df.apply(self.metric_collection.fit_and_transform, raw=True, axis=0, result_type='broadcast')
        metrics_df = results_df_transformed.apply(self.metric_collection, axis=0)
        return metrics_df
                    
    def get_iqa_for_file(self, file, pert_mag, attack):
        iqa_path = self.iqa_root + f'{pert_mag}/{attack}/iqa_report.csv'
        with open(iqa_path, 'r') as f:
            csv_obj = csv.reader(f)
            for line in csv_obj:
                if line[0].endswith(str(file)):
                    return [float(i) for i in line[1:]]
                
    
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
    
    def get_freq_correlations(self):
        # compute mean frequency per image
        # compute difference bewteen MOS result and distortion class

        l2_results = self.results_df['l2']
        l2_results = self.metric_collection.fit_and_transform(l2_results)
        frequency_components_df = self.files_df.apply(self.get_frequency_components)
        r, p_value = self.compute_freq_correlations(l2_results, frequency_components_df)
        return r, p_value
    
    def compute_freq_correlations(self, l2_results, frequency_components_df):
        frequency_components = frequency_components_df.to_numpy()
        differences = l2_results.to_numpy() - self.metric_collection.mos_scores
        pearson_r, p_value = pearsonr(frequency_components, differences)

        # plot freq correlations
        # x - frequency component y - difference
        # hopefully, as the mean freq increases, so will teh gap between l2 and mos

        zipped = zip(frequency_components, differences)
        # sort by mean frequency
        sorted_zipped = sorted(zipped, key=lambda x: x[0])

        freq_comps = [x[0] for x in sorted_zipped]
        diffs = [x[1] for x in sorted_zipped]

        #create curve
        params = MetricCollection.fit_data(freq_comps, diffs)
        x_fit = np.linspace(min(freq_comps), max(freq_comps), 200)
        y_fit = MetricCollection.logistic_5pl(x_fit, *params)


        plt.scatter(freq_comps, diffs)
        plt.plot(x_fit, y_fit, color='red', label='Fitted 5PL Curve', linewidth=2)
        plt.title("Mean Frequency vs Difference between actual and perceived Distortion")
        plt.xlabel("Mean Frequency")
        plt.ylabel("Delta")
        plt.grid(True)
        plt.show()


        return pearson_r, p_value

    def get_frequency_components(self, file):
        path_to_ori = self.iqa_root+ 'images/' + file.split('/')[-1] + '.png'
        img = Image.open(path_to_ori)
        img = self.to_tensor(img)
        rad_profile = self.dataset_contructor.compute_rad_profile(img)
        denom = rad_profile.sum()
        weighter = torch.arange(0,len(rad_profile))
        counter = rad_profile * weighter
        counter = counter.mean()
        return (counter / denom).item()
    
    ## (2) Luminance and color features

    def compute_clr_lum_correlations(self, l2_results, clr_components_df):
        clr_components = clr_components_df.to_list()

        y_components = np.array([x[0] for x in clr_components])
        cb_components = np.array([x[1] for x in clr_components])
        cr_components = np.array([x[2] for x in clr_components])

        differences = l2_results.to_numpy() - self.metric_collection.mos_scores

        r_y, p_value_y = pearsonr(y_components, differences)
        r_cb, p_value_cb = pearsonr(cb_components, differences)
        r_cr, p_value_cr = pearsonr(cr_components, differences)

        y_zipped = zip(y_components, differences)
        cb_zipped = zip(cb_components, differences)
        cr_zipped = zip(cr_components, differences)

        # sort by mean color
        y_sorted_zipped = sorted(y_zipped, key=lambda x: x[0])
        cb_sorted_zipped = sorted(cb_zipped, key=lambda x: x[0])
        cr_sorted_zipped = sorted(cr_zipped, key=lambda x: x[0])

        y_comps = [x[0] for x in y_sorted_zipped]
        y_diffs = [x[1] for x in y_sorted_zipped]

        cb_comps = [x[0] for x in cb_sorted_zipped]
        cb_diffs = [x[1] for x in cb_sorted_zipped]

        cr_comps = [x[0] for x in cr_sorted_zipped]
        cr_diffs = [x[1] for x in cr_sorted_zipped]


        params = MetricCollection.fit_data(y_comps, y_diffs)
        x_fit = np.linspace(min(y_comps), max(y_comps), 200)
        y_fit = MetricCollection.logistic_5pl(x_fit, *params)

        plt.scatter(y_comps, y_diffs)
        plt.plot(x_fit, y_fit, color='red', label='Fitted 5PL Curve', linewidth=2)
        plt.title("Mean Luminance vs Difference between actual and perceived Distortion")
        plt.xlabel("Mean Luminance")
        plt.ylabel("Delta")
        plt.grid(True)
        plt.show()

        plt.scatter(cb_comps, cb_diffs)
        plt.title("Mean Blue Crominance vs Difference between actual and perceived Distortion")
        plt.xlabel("Mean Blue Crominance")
        plt.ylabel("Delta")
        plt.grid(True)
        plt.show()

        plt.scatter(cr_comps, cr_diffs)
        plt.title("Mean Red Crominance vs Difference between actual and perceived Distortion")
        plt.xlabel("Mean Red Crominance")
        plt.ylabel("Delta")
        plt.grid(True)
        plt.show()

        return [(r_y, p_value_y),  (r_cb, p_value_cb) ,(r_cr, p_value_cr)]
    
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
    
    """def rm_outliers(self, votes, ol_threshold=3.0):
        z_scores = self.compute_z(votes)
        remove = [z_score > ol_threshold for z_score in z_scores]
        filtered = [vote for vote, remove_ind in zip(votes, remove) if not remove_ind]
        return filtered"""


    
    def compute_z(self, votes):
        n = len(votes)
        mu = sum(votes) / n
        sigma = math.sqrt(sum([(vote - mu)**2 for vote in votes])/ (n-1))
        z_scores = [abs(((vote - mu) / (sigma + self.eps))) for vote in votes]
        return z_scores
    

    def get_color_lum_correlations(self):
        # compute mean frequency per image
        # compute difference bewteen MOS result and distortion class
        l2_results = self.results_df['l2']
        l2_results = self.metric_collection.fit_and_transform(l2_results)
        clr_components_df = self.files_df.apply(self.get_color_lum_components)
        clr_results = self.compute_clr_lum_correlations(l2_results, clr_components_df)

        return clr_results
    
    def get_color_lum_components(self, file):
        path_to_ori = self.iqa_root+ 'images/' + file.split('/')[-1] + '.png'
        img = Image.open(path_to_ori)
        img = img.convert('YCbCr')
        img = self.to_tensor(img)
        color_lum = img.mean(dim=(1,2))
        return color_lum.tolist()
    

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

    def __init__(self, mos_np): 
        self.mos_scores = mos_np
    
    @staticmethod
    def logistic_5pl(x, a, b, c, d, g):
        return a * (0.5 - (1/(1+np.exp(b*(x - c))))) + d*x + g

    
    def fit_and_transform(self, iqm_scores):

        #iqm_scores = self.minmax(iqm_scores)
        params = MetricCollection.fit_data(iqm_scores, self.mos_scores)
        transformed_iqm_scores = MetricCollection.logistic_5pl(iqm_scores, *params)

        x_fit = np.linspace(min(iqm_scores), max(iqm_scores), 200)
        y_fit = self.logistic_5pl(x_fit, *params)

        plt.figure(figsize=(8, 5))
        plt.scatter(iqm_scores, self.mos_scores, color='blue', label='Raw Data (IQM vs. MOS)', alpha=0.7)
        plt.plot(x_fit, y_fit, color='red', label='Fitted 5PL Curve', linewidth=2)
        plt.xlabel(f'{iqm_scores.name} score')
        plt.ylabel('MOS')
        plt.title(f'5-Parameter Logistic Fit to {iqm_scores.name} vs. MOS')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()
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

    def __call__(self, iqm_scores, transform_iqm=False):
        
        # map by logistic curve
        if transform_iqm:
            iqm_scores = MetricCollection.fit_and_transform(iqm_scores, self.mos_scores)

        # pearson
        r_pearson, p_value_pearson = pearsonr(iqm_scores, self.mos_scores)
    
        #rsme
        r_rsme = MetricCollection.rmse(iqm_scores, self.mos_scores)

        # spearman rank order correlation
        r_spearman, p_value_spearman = spearmanr(iqm_scores, self.mos_scores)

        # outlier ratio
        r_or = MetricCollection.outlier_ratio(iqm_scores, self.mos_scores)


        results =  [r_pearson, r_spearman, r_rsme, r_or, p_value_pearson, p_value_spearman]
        return [round(i, 5) for i in results]

    @staticmethod
    def rmse(iqm_scores, mos_scores):
        return np.sqrt(((iqm_scores - mos_scores)**2).sum())
    
    @staticmethod
    def outlier_ratio(iqm_scores, mos_scores):
        mos_std_dev = mos_scores.std()
        difference = np.absolute(iqm_scores - mos_scores)
        outliers = difference[difference > 2*mos_std_dev]
        return len(outliers) / len(mos_scores)

class CIAnalysis:

    def __init__(self, ratings_dict, c_level=0.95):
        self.ratings_dict = ratings_dict
        self.c_level = c_level
        self.ci_dict = self.process_ratings()
    
    def get_ci_tuple(self, ratings):
        ci_entry = ConfidenceInterval(ratings, c_level=self.c_level)
        lower, upper = ci_entry.ci()
        return ci_entry.sample_mu, lower, upper
    
    def process_ratings(self):
        print(f'- computing confidence intervals for {self.c_level} confidence...')
        ci_dict = self.ratings_dict.copy()
        for attack_name, v_dict in self.ratings_dict.items():
            for delta, ratings in v_dict.items():
                ci_dict[attack_name][delta] = self.get_ci_tuple(ratings)
        return ci_dict
    
    def transform(self):
        id, means, ci_lower, ci_upper = [],[],[], []
        for attack_name, v_dict in self.ci_dict.items():
            for delta, intervals in v_dict.items():
                id.append(f'{attack_name} : {delta}')
                mean, lower, upper = intervals
                means.append(mean)
                ci_lower.append(lower)
                ci_upper.append(upper)
        return id, means, ci_lower, ci_upper

    def plot(self):
        id, means, ci_lower, ci_upper = self.transform()
        plt.figure(figsize=(14,6))
        plt.bar(id, means, yerr=[np.array(means)-np.array(ci_lower), np.array(ci_upper)-np.array(means)], 
                capsize=4, color='skyblue')
        plt.xlabel("Image Index")
        plt.xticks(rotation=90)
        plt.ylabel("Mean Opinion Score (DSIS)")
        plt.title("Perceptual Evaluation of Adversarial Samples (95% CI)")
        plt.tight_layout()
        plt.show()
    
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
    
    ratings_dict = data_processor.votes
    ci_analysis = CIAnalysis(ratings_dict)
    ci_analysis.plot()

    # get agreement of MOS and IQA in terms of pearson, spearman, rmse, od as a dataframe
    metric_df = data_processor.compare_to_iqa()
    print(metric_df)

    # get frequency and MOS correlations as pearson and as a graph

    pearson_freq = data_processor.get_freq_correlations()
    print(pearson_freq)


    # get luminance and MOS correlations as pearson and as a graph
    # get blue crominance and MOS correlations as pearson and as a graph
    # get red crominance and MOS correlations as pearson and as a graph

    pearson_clr = data_processor.get_color_lum_correlations()
    print(pearson_clr)

    print('done')


