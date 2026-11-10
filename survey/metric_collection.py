
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
import seaborn as sns

from scipy.optimize import curve_fit
from scipy.stats import pearsonr, spearmanr

from sklearn.model_selection import KFold, GroupKFold, RepeatedKFold
from scipy.special import expit
from pathlib import Path

from utils import FivePLFitError


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



class MetricCollection:

    def __init__(self, image_stats): 
        self.mos_scores = image_stats['mos'].to_numpy()
        self.ig_statistics = image_stats

    def cross_validated_5pl(
        self,
        raw_scores,
        mos_scores,
        n_splits=5,
        folds=None,
    ):
        metric_name = raw_scores.name

        raw_scores = np.asarray(raw_scores, dtype=float)
        mos = np.asarray(mos_scores, dtype=float)


        p0 = [1.0, 5.0, np.median(raw_scores), 0.0, np.mean(mos)]

        mapped_predictions = np.full(
            len(raw_scores),
            np.nan,
        )

        if folds == None:
            folds = KFold(
                n_splits=n_splits,
                shuffle=True,
                random_state=RANDOM_SEED,
            )

            fold_indices = folds.split(raw_scores)

        else:

            fold_indices = folds

        for train_indices, test_indices in fold_indices:

            transformed_test_scores, params = self.fit_and_get_params_for_metric(train_indices, test_indices, raw_scores, mos, metric_name, p0)
            mapped_predictions[test_indices] = transformed_test_scores

        return mapped_predictions, params

    def fit_and_get_params_for_metric(self, train_indices, test_indices, raw_scores, mos, metric_name, p0):

        train, test, _ = MetricCollection.normalize_scores(raw_scores, train_indices, test_indices, metric_name)
        train_target, test_target = mos[train_indices], mos[test_indices]

        try:
            parameters, _ = curve_fit(
                MetricCollection.logistic_5pl,
                train,
                train_target,
                p0,
                maxfev=1000000,
                method="trf",
                x_scale="jac",

            )

            transformed_test_scores = MetricCollection.logistic_5pl(test, *parameters)

        except (
            RuntimeError,
            ValueError,
            FloatingPointError,
            np.linalg.LinAlgError,
        ) as exc:
            raise FivePLFitError(
                f"5PL fit failed for metric={metric_name}. "
            ) from exc

        return transformed_test_scores, parameters

    @staticmethod
    def logistic_5pl(x, a, b, c, d, g):
        return (
            a * (0.5 - expit(-b * (x - c)))
            + d * x
            + g
        )

    @staticmethod
    def normalize_scores(raw_scores, train_idx, test_idx, metric):

        constant = False

        x_train_raw = raw_scores[train_idx]
        x_test_raw = raw_scores[test_idx]

        x_min = x_train_raw.min()
        x_max = x_train_raw.max()
        x_span = x_max - x_min

        if x_span <= 1e-12:
            constant = True

        x_train = (x_train_raw - x_min) / x_span
        x_test = (x_test_raw - x_min) / x_span

        return x_train, x_test, constant
    
    @staticmethod
    def hyperplane(x, slope, intercept):
        return slope * x + intercept
    
    def fit_and_transform(self, iqm_scores, mos_scores, visualize=False, is_original_mos=False, folds=None):

        sns.set_theme(
            style="whitegrid",
            context="paper",
            font_scale=FONT_SCALE
        )

        print(f"\n\t cross-validation logistic curve fit for metric {iqm_scores.name.upper()}")

        transformed_iqm_scores, params= self.cross_validated_5pl(raw_scores=iqm_scores, mos_scores=mos_scores, n_splits=5, folds=folds)

        x_fit = np.linspace(min(iqm_scores), max(iqm_scores), 200)
        y_fit = self.logistic_5pl(x_fit, *params)

        if visualize or is_original_mos:
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

    """@staticmethod
    def fit_data(iqm_scores, mos_scores):

        p0 = [1.0, 5.0, np.median(iqm_scores), 0.0, np.mean(mos_scores)]

        params, _ = curve_fit(MetricCollection.logistic_5pl, iqm_scores, mos_scores, p0, maxfev=1000000)
        return params

    @staticmethod
    def fit_data_with_custom_p0(iqm_scores, mos_scores, p0):
        params, _ = curve_fit(MetricCollection.logistic_5pl, iqm_scores, mos_scores, p0, maxfev=1000000)
        return params"""

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

    def __call__(self, raw_scores, mos_scores, rating_std, visualize=False, is_original_mos=False, folds=None):
        
        # map by logistic curve

        metric_name = raw_scores.name
        transformed_scores = self.fit_and_transform(raw_scores, mos_scores, visualize=visualize, is_original_mos=is_original_mos, folds=folds)

        # pearson
        r_pearson = pearsonr(transformed_scores, mos_scores).statistic
    
        #rsme
        r_rsme = MetricCollection.rmse(transformed_scores, mos_scores)

        # spearman rank order correlation
        r_spearman = MetricCollection.direction_corrected_srocc(raw_scores, mos_scores, metric_name)

        # outlier ratio
        r_or = MetricCollection.outlier_ratio(transformed_scores, mos_scores, rating_std=rating_std)

        results =  [r_pearson, r_spearman, r_rsme, r_or]
        return [round(i, 5) for i in results]


    @staticmethod
    def direction_corrected_srocc(iqm_scores, mos_scores, metric_name):
        aligned_score = (
            iqm_scores
            * METRIC_DIRECTION[metric_name]
        )

        rho = spearmanr(
            aligned_score,
            mos_scores,
        ).statistic

        return rho

    @staticmethod
    def rmse(iqm_scores, mos_scores):
        return np.sqrt(((iqm_scores - mos_scores)**2).mean())


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

    def sample_5pl_initialization(self, rng, lower_bounds, upper_bounds):

        a_inf, b_inf, c_inf, d_inf, g_inf = lower_bounds
        a_sup, b_sup, c_sup, d_sup, g_sup = upper_bounds
                

        return np.array([
            rng.uniform(a_inf, a_sup),  # a

            # Log-uniform because plausible steepness spans
            # more than one order of magnitude.
            np.exp(
                rng.uniform(
                    np.log(b_inf),
                    np.log(b_sup),
                )
            ),

            rng.uniform(c_inf, c_sup),  # c
            rng.uniform(d_inf, d_sup),  # d
            rng.uniform(g_inf, g_sup),  # g
        ])

    def fitting_stability_per_metric(self, raw_scores, train_idx, test_idx, mos_scores, p0):
        metric_name = raw_scores.name

        raw_scores = np.asarray(raw_scores, dtype=float)
        mos = np.asarray(mos_scores, dtype=float)

        mapped_predictions = np.full(
            len(raw_scores),
            np.nan,
        )

    def quantify_fitting_stability(
        self,
        raw_scores_df,
        rng,
        n_splits=5,
        n_repeats=100,
        n_starts=10,
    ):
        # ---------------------------------------------------------
        # Configuration
        # ---------------------------------------------------------
        metadata_columns = [
            "file_id",
            "mos",
            "rating_std",
            "n_ratings",
        ]

        metric_columns = [
            column
            for column in raw_scores_df.columns
            if column not in metadata_columns
        ]

        mos = raw_scores_df["mos"].to_numpy(dtype=float)
        rating_std = raw_scores_df["rating_std"].to_numpy(dtype=float)

        default_start = np.array([
            1.0,  # a
            5.0,  # b
            0.5,  # c
            0.0,  # d
            0.5,  # g
        ])

        lower_bounds_5pl = np.array([
            0.0,    # a
            0.1,    # b
            -0.25,   # c
            0.0,    # d
            -0.5,    # g
        ])

        upper_bounds_5pl = np.array([
            2.0,   # a
            50.0,   # b
            1.25,  # c
            2.0,   # d
            1.5,   # g
        ])

        # One seed for the complete repeated-CV design.
        split_seed = int(
            rng.integers(
                0,
                np.iinfo(np.int32).max,
            )
        )

        repeated_cv = RepeatedKFold(
            n_splits=n_splits,
            n_repeats=n_repeats,
            random_state=split_seed,
        )

        all_splits = list(
            repeated_cv.split(
                np.arange(len(raw_scores_df))
            )
        )

        assert len(all_splits) == n_splits * n_repeats

        splits_by_repetition = [
            all_splits[
                repetition * n_splits:
                (repetition + 1) * n_splits
            ]
            for repetition in range(n_repeats)
        ]

        fold_records = []
        prediction_records = []
        repetition_records = []

        # ---------------------------------------------------------
        # Repeated cross-validation
        # ---------------------------------------------------------
        for repetition, repetition_folds in enumerate(splits_by_repetition):
            print(
                f"Performing repeated CV "
                f"{repetition + 1}/{n_repeats}"
            )

            # One complete OOF vector per metric and repetition.
            repetition_predictions = {
                metric: np.full(
                    len(raw_scores_df),
                    np.nan,
                )
                for metric in metric_columns
            }

            for fold_nr, (train_idx, test_idx) in enumerate(repetition_folds):
                y_train = mos[train_idx]
                y_test = mos[test_idx]

                for metric in metric_columns:
                    x_raw = raw_scores_df[
                        metric
                    ].to_numpy(dtype=float)

                    # Align all metrics so that larger values mean
                    # greater degradation.
                    direction = METRIC_DIRECTION[metric]
                    x_raw = direction * x_raw

                    # ---------------------------------------------
                    # Fold-specific normalization
                    # ---------------------------------------------

                    x_train, x_test, constant = MetricCollection.normalize_scores(x_raw, train_idx, test_idx, metric)

                    if constant:
                        fold_records.append({
                            "repetition": repetition,
                            "fold": fold_nr,
                            "metric": metric,
                            "converged": False,
                            "failure_reason": (
                                "constant training scores"
                            ),
                        })
                        continue

                    # ---------------------------------------------
                    # Multi-start fitting
                    # ---------------------------------------------
                    initializations = [
                        default_start.copy()
                    ]

                    for _ in range(n_starts - 1):
                        initializations.append(
                            self.sample_5pl_initialization(
                                rng,
                                lower_bounds_5pl,
                                upper_bounds_5pl,
                            )
                        )

                    best_fit = None
                    successful_starts = 0

                    for start_id, p0 in enumerate(initializations):
                        try:
                            params, covariance = curve_fit(
                                self.logistic_5pl,
                                x_train,
                                y_train,
                                p0=p0,
                                bounds=(
                                    lower_bounds_5pl,
                                    upper_bounds_5pl,
                                ),
                                maxfev=1000000,
                            )

                            train_predictions = (
                                self.logistic_5pl(
                                    x_train,
                                    *params,
                                )
                            )

                            if not np.all(np.isfinite(train_predictions)):
                                continue

                            training_sse = np.sum((train_predictions - y_train) ** 2)

                            successful_starts += 1

                            if (
                                best_fit is None
                                or training_sse
                                < best_fit["training_sse"]
                            ):
                                best_fit = {
                                    "params": params,
                                    "covariance": covariance,
                                    "training_sse": (
                                        training_sse
                                    ),
                                    "start_id": start_id,
                                }

                        except (
                            RuntimeError,
                            ValueError,
                            FloatingPointError,
                        ):
                            continue

                    # ---------------------------------------------
                    # All starts failed
                    # ---------------------------------------------
                    if best_fit is None:
                        fold_records.append({
                            "repetition": repetition,
                            "fold": fold_nr,
                            "metric": metric,
                            "converged": False,
                            "failure_reason": (
                                "all initializations failed"
                            ),
                            "successful_starts": 0,
                        })
                        continue

                    params = best_fit["params"]

                    test_predictions = self.logistic_5pl(
                        x_test,
                        *params,
                    )

                    repetition_predictions[
                        metric
                    ][test_idx] = test_predictions

                    # ---------------------------------------------
                    # Diagnostics
                    # ---------------------------------------------
                    parameter_tolerance = (
                        1e-5
                        * (
                            upper_bounds_5pl
                            - lower_bounds_5pl
                        )
                    )

                    at_lower_bound = np.isclose(
                        params,
                        lower_bounds_5pl,
                        atol=parameter_tolerance,
                        rtol=0.0,
                    )

                    at_upper_bound = np.isclose(
                        params,
                        upper_bounds_5pl,
                        atol=parameter_tolerance,
                        rtol=0.0,
                    )

                    parameter_at_bound = np.any(
                        at_lower_bound
                        | at_upper_bound
                    )

                    x_grid = np.linspace(
                        x_train.min(),
                        x_train.max(),
                        1000,
                    )

                    a, b, c, d, g = params

                    sigmoid_values = expit(
                        b * (x_grid - c)
                    )

                    derivative = (
                        a
                        * b
                        * sigmoid_values
                        * (1.0 - sigmoid_values)
                        + d
                    )

                    minimum_derivative = derivative.min()

                    monotonic = (
                        minimum_derivative >= -1e-8
                    )

                    covariance_condition = np.nan

                    if np.all(np.isfinite(
                        best_fit["covariance"]
                    )):
                        covariance_condition = (
                            np.linalg.cond(
                                best_fit["covariance"]
                            )
                        )

                    fold_records.append({
                        "repetition": repetition,
                        "fold": fold_nr,
                        "metric": metric,

                        "converged": True,
                        "failure_reason": None,

                        "successful_starts": (
                            successful_starts
                        ),
                        "selected_start": (
                            best_fit["start_id"]
                        ),

                        "a": a,
                        "b": b,
                        "c": c,
                        "d": d,
                        "g": g,

                        "training_sse": (
                            best_fit["training_sse"]
                        ),
                        "test_rmse": MetricCollection.rmse(test_predictions, y_test),

                        "parameter_at_bound": (
                            parameter_at_bound
                        ),
                        "monotonic": monotonic,
                        "minimum_derivative": (
                            minimum_derivative
                        ),

                        "prediction_min": (
                            test_predictions.min()
                        ),
                        "prediction_max": (
                            test_predictions.max()
                        ),
                        "prediction_outside_mos": (
                            np.any(
                                (test_predictions < 0)
                                | (test_predictions > 1)
                            )
                        ),

                        "covariance_condition": (
                            covariance_condition
                        ),

                        # More interpretable derived quantities.
                        "maximum_logistic_slope": (
                            a * b / 4.0
                        ),
                    })

                    for local_position, sample_idx in enumerate(
                        test_idx
                    ):
                        prediction_records.append({
                            "repetition": repetition,
                            "fold": fold_nr,
                            "metric": metric,
                            "sample_index": sample_idx,
                            "file_id": raw_scores_df.iloc[
                                sample_idx
                            ]["file_id"],
                            "mos": mos[sample_idx],
                            "prediction": (
                                test_predictions[
                                    local_position
                                ]
                            ),
                        })

            # -----------------------------------------------------
            # Repetition-level statistics
            # -----------------------------------------------------
            for metric in metric_columns:
                predictions = repetition_predictions[
                    metric
                ]

                successful = np.all(
                    np.isfinite(predictions)
                )

                if not successful:
                    repetition_records.append({
                        "repetition": repetition,
                        "metric": metric,
                        "successful": False,
                        "plcc": np.nan,
                        "rmse": np.nan,
                        "or": np.nan,
                    })
                    continue

                repetition_records.append({
                    "repetition": repetition,
                    "metric": metric,
                    "successful": True,

                    "plcc": pearsonr(
                        predictions,
                        mos,
                    ).statistic,

                    "rmse": MetricCollection.rmse(predictions, mos),

                    "or": MetricCollection.outlier_ratio(
                        predictions,
                        mos,
                        rating_std=rating_std
                    ),
                })

        # ---------------------------------------------------------
        # Summarize
        # ---------------------------------------------------------
        fold_results = pd.DataFrame(
            fold_records
        )

        prediction_results = pd.DataFrame(
            prediction_records
        )

        repetition_results = pd.DataFrame(
            repetition_records
        )

        # Rank only successful fits.
        repetition_results["plcc_rank"] = (
            repetition_results
            .groupby("repetition")["plcc"]
            .rank(
                ascending=False,
                method="min",
            )
        )

        repetition_results["or_rank"] = (
            repetition_results
            .groupby("repetition")["or"]
            .rank(
                ascending=True,
                method="min",
            )
        )

        fold_results.to_csv("./fitting_stability_results/fold_results.csv")
        prediction_results.to_csv("./fitting_stability_results/prediction_results.csv")
        repetition_results.to_csv("./fitting_stability_results/repetition_results.csv")
        print("Saved fitting stability results to ./fitting_stability_results/.")

        return {
            "fold_results": fold_results,
            "prediction_results": (
                prediction_results
            ),
            "repetition_results": (
                repetition_results
            ),
            "split_seed": split_seed,
        }