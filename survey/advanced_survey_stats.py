import numpy as np
import pandas as pd

from scipy.stats import spearmanr, pearsonr

def spearman_brown_correction(correlation):
    if not np.isfinite(correlation):
        return np.nan

    denominator = 1.0 + correlation

    if np.isclose(denominator, 0.0):
        return np.nan

    return (
        2.0 * correlation
        / denominator
    )


def safe_spearman(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    valid = np.isfinite(x) & np.isfinite(y)
    x = x[valid]
    y = y[valid]

    if len(x) < 3:
        return np.nan

    if np.unique(x).size < 2:
        return np.nan

    if np.unique(y).size < 2:
        return np.nan

    return float(spearmanr(x, y).statistic)


def safe_pearson(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    valid = np.isfinite(x) & np.isfinite(y)
    x = x[valid]
    y = y[valid]

    if len(x) < 3:
        return np.nan

    if np.std(x) < 1e-12:
        return np.nan

    if np.std(y) < 1e-12:
        return np.nan

    return float(pearsonr(x, y).statistic)


def repeated_split_half_reliability(
    ratings,
    *,
    participant_col="participant_id",
    image_col="file_id",
    rating_col="rating",
    n_splits=5000,
    min_ratings_per_image_per_half=2,
    random_seed=42,
    maximum_attempt_factor=10,
):
    print(f"\n Computing repeaded split-half reliability with spearman-brown corrected SROCC, PLCC correlation scores for {n_splits} random repeaded splits...\n")
    rng = np.random.default_rng(random_seed)

    participant_ids = np.asarray(
        ratings[participant_col].unique()
    )

    image_ids = pd.Index(
        ratings[image_col].unique()
    )

    results = []
    attempts = 0
    maximum_attempts = (
        n_splits * maximum_attempt_factor
    )

    while len(results) < n_splits:
        attempts += 1

        if attempts > maximum_attempts:
            raise RuntimeError(
                f"Only {len(results)}/{n_splits} valid "
                "participant splits could be constructed. "
                "Inspect the rating allocation or reduce "
                "min_ratings_per_image_per_half."
            )

        shuffled_ids = rng.permutation(
            participant_ids
        )

        midpoint = len(shuffled_ids) // 2

        half_a_ids = shuffled_ids[:midpoint]
        half_b_ids = shuffled_ids[midpoint:]

        half_a = ratings[
            ratings[participant_col].isin(
                half_a_ids
            )
        ]

        half_b = ratings[
            ratings[participant_col].isin(
                half_b_ids
            )
        ]

        count_a = (
            half_a
            .groupby(image_col)[rating_col]
            .count()
            .reindex(image_ids, fill_value=0)
        )

        count_b = (
            half_b
            .groupby(image_col)[rating_col]
            .count()
            .reindex(image_ids, fill_value=0)
        )

        valid_images = (
            (
                count_a
                >= min_ratings_per_image_per_half
            )
            & (
                count_b
                >= min_ratings_per_image_per_half
            )
        )

        # Require every image for directly comparable splits.
        if not valid_images.all():
            continue

        mos_a = (
            half_a
            .groupby(image_col)[rating_col]
            .mean()
            .reindex(image_ids)
        )

        mos_b = (
            half_b
            .groupby(image_col)[rating_col]
            .mean()
            .reindex(image_ids)
        )

        half_srocc = safe_spearman(
            mos_a.to_numpy(),
            mos_b.to_numpy(),
        )

        half_plcc = safe_pearson(
            mos_a.to_numpy(),
            mos_b.to_numpy(),
        )

        if not (
            np.isfinite(half_srocc)
            and np.isfinite(half_plcc)
        ):
            continue

        results.append({
            "split": len(results),
            "attempt": attempts,
            "n_images": len(image_ids),
            "n_participants_a": len(half_a_ids),
            "n_participants_b": len(half_b_ids),
            "srocc_half": half_srocc,
            "srocc_spearman_brown": (
                spearman_brown_correction(
                    half_srocc
                )
            ),
            "plcc_half": half_plcc,
            "plcc_spearman_brown": (
                spearman_brown_correction(
                    half_plcc
                )
            ),
            "minimum_ratings_a": int(
                count_a.min()
            ),
            "minimum_ratings_b": int(
                count_b.min()
            ),
        })

    return pd.DataFrame(results)


def summarize_split_half(results):
    rows = []

    for statistic in (
        "srocc_spearman_brown",
        "plcc_spearman_brown",
    ):
        values = (
            results[statistic]
            .dropna()
            .to_numpy()
        )

        rows.append({
            "statistic": statistic,
            "n_valid_splits": len(values),
            "mean": np.mean(values),
            "median": np.median(values),
            "q025": np.quantile(values, 0.025),
            "q25": np.quantile(values, 0.25),
            "q75": np.quantile(values, 0.75),
            "q975": np.quantile(values, 0.975),
            "minimum": np.min(values),
            "maximum": np.max(values),
        })

    return pd.DataFrame(rows)

def participant_leave_one_out_consistency(
    ratings,
    *,
    participant_col="participant_id",
    image_col="file_id",
    rating_col="rating",
    minimum_images=5,
):
    participant_results = []

    participant_ids = (
        ratings[participant_col]
        .drop_duplicates()
        .to_numpy()
    )

    for participant_id in participant_ids:
        participant_ratings = ratings[
            ratings[participant_col]
            == participant_id
        ]

        other_ratings = ratings[
            ratings[participant_col]
            != participant_id
        ]

        participant_scores = (
            participant_ratings
            .set_index(image_col)[rating_col]
            .rename("participant_rating")
        )

        leave_one_out_mos = (
            other_ratings
            .groupby(image_col)[rating_col]
            .mean()
            .rename("leave_one_out_mos")
        )

        paired = pd.concat(
            [
                participant_scores,
                leave_one_out_mos,
            ],
            axis=1,
            join="inner",
        ).dropna()

        if len(paired) < minimum_images:
            participant_results.append({
                "participant_id": participant_id,
                "n_images": len(paired),
                "srocc": np.nan,
                "plcc": np.nan,
                "mean_signed_difference": np.nan,
                "mean_absolute_difference": np.nan,
            })
            continue

        participant_srocc = safe_spearman(
            paired["participant_rating"],
            paired["leave_one_out_mos"],
        )

        participant_plcc = safe_pearson(
            paired["participant_rating"],
            paired["leave_one_out_mos"],
        )

        difference = (
            paired["participant_rating"]
            - paired["leave_one_out_mos"]
        )

        participant_results.append({
            "participant_id": participant_id,
            "n_images": len(paired),
            "srocc": participant_srocc,
            "plcc": participant_plcc,
            "mean_signed_difference": (
                difference.mean()
            ),
            "mean_absolute_difference": (
                difference.abs().mean()
            ),
        })

    return pd.DataFrame(participant_results)

def summarize_participant_consistency(
    participant_results,
):
    rows = []

    for statistic in ("srocc", "plcc"):
        values = (
            participant_results[statistic]
            .dropna()
            .to_numpy()
        )

        rows.append({
            "statistic": statistic,
            "n_participants": len(values),
            "mean": np.mean(values),
            "median": np.median(values),
            "q25": np.quantile(values, 0.25),
            "q75": np.quantile(values, 0.75),
            "minimum": np.min(values),
            "maximum": np.max(values),
        })

    return pd.DataFrame(rows)