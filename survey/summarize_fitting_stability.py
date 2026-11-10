from pathlib import Path
import math

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

METRIC_ORDER = [
    "topiq_fr",
    "ms_ssim",
    "mad",
    "vif",
    "lpips",
    "psnr",
    "l2",
    "dists",
    "ssim",
    "ahiq",
]

METRIC_LABELS = {
    "topiq_fr": "TOPIQ-FR",
    "ms_ssim": "MS-SSIM",
    "mad": "MAD",
    "vif": "VIF",
    "lpips": "LPIPS",
    "psnr": "PSNR",
    "l2": "$L_2$",
    "dists": "DISTS",
    "ssim": "SSIM",
    "ahiq": "AHIQ",
}

PARAMETERS = ["a", "b", "c", "d", "g"]


# ---------------------------------------------------------------------
# General helpers
# ---------------------------------------------------------------------

def read_results_csv(path):
    """Read a result CSV and remove pandas-generated index columns."""
    df = pd.read_csv(path)

    return df.loc[
        :,
        ~df.columns.str.match(r"^Unnamed:")
    ].copy()


def require_columns(df, required, table_name):
    missing = sorted(set(required) - set(df.columns))

    if missing:
        raise ValueError(
            f"{table_name} is missing the following columns: {missing}"
        )


def order_metrics(df):
    """Apply the desired metric order."""
    df = df.copy()

    df["metric"] = pd.Categorical(
        df["metric"],
        categories=METRIC_ORDER,
        ordered=True,
    )

    return (
        df.sort_values("metric")
        .reset_index(drop=True)
    )


def metric_label(metric):
    return METRIC_LABELS.get(str(metric), str(metric))


def format_interval(median, q1, q3, digits=3):
    """Format a parameter as median [Q1, Q3]."""
    if not np.all(np.isfinite([median, q1, q3])):
        return "--"

    return (
        f"{median:.{digits}f} "
        f"[{q1:.{digits}f}, {q3:.{digits}f}]"
    )


def format_mean_sd(mean, sd, digits=4):
    if not np.all(np.isfinite([mean, sd])):
        return "--"

    return (
        f"${mean:.{digits}f} "
        f"\\pm {sd:.{digits}f}$"
    )


def format_scientific_latex(value, digits=2):
    """Format 15000000 as $1.50 \\times 10^{7}$."""
    if not np.isfinite(value):
        return "--"

    if value == 0:
        return "$0$"

    exponent = int(math.floor(math.log10(abs(value))))
    mantissa = value / (10 ** exponent)

    return (
        f"${mantissa:.{digits}f} "
        f"\\times 10^{{{exponent}}}$"
    )


def wrap_latex_table(
    display_df,
    caption,
    label,
):
    """
    Wrap a pandas-generated tabular environment in a table* and resizebox.
    Requires booktabs and graphicx.
    """
    column_format = "l" + "c" * (len(display_df.columns) - 1)

    tabular = display_df.to_latex(
        index=False,
        escape=False,
        na_rep="--",
        column_format=column_format,
    )

    return f"""\\begin{{table*}}[t]
\\centering
\\caption{{{caption}}}
\\label{{{label}}}
\\scriptsize
\\resizebox{{\\textwidth}}{{!}}{{%
{tabular}
}}
\\end{{table*}}
"""


# ---------------------------------------------------------------------
# 1. Parameter stability
# ---------------------------------------------------------------------

def compute_parameter_stability(fold_results):
    """
    Summarize parameter estimates and fit diagnostics across all
    fold-specific fits.

    Parameter stability:
      - q1, median, and q3 for a, b, c, d, and g
      - parameter-at-bound percentage
      - covariance condition number

    Optimization and prediction diagnostics:
      - convergence rate
      - successful initialization count
      - monotonicity rate
      - out-of-range fold rate
      - held-out fold RMSE
    """
    required = {
        "metric",
        "repetition",
        "fold",
        "converged",
        "successful_starts",
        "parameter_at_bound",
        "monotonic",
        "prediction_outside_mos",
        "covariance_condition",
        "test_rmse",
        *PARAMETERS,
    }
    require_columns(
        fold_results,
        required,
        "fold_results",
    )

    rows = []

    for metric, metric_df in fold_results.groupby(
        "metric",
        observed=True,
        sort=False,
    ):
        condition_numbers = metric_df[
            "covariance_condition"
        ].replace([np.inf, -np.inf], np.nan)

        row = {
            "metric": metric,
            "n_fits": len(metric_df),
            "n_repetitions": metric_df["repetition"].nunique(),
            "convergence_percent":
                100 * metric_df["converged"].mean(),
            "successful_starts_mean":
                metric_df["successful_starts"].mean(),
            "successful_starts_min":
                metric_df["successful_starts"].min(),
            "monotonic_percent":
                100 * metric_df["monotonic"].mean(),
            "boundary_fit_percent":
                100 * metric_df["parameter_at_bound"].mean(),
            "covariance_condition_median":
                condition_numbers.median(),
            "covariance_condition_q1":
                condition_numbers.quantile(0.25),
            "covariance_condition_q3":
                condition_numbers.quantile(0.75),
            "out_of_range_fit_percent":
                100 * metric_df[
                    "prediction_outside_mos"
                ].mean(),
            "test_rmse_mean":
                metric_df["test_rmse"].mean(),
            "test_rmse_sd":
                metric_df["test_rmse"].std(ddof=1),
        }

        for parameter in PARAMETERS:
            estimates = metric_df[parameter]

            row[f"{parameter}_q1"] = estimates.quantile(0.25)
            row[f"{parameter}_median"] = estimates.median()
            row[f"{parameter}_q3"] = estimates.quantile(0.75)
            row[f"{parameter}_min"] = estimates.min()
            row[f"{parameter}_max"] = estimates.max()

        rows.append(row)

    result = pd.DataFrame(rows)

    return order_metrics(result)


def format_parameter_stability_table(parameter_numeric):
    """
    Create the compact, manuscript-facing parameter stability table.

    The numeric table should still be saved separately because this
    formatted table contains strings.
    """
    rows = []

    for _, row in parameter_numeric.iterrows():
        formatted = {
            "Metric": metric_label(row["metric"]),
        }

        for parameter in PARAMETERS:
            formatted[
                f"${parameter}$: median $[Q_1,Q_3]$"
            ] = format_interval(
                row[f"{parameter}_median"],
                row[f"{parameter}_q1"],
                row[f"{parameter}_q3"],
            )

        formatted["Converged (\\%)"] = (
            f"{row['convergence_percent']:.1f}"
        )
        formatted["Successful starts"] = (
            f"{row['successful_starts_mean']:.1f}"
        )
        formatted["Boundary fits (\\%)"] = (
            f"{row['boundary_fit_percent']:.1f}"
        )
        formatted["Median $\\kappa(\\boldsymbol{\\Sigma})$"] = (
            format_scientific_latex(
                row["covariance_condition_median"]
            )
        )
        formatted["Out-of-range fits (\\%)"] = (
            f"{row['out_of_range_fit_percent']:.1f}"
        )
        formatted["Test RMSE"] = format_mean_sd(
            row["test_rmse_mean"],
            row["test_rmse_sd"],
        )

        rows.append(formatted)

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------
# 2. Prediction stability
# ---------------------------------------------------------------------

def compute_prediction_stability(
    prediction_results,
    repetition_results,
):
    """
    Measure the sensitivity of held-out predictions to fold assignment.

    First, compute the prediction distribution for every metric-stimulus
    pair across repeated CV partitions. Then summarize those stimulus-level
    distributions for each metric.

    Returns:
        by_stimulus:
            One row per metric and stimulus.

        summary:
            One row per metric, including repeated-CV performance and
            prediction-stability statistics.
    """
    required_predictions = {
        "repetition",
        "fold",
        "metric",
        "file_id",
        "mos",
        "prediction",
    }
    require_columns(
        prediction_results,
        required_predictions,
        "prediction_results",
    )

    required_repetitions = {
        "repetition",
        "metric",
        "successful",
        "plcc",
        "rmse",
        "or",
        "plcc_rank",
    }
    require_columns(
        repetition_results,
        required_repetitions,
        "repetition_results",
    )

    predictions = prediction_results.copy()

    # Each stimulus should have at most one out-of-fold prediction
    # for each metric in each repetition.
    prediction_key = [
        "metric",
        "file_id",
        "repetition",
    ]

    duplicated = predictions.duplicated(
        prediction_key,
        keep=False,
    )

    if duplicated.any():
        examples = predictions.loc[
            duplicated,
            prediction_key,
        ].head()

        raise ValueError(
            "Duplicate out-of-fold predictions were found for the "
            "same metric, stimulus, and repetition.\n"
            f"Examples:\n{examples}"
        )

    # MOS should remain identical across repetitions.
    mos_counts = (
        predictions
        .groupby(
            ["metric", "file_id"],
            observed=True,
        )["mos"]
        .nunique(dropna=False)
    )

    if (mos_counts > 1).any():
        raise ValueError(
            "MOS is not constant across repetitions for at least "
            "one metric-stimulus pair."
        )

    predictions["prediction_outside_mos"] = (
        (predictions["prediction"] < 0)
        | (predictions["prediction"] > 1)
    )

    # Stability for every individual stimulus.
    by_stimulus = (
        predictions
        .groupby(
            ["metric", "file_id"],
            observed=True,
        )
        .agg(
            mos=("mos", "first"),
            n_predictions=("prediction", "size"),
            n_repetitions=("repetition", "nunique"),
            prediction_mean=("prediction", "mean"),
            prediction_sd=("prediction", "std"),
            prediction_min=("prediction", "min"),
            prediction_max=("prediction", "max"),
            out_of_range_prediction_percent=(
                "prediction_outside_mos",
                lambda x: 100 * x.mean(),
            ),
        )
        .reset_index()
    )

    by_stimulus["prediction_range"] = (
        by_stimulus["prediction_max"]
        - by_stimulus["prediction_min"]
    )

    # Aggregate the stimulus-specific stability statistics by metric.
    prediction_summary = (
        by_stimulus
        .groupby(
            "metric",
            observed=True,
        )
        .agg(
            n_stimuli=("file_id", "nunique"),
            repetitions_per_stimulus_min=(
                "n_repetitions",
                "min",
            ),
            repetitions_per_stimulus_max=(
                "n_repetitions",
                "max",
            ),
            mean_within_stimulus_prediction_sd=(
                "prediction_sd",
                "mean",
            ),
            median_within_stimulus_prediction_sd=(
                "prediction_sd",
                "median",
            ),
            max_within_stimulus_prediction_sd=(
                "prediction_sd",
                "max",
            ),
            mean_prediction_range=(
                "prediction_range",
                "mean",
            ),
            median_prediction_range=(
                "prediction_range",
                "median",
            ),
            max_prediction_range=(
                "prediction_range",
                "max",
            ),
            mean_out_of_range_prediction_percent=(
                "out_of_range_prediction_percent",
                "mean",
            ),
        )
        .reset_index()
    )

    # Repeated-CV predictive performance.
    repetition_summary = (
        repetition_results
        .groupby(
            "metric",
            observed=True,
        )
        .agg(
            n_repetitions=("repetition", "nunique"),
            successful_repetition_percent=(
                "successful",
                lambda x: 100 * x.mean(),
            ),
            plcc_mean=("plcc", "mean"),
            plcc_sd=("plcc", "std"),
            rmse_mean=("rmse", "mean"),
            rmse_sd=("rmse", "std"),
            or_mean=("or", "mean"),
            or_sd=("or", "std"),
            plcc_rank_mean=("plcc_rank", "mean"),
            plcc_rank_sd=("plcc_rank", "std"),
            first_rank_count=(
                "plcc_rank",
                lambda x: int((x == 1).sum()),
            ),
        )
        .reset_index()
    )

    summary = repetition_summary.merge(
        prediction_summary,
        on="metric",
        how="outer",
        validate="one_to_one",
    )

    return (
        order_metrics(by_stimulus),
        order_metrics(summary),
    )


def format_prediction_stability_table(prediction_numeric):
    """Create the manuscript-facing prediction stability table."""
    rows = []

    for _, row in prediction_numeric.iterrows():
        rows.append({
            "Metric": metric_label(row["metric"]),
            "PLCC": format_mean_sd(
                row["plcc_mean"],
                row["plcc_sd"],
            ),
            "RMSE": format_mean_sd(
                row["rmse_mean"],
                row["rmse_sd"],
            ),
            "OR": format_mean_sd(
                row["or_mean"],
                row["or_sd"],
            ),
            "Mean PLCC rank": (
                f"{row['plcc_rank_mean']:.2f}"
            ),
            "Ranked first": (
                f"{int(row['first_rank_count'])}"
            ),
            "Mean $\\sigma_{\\hat{y}}$": (
                f"{row['mean_within_stimulus_prediction_sd']:.4f}"
            ),
            "Median $\\sigma_{\\hat{y}}$": (
                f"{row['median_within_stimulus_prediction_sd']:.4f}"
            ),
            "Maximum $\\sigma_{\\hat{y}}$": (
                f"{row['max_within_stimulus_prediction_sd']:.4f}"
            ),
            "Mean prediction range": (
                f"{row['mean_prediction_range']:.4f}"
            ),
            "Out-of-range predictions (\\%)": (
                f"{row['mean_out_of_range_prediction_percent']:.2f}"
            ),
        })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------
# Run the analysis and export the tables
# ---------------------------------------------------------------------

def main():

    input_dir = Path("./fitting_stability_results")
    output_dir = Path("./fitting_stability_results/fitting_stability_summaries")
    output_dir.mkdir(parents=True, exist_ok=True)

    fold_results = read_results_csv(
        input_dir / "fold_results.csv"
    )
    repetition_results = read_results_csv(
        input_dir / "repetition_results.csv"
    )
    prediction_results = read_results_csv(
        input_dir / "prediction_results.csv"
    )

    # Parameter stability
    parameter_numeric = compute_parameter_stability(
        fold_results
    )
    parameter_display = format_parameter_stability_table(
        parameter_numeric
    )

    # Prediction stability
    prediction_by_stimulus, prediction_numeric = (
        compute_prediction_stability(
            prediction_results,
            repetition_results,
        )
    )
    prediction_display = (
        format_prediction_stability_table(
            prediction_numeric
        )
    )

    # Numeric outputs remain suitable for further analysis.
    parameter_numeric.to_csv(
        output_dir / "parameter_stability_numeric.csv",
        index=False,
    )
    prediction_by_stimulus.to_csv(
        output_dir / "prediction_stability_by_stimulus.csv",
        index=False,
    )
    prediction_numeric.to_csv(
        output_dir / "prediction_stability_numeric.csv",
        index=False,
    )

    # Manuscript-facing LaTeX tables.
    parameter_latex = wrap_latex_table(
        parameter_display,
        caption=(
            "Fold-level stability diagnostics for the "
            "five-parameter logistic mappings. Parameter "
            "estimates are reported as the median "
            "$[Q_1,Q_3]$ across fold-specific fits."
        ),
        label="tab:5pl_parameter_stability",
    )

    prediction_latex = wrap_latex_table(
        prediction_display,
        caption=(
            "Stability of out-of-fold predictions across "
            "repeated cross-validation partitions. PLCC, "
            "RMSE, and OR are reported as mean $\\pm$ "
            "standard deviation across repetitions."
        ),
        label="tab:5pl_prediction_stability",
    )

    (output_dir / "parameter_stability_table.tex").write_text(
        parameter_latex,
        encoding="utf-8",
    )
    (output_dir / "prediction_stability_table.tex").write_text(
        prediction_latex,
        encoding="utf-8",
    )

    print("\nParameter stability:")
    print(parameter_display.to_string(index=False))

    print("\nPrediction stability:")
    print(prediction_display.to_string(index=False))


if __name__ == "__main__":
    main()