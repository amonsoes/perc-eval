from collections import defaultdict
import numpy as np

class FivePLFitError(RuntimeError):
    """The 5PL model could not be fitted for a metric/fold."""

def build_votes_from_ratings(
    ratings,
    image_col="file_id",
    rating_col="rating",
):
    votes = defaultdict(lambda: defaultdict(list))

    for stimulus_id, group in ratings.groupby(
        image_col,
        sort=False,
    ):
        attack, perturbation = stimulus_id.rsplit("-", 1)

        votes[attack][perturbation] = (
            group[rating_col].tolist()
        )

    return votes

def combine_pe_ci(point_estimates, confidence_intervals, out_file, flip=False):

    point_estimates = point_estimates.round(3)
    confidence_intervals = confidence_intervals.round(3)

    fp_csv = out_file + '.csv'
    fp_tex = out_file + '.tex'

    common_cols = point_estimates.columns.intersection(confidence_intervals.columns)

    combined_df = point_estimates[common_cols].copy()

    for row in combined_df.index:
        for col in common_cols:
            point = point_estimates.at[row, col]
            lh = confidence_intervals.at[row, col]
            if isinstance(lh, np.ndarray):
                low, high = confidence_intervals.at[row, col]

                combined_df.at[row, col] = (
                    f"{point:.3f} [{low:.3f}, {high:.3f}]"
                )
            else:
                combined_df.at[row, col] = (
                    f"{point:.3f} [{lh:.3f}]"
                )


    if flip:
        combined_df = combined_df.T

    combined_df.to_csv(fp_csv)
    combined_df.to_latex(fp_tex)
    
    return combined_df