from collections import defaultdict

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