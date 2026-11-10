import argparse
import torch
import torchvision.transforms as T
import torch.nn.functional as F
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np


from PIL import Image
from scipy.stats import spearmanr
from src.model.pretrained import CNNLoader


from options import args

to_tensor = T.ToTensor()
normalize = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

def bootstrap_spearman(df):

    x = df["MOS"].to_numpy()
    y = df["Logit Margin Drop"].to_numpy()

    rho = spearmanr(x, y).statistic

    rng = np.random.default_rng(42)
    bootstrap_rhos = []

    for _ in range(5000):
        indices = rng.integers(0, len(x), size=len(x))

        x_boot = x[indices]
        y_boot = y[indices]

        # Spearman correlation is undefined if either variable is constant
        if np.unique(x_boot).size < 2:
            continue

        if np.unique(y_boot).size < 2:
            continue

        bootstrap_rhos.append(
            spearmanr(x_boot, y_boot).statistic
        )

    rho_lower, rho_upper = np.percentile(
        bootstrap_rhos,
        [2.5, 97.5],
    )

    print(f"Spearman rho: {rho:.2f}")
    print(f"Bootstrap 95% CI: [{rho_lower:.2f}, {rho_upper:.2f}]")

    return (rho, rho_lower, rho_upper)

def signed_margin(logits, labels):
    """
    logits: [B, C]
    labels: [B]
    Returns z_y - max_{k != y} z_k.
    """
    selected = logits.gather(1, labels[:, None]).squeeze(1)

    label_mask = F.one_hot(
        labels, num_classes=logits.shape[1]
    ).bool()

    strongest_other = (
        logits.masked_fill(label_mask, float("-inf"))
        .max(dim=1)
        .values
    )

    return selected - strongest_other

@torch.inference_mode()
def compute_prediction_displacement(model, clean_input, adv_input):
    clean_logits = model(clean_input)
    adv_logits = model(adv_input)

    # Reference class: prediction for the clean image
    reference_class = clean_logits.argmax(dim=1)

    clean_margin = signed_margin(clean_logits, reference_class)
    adv_margin = signed_margin(adv_logits, reference_class)

    clean_pred = clean_logits.argmax(dim=1)
    adv_pred = adv_logits.argmax(dim=1)

    clean_prob = (
        clean_logits.softmax(dim=1)
        .gather(1, reference_class[:, None])
        .squeeze(1)
    )
    adv_prob = (
        adv_logits.softmax(dim=1)
        .gather(1, reference_class[:, None])
        .squeeze(1)
    )

    return {
        "reference_class": reference_class.item(),
        "clean_prediction": clean_pred.item(),
        "adv_prediction": adv_pred.item(),
        "prediction_flip": bool((clean_pred != adv_pred).item()),
        "clean_margin": clean_margin.item(),
        "adv_margin": adv_margin.item(),
        "margin_drop": (clean_margin - adv_margin).item(),
        "clean_reference_probability": clean_prob.item(),
        "adv_reference_probability": adv_prob.item(),
        "probability_drop": (clean_prob - adv_prob).item(),
    }

def load_image(path, device='cpu'):
    image = Image.open(path).convert("RGB")
    image = to_tensor(image)
    normalized_image =  normalize(image).unsqueeze(0).to(device)
    return image, normalized_image
    

attack_2_surrogate = {
    "pgdl2": "deit",
    "percal": "deit",
    "rcw":"deit",
    "hpf_fgsm": "deit",
    "sgd_uap": "deit",
    "dim": "deit",
    "square_attack": "deit",
    "vmifgsm": "resnet",
    "sparsesigmaattack":"resnet",
    "sga_uap": "resnet",
    "bsr": "resnet",
    "ppba": "resnet",
    "cgsp": "resnet",
    "cgnc": "resnet",
}

def load_mos_table(path):
    df = pd.read_csv(path)
    return df

def get_prediction_displacement_for_surrogate(model_name):
    
    n_classes = 1000
    loader = CNNLoader(args.pretrained, args.adversarial_pretrained_opt)
    model, _ = loader.transfer(model_name,
                                      n_classes, 
                                      feature_extract=args.as_ft_extractor, 
                                      device=args.device)
    model = model.eval()
    model.model_name = args.model_name

    sample_2_info = {k:{} for k,v in attack_2_surrogate.items() if v == model_name}

    for attack, inds in dist_dict.items():
        if attack_2_surrogate[attack] == model_name:
            for pert_dist, ind in inds.items():

                index = ind[0]
                attack_pert = f'{attack}-{pert_dist}'
                img_path = path_real_images + str(index) + ".png"
                adv_img_path = path_adv_images + str(pert_dist) + "/" + attack + "/" + str(index) + ".png"

                ori_image, image = load_image(img_path)
                ori_adv_image, adv_image = load_image(adv_img_path)

                prediction_discplacement = compute_prediction_displacement(model, image, adv_image)

                line_for_file = mos_table[mos_table["file_id"] == attack_pert]
                mos_value = line_for_file["mos"]
                prediction_discplacement['mos'] = mos_value.item()
                prediction_discplacement['sample'] = str(index) + '.png'
                prediction_discplacement['attack']
                prediction_discplacement['perturbation_group'] = pert_dist
                prediction_discplacement['l2'] = (ori_image - ori_adv_image).pow(2).sum().sqrt().item()
                prediction_discplacement['surrogate'] = model_name
                sample_2_info[attack][str(pert_dist)] =  prediction_discplacement

    return sample_2_info

def visualize_pred_displacement(info_dict, model_name):

    if model_name == 'resnet':
        model_name = 'ResNet'
    elif model_name == 'deit':
        model_name = 'DeiT'
    else:
        raise ValueError('Model not recognized.')

    sns.set_theme(style="whitegrid", context="paper", font_scale=1.6)

    data = []
    for attack, perts in info_dict.items():
        for pert, infos in perts.items():
            data.append(infos)

    data = sorted(data, key=lambda x: x['MOS'])

    mos_scores = [x['MOS'] for x in data]
    margin_gap_scores = [x['margin_drop'] for x in data]
    pred_flipped = [x['prediction_flip'] for x in data]
    ci_lower = [x['ci_lower'] for x in data]
    ci_upper = [x['ci_upper'] for x in data]

    df = pd.DataFrame(
        zip(mos_scores,margin_gap_scores,pred_flipped, ci_lower, ci_upper), 
        columns=['MOS', 'Logit Margin Drop', 'Prediction Flip', 'Lower CI Bound', 'Upper CI Bound']
        )

    fig, ax = plt.subplots(figsize=(8.0, 5.6), constrained_layout=True)

    ax.errorbar(
    df["MOS"],
    df["Logit Margin Drop"],
    xerr=[
        df["MOS"] - df["Lower CI Bound"],
        df["Upper CI Bound"] - df["MOS"],
    ],
    fmt="none",
    ecolor="0.75",
    elinewidth=0.8,
    capsize=1.5,
    alpha=0.7,
    zorder=1,
    )

    unchanged = df[~df["Prediction Flip"]]
    changed = df[df["Prediction Flip"]]

    ax.scatter(
        unchanged["MOS"],
        unchanged["Logit Margin Drop"],
        marker="o",
        s=48,
        facecolors="white",
        edgecolors="#0072B2",
        linewidths=1.3,
        label="No",
        zorder=3,
    )

    # Filled crosses: prediction changed
    ax.scatter(
        changed["MOS"],
        changed["Logit Margin Drop"],
        marker="X",
        s=58,
        color="#D55E00",
        edgecolors="white",
        linewidths=0.5,
        label="Yes",
        zorder=3,
    ) 

    # Robust descriptive trend with bootstrap confidence band
    sns.regplot(
        data=df,
        x="MOS",
        y="Logit Margin Drop",
        scatter=False,
        robust=True,
        ci=95,
        color="0.25",
        line_kws={"linewidth": 1.4},
        ax=ax,
    )

    # Zero means no reduction in the clean-reference margin
    ax.axhline(
        0,
        color="0.4",
        linestyle="--",
        linewidth=1,
        zorder=0,
    )

    ax.text(
        0.91,
        0.3,
        "No margin change",
        color="0.4",
        fontsize=11,
        va="bottom",
    )

    rho, rho_lower, rho_upper = bootstrap_spearman(df)

    ax.text(
        0.98,
        0.04,
        (
            rf"$\rho={rho:.2f}$, "
            rf"95% CI [{rho_lower:.2f}, {rho_upper:.2f}], "
            rf"$n={len(df)}$"
        ),
        transform=ax.transAxes,
        ha="right",
        va="bottom",
    )


    ax.set(
        xlabel="MOS (higher = greater perceived distortion)",
        ylabel=r"Clean-reference margin reduction, $\Delta m$ (logit units)",
    )

    plt.title(
        f"Perceived Distortion And Prediction Displacement ({model_name})",fontweight='bold'
    )

    ax.legend(
        title="Clean prediction changed",
        frameon=False,
        loc="upper left",
    )

    ax.grid(axis="x", visible=False)
    sns.despine(ax=ax)

    plt.show()
    print('done')



if __name__ == '__main__':

    path_real_images = "./survey/data/perc_eval/images_new/"
    path_adv_images = "./survey/data/perc_eval/"
    mos_table = load_mos_table("./survey/data/perc_eval/scores_and_mos.csv")

    dist_dict = {

            'percal':{'4.0':[12],
                      '8.0':[18],
                      '15.0':[10],
                      '23.0':[3],
                      '38.0':[5]},
            'square_attack':{'4.0':[20],
                      '8.0':[27],
                      '15.0':[28],
                      '23.0':[32],
                      '38.0':[36]},
            'ppba':{'4.0':[40],
                      '8.0':[44],
                      '15.0':[49],
                      '23.0':[53],
                      '38.0':[56]},
            'rcw':{'4.0':[77],
                      '8.0':[66],
                      '15.0':[71],
                      '23.0':[74],
                      '38.0':[63]},
            'cgnc':{'4.0':[83],
                      '8.0':[85],
                      '15.0':[88],
                      '23.0':[93],
                      '38.0':[97]},
            'cgsp':{'4.0':[102],
                      '8.0':[107],
                      '15.0':[109],
                      '23.0':[112],
                      '38.0':[117]},
            'sparsesigmaattack':{'4.0':[120],
                      '8.0':[127],
                      '15.0':[131],
                      '23.0':[134],
                      '38.0':[139]},
            'hpf_fgsm':{'4.0':[142],
                      '8.0':[145],
                      '15.0':[150],
                      '23.0':[153],
                      '38.0':[157]},
            'dim':{'4.0':[161],
                      '8.0':[165],
                      '15.0':[168],
                      '23.0':[175],
                      '38.0':[179]},
            'bsr':{'4.0':[180],
                      '8.0':[184],
                      '15.0':[188],
                      '23.0':[192],
                      '38.0':[196]},
            'vmifgsm':{'4.0':[202],
                      '8.0':[205],
                      '15.0':[209],
                      '23.0':[214],
                      '38.0':[219]},
            'pgdl2':{'4.0':[220],
                      '8.0':[224],
                      '15.0':[228],
                      '23.0':[235],
                      '38.0':[238]},
            'sgd_uap':{'4.0':[241],
                      '8.0':[245],
                      '15.0':[248],
                      '23.0':[253],
                      '38.0':[258]},
            'sga_uap':{'4.0':[263],
                      '8.0':[265],
                      '15.0':[271],
                      '23.0':[275],
                      '38.0':[279]},

        }

    sample_2_info_resnet = get_prediction_displacement_for_surrogate("resnet")
    sample_2_info_deit = get_prediction_displacement_for_surrogate("deit")


    visualize_pred_displacement(sample_2_info_resnet, 'resnet')
    visualize_pred_displacement(sample_2_info_deit, 'deit')






    print('done')



    







    

    