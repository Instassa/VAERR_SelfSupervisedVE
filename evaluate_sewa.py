from __future__ import annotations

import argparse
import os

import numpy as np
from behavioural_computing.utils.device import get_device
from behavioural_computing.valence_arousal_estimation_audio.estimate_valence_arousal import (
    estimate_valence_arousal,
)
from behavioural_computing.valence_arousal_estimation_audio.estimate_valence_arousal import (
    get_model,
)
from behavioural_computing.valence_arousal_estimation_audio.utils import (
    convert_str_list_to_float,
)
from behavioural_computing.valence_arousal_estimation_audio.utils import get_ccc
from behavioural_computing.valence_arousal_estimation_audio.utils import load_model
from behavioural_computing.valence_arousal_estimation_audio.utils import (
    read_lines_from_file,
)

# @manual=//python/wheel/scipy:scipy
from scipy.stats import pearsonr
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error


def parse_arguments():
    parser = argparse.ArgumentParser(description="Emotion AV Training")

    # Dataset to be used. At the moment we only use SEWA but more datasets might be added in the future.
    parser.add_argument("--dataset", type=str, default="sewa", choices=["sewa"])

    # At the moment only raw audio is used but in the future more types might be used, e.g., MFCCs or spectrograms
    parser.add_argument(
        "--modality", type=str, default="raw_audio", choices=["raw_audio"]
    )

    parser.add_argument("--model-path", default="", help="model path")
    parser.add_argument("--params-path", default="", help="model params path")

    parser.add_argument(
        "--allow-size-mismatch",
        default=True,
        action="store_true",
        help="If True, allows init from model with mismatching weight tensors. Useful to init from model with diff. number of classes",
    )

    parser.add_argument(
        "--output-folder",
        default="",
        help="Folder path where the output will be written",
    )
    parser.add_argument("--sewa-folder-path", default="", help="path to SEWA files")
    parser.add_argument(
        "--annot-path", default="", help="Path to SEWA annotation files"
    )

    parser.add_argument(
        "--sr", default=16000, help=" sampling rate of input audio file"
    )
    args = parser.parse_args()

    return args


args = parse_arguments()

AROUSAL_GT_ANNOT_FILENAME = "_Arousal_V_Aligned.csv"
VALENCE_GT_ANNOT_FILENAME = "_Valence_V_Aligned.csv"
FILENAME_NPZ = "results.npz"
FILENAME_TXT = ["arousal_results.txt", "valence_results.txt"]
ANNOT_NAMES = ["Arousal", "Valence"]

gpu_device = 0
device = get_device()


def save_results_to_txt(
    filenames: list[str],
    stats_per_video: list[dict[list]],
    stats_avg_per_video: list[dict[float]],
    stats_conc: list[dict[float]],
    output_folder: str,
):
    for ind_txt, fname_txt in enumerate(FILENAME_TXT):
        output_file_txt = os.path.join(args.output_folder, fname_txt)

        lines_list = [
            f"{filenames[ind]}, {stats_per_video[ind_txt]['CCC'][ind]}, {stats_per_video[ind_txt]['Corr'][ind]}, {stats_per_video[ind_txt]['MSE'][ind]}, {stats_per_video[ind_txt]['MAE'][ind]} \n"
            for ind in range(len(filenames))
        ]
        avg_over_all_videos_str = f"Average over all videos, CCC: {stats_avg_per_video[ind_txt]['CCC']}, Corr: {stats_avg_per_video[ind_txt]['Corr']}, MAE: {stats_avg_per_video[ind_txt]['MAE']}, MSE:  {stats_avg_per_video[ind_txt]['MSE']} \n"
        conc_perf_str = f"Concatenation of all videos, CCC: {stats_conc[ind_txt]['CCC']}, Corr: {stats_conc[ind_txt]['Corr']}, MAE: {stats_conc[ind_txt]['MAE']}, MSE: {stats_conc[ind_txt]['MSE']} \n"
        lines_list = lines_list + [avg_over_all_videos_str, conc_perf_str]

        print(f"Results written in : {output_file_txt}")
        with open(output_file_txt, "w") as out_f:
            out_f.writelines(lines_list)


def compute_performance(pred_list: list[list[float]], target_list: list[list[float]]):

    nb_annot = len(pred_list)

    stats_per_video = [{} for i in range(nb_annot)]
    stats_avg_per_video = [{} for i in range(nb_annot)]
    stats_conc = [{} for i in range(nb_annot)]

    # compute CCC (=Concordance Correlation Coefficient), corr, MAE, MSE per video for valence and arousal
    for ind in range(nb_annot):

        pred_per_file_list = pred_list[ind]
        targ_per_file_list = target_list[ind]

        # concatenate all predictions and targets from all videos for computing the final stats
        all_preds_conc = np.hstack(pred_per_file_list)
        all_trgs_conc = np.hstack(targ_per_file_list)

        corr_per_video = [
            pearsonr(pred_per_file_list[i], targ_per_file_list[i])[0]
            for i in range(len(pred_per_file_list))
        ]
        ccc_per_video = [
            get_ccc(pred_per_file_list[i], targ_per_file_list[i])
            for i in range(len(pred_per_file_list))
        ]
        mse_per_video = [
            mean_squared_error(pred_per_file_list[i], targ_per_file_list[i])
            for i in range(len(pred_per_file_list))
        ]
        mae_per_video = [
            mean_absolute_error(pred_per_file_list[i], targ_per_file_list[i])
            for i in range(len(pred_per_file_list))
        ]

        stats_per_video[ind]["Corr"] = corr_per_video
        stats_per_video[ind]["CCC"] = ccc_per_video
        stats_per_video[ind]["MAE"] = mse_per_video
        stats_per_video[ind]["MSE"] = mae_per_video

        stats_avg_per_video[ind]["Corr"] = np.nanmean(corr_per_video)
        stats_avg_per_video[ind]["CCC"] = np.mean(ccc_per_video)
        stats_avg_per_video[ind]["MAE"] = np.mean(mse_per_video)
        stats_avg_per_video[ind]["MSE"] = np.mean(mae_per_video)

        stats_conc[ind]["CCC"] = get_ccc(all_preds_conc, all_trgs_conc)
        stats_conc[ind]["Corr"] = pearsonr(all_preds_conc, all_trgs_conc)[0]
        stats_conc[ind]["MSE"] = mean_squared_error(all_preds_conc, all_trgs_conc)
        stats_conc[ind]["MAE"] = mean_absolute_error(all_preds_conc, all_trgs_conc)

    return stats_per_video, stats_avg_per_video, stats_conc


def return_sewa_paths(sewa_folder_path: str):

    audio_filepaths_list = []

    dir_list = os.listdir(sewa_folder_path)

    for folder in dir_list:

        folder_path = os.path.join(sewa_folder_path, folder)

        audio_files_list = [x for x in os.listdir(folder_path) if x.endswith("wav")]

        for file in audio_files_list:

            audio_file_path = os.path.join(folder_path, file)

            audio_filepaths_list.append(audio_file_path)

    return audio_filepaths_list


def main():

    if args.dataset == "sewa":
        args.fps = 50
    args.n_classes = len(ANNOT_NAMES)

    model = get_model(args, device)

    model = load_model(
        args.model_path, model, allow_size_mismatch=args.allow_size_mismatch
    )
    print("Model was successfully loaded")

    bestParams = np.load(args.params_path)["bestParams"]

    audio_filepaths_list = return_sewa_paths(args.sewa_folder_path)

    pred_list = [[], []]
    trg_list = [[], []]

    for filepath in audio_filepaths_list:
        head, tail = os.path.split(filepath)
        filename = head.split("/")[-1]
        filename_arousal_gt = filename + AROUSAL_GT_ANNOT_FILENAME
        filename_valence_gt = filename + VALENCE_GT_ANNOT_FILENAME
        filepath_arousal_gt = os.path.join(args.annot_path, filename_arousal_gt)
        filepath_valence_gt = os.path.join(args.annot_path, filename_valence_gt)

        arousal_pred, valence_pred = estimate_valence_arousal(
            model, bestParams, args.sr, filepath
        )
        arousal_gt = read_lines_from_file(filepath_arousal_gt)
        valence_gt = read_lines_from_file(filepath_valence_gt)

        del valence_gt[0]  # first line contains headers
        del arousal_gt[0]  # first line contains headers

        arousal_gt_np = convert_str_list_to_float(arousal_gt)
        valence_gt_np = convert_str_list_to_float(valence_gt)

        pred_list[0].append(arousal_pred)
        pred_list[1].append(valence_pred)
        trg_list[0].append(arousal_gt_np)
        trg_list[1].append(valence_gt_np)

    stats_per_video, stats_avg_per_video, stats_conc = compute_performance(
        pred_list, trg_list
    )
    output_file_npz = os.path.join(args.output_folder, FILENAME_NPZ)
    print(f"Results written in : {output_file_npz}")
    np.savez(
        output_file_npz,
        filenames=audio_filepaths_list,
        stats_per_video=stats_per_video,
        stats_avg_per_video=stats_avg_per_video,
        stats_conc=stats_conc,
    )
    save_results_to_txt(
        audio_filepaths_list,
        stats_per_video,
        stats_avg_per_video,
        stats_conc,
        args.output_folder,
    )


if __name__ == "__main__":
    main()
