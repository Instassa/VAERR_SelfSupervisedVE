# Following two lines are used since otherwise buck run returns an error when importing librosa
# Fix found on https://www.internalfb.com/diff/D20103134
# //libfb/py:ctypesmonkeypatch should also be added in TARGETS
import libfb.py.ctypesmonkeypatch  # isort:skip

libfb.py.ctypesmonkeypatch.install()

import argparse
import os

# Consider using torchaudio in the future instead of librosa
import librosa
import numpy as np
import torch
from behavioural_computing.utils.device import get_device
from behavioural_computing.valence_arousal_estimation_audio.model import EmotionModel
from behavioural_computing.valence_arousal_estimation_audio.utils import load_model

# @manual=//python/wheel/scipy:scipy
from scipy.signal import medfilt


def parse_arguments():
    parser = argparse.ArgumentParser(description="Emotion AV Training")

    # Dataset to be used. At the moment we only use SEWA but more datasets might be added in the future.
    parser.add_argument("--dataset", type=str, default="sewa", choices=["sewa"])

    # At the moment only raw audio is used but in the future more types might be used, e.g., MFCCs or spectrograms
    parser.add_argument(
        "--modality", type=str, default="raw_audio", choices=["raw_audio"]
    )
    # paths
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
        help="folder path where the output will be written",
    )
    parser.add_argument("--input-path", default="", help="input audio file path ")

    parser.add_argument(
        "--sr", default=16000, help=" sampling rate of input audio file"
    )

    args = parser.parse_args()

    return args


ANNOT_NAMES = ["Arousal", "Valence"]

gpu_device = 0
device = get_device()


def estimate_valence_arousal(model, best_params, sampling_rate: int, input_Path: str):

    audio_raw_stream, sr = librosa.load(input_Path, sr=sampling_rate)
    audio_raw_stream = torch.from_numpy(audio_raw_stream).unsqueeze(0)

    post_processed_out = []

    model.eval()

    with torch.no_grad():

        lengths = tuple([audio_raw_stream.size(1)] * len(audio_raw_stream))

        audio_raw_stream = audio_raw_stream.to(device)

        outputs = model(audio_raw_stream, lengths)

        out = outputs.cpu().numpy()
        out = np.squeeze(out)

        no_outputs = out.shape[1]

        for ind in range(no_outputs):
            # w = window length for median filtering
            w = best_params[ind][0]
            # b = bias, to be added to the predicted values
            b = best_params[ind][1]
            # s = scaling factor, used to scale the predicted values
            s = best_params[ind][2]

            pp_output = (medfilt(out[:, ind], int(w)) + b) * s
            post_processed_out.append(pp_output)

    return post_processed_out[0], post_processed_out[1]


def get_model(args, device):

    model = EmotionModel(n_classes=args.n_classes, device=device).to(device)
    return model


def main():

    args = parse_arguments()
    if args.dataset == "sewa":
        args.fps = 50
    args.n_classes = len(ANNOT_NAMES)

    model = get_model(args, device)
    model = load_model(
        args.model_path, model, allow_size_mismatch=args.allow_size_mismatch
    )
    print("Model was successfully loaded")

    best_params = np.load(args.params_path)["bestParams"]

    pred_tuple = estimate_valence_arousal(model, best_params, args.sr, args.input_path)

    os.makedirs(args.output_folder, exist_ok=True)
    head, tail = os.path.split(args.input_path)
    fName = tail.split(".")[0]

    for ind, pred in enumerate(pred_tuple):

        list_of_lines = [f"{elem}\n" for elem in pred]
        full_filename = f"{fName}_Predicted_{ANNOT_NAMES[ind]}.csv"
        target_filepath = os.path.join(args.output_folder, full_filename)

        print(f"Output path: {target_filepath}")
        with open(target_filepath, "w") as out_f:
            out_f.writelines(list_of_lines)


if __name__ == "__main__":
    main()
