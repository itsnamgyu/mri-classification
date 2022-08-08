import json
import os
from typing import List

from data import Data

OUTPUT_DIR = os.path.join(os.path.abspath(os.path.dirname(__file__)), "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)


def compile_predictions_from_phase_output(data: Data, preds, slice_index: List[List[str]]):
    """
    :param preds:
    Prediction outputs from phase-based model
    :param slice_index:
    List of list of sids
    :return:
    {
        sid: {
            scores: {}
            prediction: str,
            ground_truth: str
        },
        ...
    }
    """
    labels = ["ap", "bs", "mid", "oap", "obs", ]
    predictions = dict()  # { sid: { ap: float, bs: float, mid: float, oap: float, obs: float, prediction: str } }
    for p, slice_list in enumerate(slice_index):
        phase_preds = preds[p]
        for s, sid in enumerate(slice_list):
            slice_preds = phase_preds[s]
            d = dict()
            scores = dict()
            for i, label in enumerate(labels):
                scores[label] = str(slice_preds[i])
            d["scores"] = scores
            d["prediction"] = labels[slice_preds.argmax()]
            d["ground_truth"] = data.labels[sid]
            predictions[sid] = d

    return predictions


def compile_predictions_from_slice_output(data: Data, preds, slice_list: List[str]):
    """
    :param preds:
    Prediction outputs from slice-based model
    :param slice_list:
    List of sids
    :return:
    {
        sid: {
            scores: {}
            prediction: str,
            ground_truth: str
        },
        ...
    }
    """
    labels = ["ap", "bs", "mid", "oap", "obs", ]
    predictions = dict()  # { sid: { ap: float, bs: float, mid: float, oap: float, obs: float, prediction: str } }
    for s, sid in enumerate(slice_list):
        slice_preds = preds[s]
        d = dict()
        scores = dict()
        for i, label in enumerate(labels):
            scores[label] = str(slice_preds[i])
        d["scores"] = scores
        d["prediction"] = labels[slice_preds.argmax()]
        d["ground_truth"] = data.labels[sid]
        predictions[sid] = d
    return predictions


def evaluate_predictions(predictions):
    items = list(sorted(predictions.items()))
    total_phases = 0
    correct_slices = 0
    correct_phases = 0

    current_phase = None
    phase_correct = False
    for sid, d in items:
        phase = sid[:-4]
        # next phase
        if current_phase != phase:
            current_phase = phase
            total_phases += 1
            if phase_correct:
                correct_phases += 1
            phase_correct = True
        # evaluate slice
        if d["prediction"] == d["ground_truth"]:
            correct_slices += 1
        else:
            phase_correct = False
    # last phase
    if phase_correct:
        correct_phases += 1

    slice_acc = correct_slices / len(predictions)
    phase_acc = correct_phases / total_phases
    return dict(accuracy=slice_acc, phase_accuracy=phase_acc)


def get_result_keys():
    return os.listdir(OUTPUT_DIR)


def get_info_path(result_key):
    return os.path.join(OUTPUT_DIR, result_key, "info.txt")


def get_result_dir(result_key):
    return os.path.join(OUTPUT_DIR, result_key)


def get_history_path(result_key):
    return os.path.join(OUTPUT_DIR, result_key, "history.json")


def get_model_weights_path(result_key):
    path = os.path.join(OUTPUT_DIR, result_key, "model_weights.h5")
    return path


def get_predictions_path(result_key):
    path = os.path.join(OUTPUT_DIR, result_key, "predictions.json")
    return path


def get_result_path(result_key):
    path = os.path.join(OUTPUT_DIR, result_key, "results.json")
    return path
