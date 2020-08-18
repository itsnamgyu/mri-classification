import glob
import json
import os
import re
import shutil
import warnings
from collections import defaultdict
from datetime import datetime
from functools import lru_cache

import matplotlib as mpl
import matplotlib.image

import project

DATA_DIR = os.path.join(project.PROJECT_DIR, "data")
DATASETS_DIR = os.path.join(DATA_DIR, "datasets")
LABELS_PATH = os.path.join(DATA_DIR, "labels.json")
_LABELS_BACKUP_PATH = os.path.join(DATA_DIR, ".labels_{}.json")

IMAGE_EXTENSIONS = ["jpg", "png"]

USERNAME = os.environ.get("USERNAME") or os.environ.get("USER")  # used to mark labeler

_slices = lambda: defaultdict(dict)  # slice_id: path
_phases = lambda: defaultdict(_slices)  # phase_id: slice_dict
_patients = lambda: defaultdict(_phases)  # patient_id: phase_dict
_datasets = lambda: defaultdict(_patients)  # dataset_id: patient_dict

path_format = "{dataset}_PA{patient:06d}_PH{phase:02d}_S{slice:02d}"
re_path = re.compile(
    "(?P<dataset>[A-Z0-9]{3})_PA(?P<patient>\d+)_PH(?P<phase>\d+)_S(?P<slice>\d+)"
)

# Labels
OAP = "oap"
AP = "ap"
MID = "mid"
BS = "bs"
OBS = "obs"

labels = [
    OAP, AP, MID, BS, OBS
]


class Data:
    def __init__(self):
        self.image_paths = []
        for ext in IMAGE_EXTENSIONS:
            self.image_paths += glob.glob(
                os.path.join(DATASETS_DIR, "**", "*.{}".format(ext))
            )

        # Organize paths into data dictionary
        self.data = _datasets()
        self.paths = {}  # data_id: path
        self.unmatched_paths = []
        for path in sorted(self.image_paths):
            match = re_path.search(path)
            if not match:
                self.unmatched_paths.append(path)
            else:
                try:
                    image_id = match.group(0)
                    dataset, patient, phase, sid = match.groups()
                    patient, phase, sid = int(patient), int(phase), int(sid)
                    self.data[dataset][patient][phase][sid] = image_id
                    self.paths[image_id] = path
                except ValueError:
                    self.unmatched_paths.append(path)
        self.datasets = sorted(self.data.keys())

        if self.unmatched_paths:
            warnings.warn(
                "{} Invalid image(s) in {} including {}".format(
                    len(self.unmatched_paths), DATASETS_DIR, self.unmatched_paths[0]
                )
            )

        # Load label data
        self.label_data = defaultdict(dict)  # path: label
        try:
            with open(LABELS_PATH) as f:
                self.label_data.update(json.load(f))
        except FileNotFoundError:
            warnings.warn("Labels file {} doesn't exist.".format(LABELS_PATH))
        self.labels = {key: value["label"] for key, value in self.label_data.items()}

    def set_label(self, image_id, label):
        assert label in [OAP, AP, MID, BS, OBS]
        self.labels[image_id] = label
        self.label_data[image_id]["label"] = label
        self.label_data[image_id]["labeled_on"] = (
            datetime.now().astimezone().isoformat()
        )
        self.label_data[image_id]["labeled_by"] = USERNAME

    @lru_cache(1024)
    def load_image(self, image_id):
        path = self.paths.get(image_id)
        if not path:
            return None
        else:
            return mpl.image.imread(path)



    def save_labels(self, verbose=1):
        backup_path = _LABELS_BACKUP_PATH.format(
            datetime.utcnow().strftime("%Y%m%d%H%M%S")
        )
        if os.path.exists(LABELS_PATH):
            shutil.copy(LABELS_PATH, backup_path)
            if verbose:
                print("Previous labels backed up to: {}".format(backup_path))

        with open(LABELS_PATH, "w") as f:
            json.dump(self.label_data, f)
            if verbose:
                print("Labels saved to:              {}".format(LABELS_PATH))


if __name__ == "__main__":
    print("Loading all data in {}".format(DATASETS_DIR))
    data = Data()
    print(json.dumps(data.data, indent=4)[:1000])
    print("...")
    if data.unmatched_paths:
        print("Unmatched paths detected")
        print(json.dumps(data.unmatched_paths, indent=4)[:1000])
        print("...")
    else:
        print("No problems detected")

    print(
        "{} images identified".format(len(data.image_paths) - len(data.unmatched_paths))
    )
