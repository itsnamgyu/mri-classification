import warnings
from argparse import ArgumentParser
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.cm
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from typing import List

from data import Data, labels, re_path
from results import *

mpl.rcParams["font.sans-serif"] = ["Arial"]
mpl.rcParams["image.interpolation"] = "none"

# If the figure does not update, try
# matplotlib.use("TkAgg")



class Labeler:
    class InvalidQueryException(Exception):
        message = "No data matching that query"

    class NoDataException(Exception):
        message = "No data to label"

    def __init__(self, data=None, dataset=None, phase=None, patient=None, predictions=None):
        if data and isinstance(data, Data):
            self.data = data
        else:
            if data:
                warnings.warn(
                    "Invalid data object {}. Reloading data for Labeler".format(data)
                )
            self.data = Data()

        if not self.data.data:
            raise Labeler.NoDataException()

        self.slide_list = []  # (dataset, patient, phase). Only these are considered for labeling.
        for _dataset, patients in self.data.data.items():
            if dataset is not None and dataset != _dataset:
                continue
            for _patient, phases in patients.items():
                if patient is not None and patient != _patient:
                    continue
                for _phase in phases:
                    if phase is not None and phase != _phase:
                        continue
                    self.slide_list.append((_dataset, _patient, _phase))

        # If predictions, only show phases that have been predicted
        if predictions:
            string_keys = set(predictions.keys())
            key_set = set()
            for string_key in string_keys:
                match = re_path.search(string_key)
                dataset, patient, phase, sid = match.groups()
                patient, phase, sid = int(patient), int(phase), int(sid)
                key_set.add((dataset, patient, phase))
            key_set = key_set.intersection(set(self.slide_list))
            self.slide_list = list(sorted(list(key_set)))

        if not self.slide_list:
            raise Labeler.InvalidQueryException()

        self.slide_list.sort()
        self.slide_index = 0
        self.predictions = predictions

        self.current_label = labels[0]
        self.figure: Figure = plt.figure(figsize=(10, 8))
        if self.figure.canvas.manager is not None:
            self.figure.canvas.manager.set_window_title("Cardiac MRI Slice Labeler")

        self.figure.suptitle("", fontsize=12, fontweight="bold")

        axes = self.figure.subplots(4, 6).flatten()
        for ax in axes:
            ax.set_axis_off()
            ax.set_title("", fontsize=9, fontweight="bold")

    def draw(self):
        # self.figure.canvas.draw()
        self.figure.canvas.draw_idle()

    def label_interface(self, start=None):
        if start:
            self.slide_index = max(min(start - 1, len(self.slide_list)- 1), 0)
        self.figure.canvas.mpl_connect("key_press_event", self._on_key_press)
        self.figure.canvas.mpl_connect("button_press_event", self._on_button_press)
        self.figure.canvas.mpl_connect("resize_event", self._on_resize)
        self._update_figure()
        plt.show()

    def save_labels(self):
        self.data.save_labels()

    def _get_slide_slices(self):
        """
        :return:
        [(sid, image_id), ...]
        """
        dataset, patient, phase = self.slide_list[self.slide_index]
        slices = sorted(self.data.data[dataset][patient][phase].items())
        return slices

    def _on_button_press(self, event):
        for i, ax in enumerate(self.figure.get_axes()):
            if event.inaxes == ax:
                try:
                    sid, image_id = self._get_slide_slices()[i]
                    self.data.set_label(image_id, self.current_label)
                except IndexError:
                    return
        self._update_ax_titles()

    def _on_key_press(self, event):
        # Label change
        try:
            self.current_label = labels[int(event.key) - 1]
            self._update_figure_title()
            return
        except (ValueError, IndexError):
            pass

        # Save labels
        if event.key == " ":
            self.save_labels()
            return

        # Slide change
        if event.key == "left":
            self.slide_index -= 1
        if event.key == "right":
            self.slide_index += 1
        if event.key == "h":
            self.slide_index -= 10
        if event.key == "l":
            self.slide_index += 10
        self.slide_index = self.slide_index % len(self.slide_list)
        self._update_figure()

    def _on_resize(self, event):
        self.figure.tight_layout(w_pad=0.75, h_pad=1)

    def _update_figure_title(self, draw=True):
        dataset, patient, phase = self.slide_list[self.slide_index]
        title = "({}/{}) {} - Patient {} - Phase {} [Label: {}]".format(
            self.slide_index + 1,
            len(self.slide_list),
            dataset,
            patient,
            phase,
            self.current_label,
        ).upper()
        self.figure.suptitle(title, fontsize=12, fontweight="bold")

        if draw:
            self.draw()

    def _update_ax_titles(self, draw=True):
        axes: List[Axes] = self.figure.get_axes()
        dataset, patient, phase = self.slide_list[self.slide_index]
        slices = sorted(self.data.data[dataset][patient][phase].items())

        # Ax Titles
        for i, item in enumerate(slices):
            slice_index, sid = item
            ground_truth = self.data.labels.get(sid, "N/A")
            prediction = None
            if self.predictions:
                try:
                    prediction = self.predictions[sid]["prediction"]
                except KeyError:
                    prediction = None
            if prediction:
                title = "Slice {} ({}) (P={})".format(slice_index, ground_truth.upper(), prediction.upper())
            else:
                title = "Slice {} ({})".format(slice_index, ground_truth.upper())
            # PRIVATE INTERFACE (tampers with private attributes for speed)
            # if axes[i].title._text != title:
            if prediction:
                if prediction == ground_truth:
                    axes[i].set_title(title, fontsize=9, fontweight="bold", color="green")
                else:
                    axes[i].set_title(title, fontsize=9, fontweight="bold", color="red")
            else:
                axes[i].set_title(title, fontsize=9, fontweight="bold")
        if draw:
            self.draw()

    def _update_figure(self):
        axes: List[Axes] = self.figure.get_axes()

        # Update images
        dataset, patient, phase = self.slide_list[self.slide_index]
        slices = sorted(self.data.data[dataset][patient][phase].items())

        for i, ax in enumerate(axes):
            # ax.images = []
            # # PRIVATE INTERFACE (tampers with private attributes for speed)
            # ax.title._text = ""
            axes[i].clear()
            axes[i].set_axis_off()
            if i < len(slices):
                sid, image_id = slices[i]
                image = self.data.load_image(image_id)
                axes[i].imshow(image, cmap=mpl.cm.gray)

        # for i, item in enumerate(slices):
        #     sid, image_id = item
        #     image = self.data.load_image(image_id)
        #     axes[i].clear()
        #     axes[i].set_axis_off()
        #     axes[i].imshow(image, cmap=mpl.cm.gray)
        self._update_ax_titles(draw=False)
        self._update_figure_title(draw=False)

        self.figure.tight_layout(w_pad=0.75, h_pad=1)
        self.draw()


def main():
    parser = ArgumentParser()

    parser.add_argument("--start", "-S", type=int)
    parser.add_argument("--dataset", "-D")
    parser.add_argument("--phase", "-P")
    parser.add_argument("--results", "-R", help="Show prediction results", action="store_true")
    args = parser.parse_args()

    result_key = None
    if args.results:
        result_keys = get_result_keys()
        print("Select result")
        print("-" * 80)
        print("{:5s}{:55s}{:20s}".format("#", "TITLE", "ACCURACY"))
        for i, key in enumerate(result_keys):
            accuracy = 0
            try:
                with open(get_result_path(key)) as f:
                    results = json.load(f)
                    accuracy = results["accuracy"]
            except:
                pass
            print("{:<5d}{:55s}{:<20.4f}".format(i, key, accuracy))
        print("-" * 80)
        print("Select result: ", end="")
        while True:
            try:
                selection = int(input())
                result_key = result_keys[selection]
                break
            except (ValueError, IndexError):
                continue

    predictions = None
    if result_key:
        predictions_path = get_predictions_path(result_key)
        try:
            with open(predictions_path) as f:
                predictions = json.load(f)
                print("Loaded predictions")
        except:
            print("No predictions in {}".format(predictions_path))

    try:
        labeler = Labeler(dataset=args.dataset, phase=args.phase, predictions=predictions)
    except (Labeler.InvalidQueryException, Labeler.NoDataException) as e:
        print(e.message)
        return
    labeler.label_interface(start=args.start)
    labeler.save_labels()


if __name__ == "__main__":
    main()
