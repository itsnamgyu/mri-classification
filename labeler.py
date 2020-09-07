import warnings
from argparse import ArgumentParser
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.cm
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from typing import List

from skimage.transform import resize

from data import Data, labels

mpl.rcParams["font.sans-serif"] = ["Arial"]
mpl.rcParams["image.interpolation"] = "none"



class Labeler:
    class InvalidQueryException(Exception):
        message = "No data matching that query"

    class NoDataException(Exception):
        message = "No data to label"

    def __init__(self, data=None, dataset=None, phase=None, patient=None):
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

        if not self.slide_list:
            raise Labeler.InvalidQueryException()

        self.slide_list.sort()
        self.slide_index = 0

        self.current_label = labels[0]
        self.figure: Figure = plt.figure(figsize=(10, 8))
        self.figure.canvas.set_window_title("Cardiac MRI Slice Labeler")
        self.figure.suptitle("", fontsize=12, fontweight="bold")

        axes = self.figure.subplots(4, 6).flatten()
        for ax in axes:
            ax.set_axis_off()
            ax.set_title("", fontsize=9, fontweight="bold")

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
            self.figure.canvas.draw()

    def _update_ax_titles(self, draw=True):
        axes: List[Axes] = self.figure.get_axes()
        dataset, patient, phase = self.slide_list[self.slide_index]
        slices = sorted(self.data.data[dataset][patient][phase].items())

        # Ax Titles
        for i, item in enumerate(slices):
            sid, image_id = item
            title = "Slice {} ({})".format(
                sid, self.data.labels.get(image_id, "N/A")
            ).upper()
            # PRIVATE INTERFACE (tampers with private attributes for speed)
            if axes[i].title._text != title:
                axes[i].set_title(title, fontsize=9, fontweight="bold")
        if draw:
            self.figure.canvas.draw()

    def _update_figure(self):
        axes: List[Axes] = self.figure.get_axes()

        # Update images
        dataset, patient, phase = self.slide_list[self.slide_index]
        slices = sorted(self.data.data[dataset][patient][phase].items())

        for ax in axes:
            ax.images = []
            # PRIVATE INTERFACE (tampers with private attributes for speed)
            ax.title._text = ""
        for i, item in enumerate(slices):
            sid, image_id = item
            image = self.data.load_image(image_id)
            axes[i].imshow(image, cmap=mpl.cm.gray)
        self._update_ax_titles()
        self._update_figure_title()

        self.figure.tight_layout(w_pad=0.75, h_pad=1)
        self.figure.canvas.draw()


def main():
    parser = ArgumentParser()

    parser.add_argument("--start", "-S", type=int)
    parser.add_argument("--dataset", "-D")
    parser.add_argument("--phase", "-P")
    args = parser.parse_args()
    try:
        labeler = Labeler(dataset=args.dataset, phase=args.phase)
    except (Labeler.InvalidQueryException, Labeler.NoDataException) as e:
        print(e.message)
        return
    labeler.label_interface(start=args.start)
    labeler.save_labels()

if __name__ == "__main__":
    main()
