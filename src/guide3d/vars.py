from pathlib import Path

import seaborn as sns

test_dataset_path = Path.cwd() / "data"
dataset_path = Path.home() / "data" / "segment-real"
viz_dataset_path = Path.home() / "data" / "segment-real-viz"

colors = {
    "matplotlib-colors": [
        *sns.palettes.color_palette("deep").as_hex(),
    ],
    "polyline-colors": "#ed0cbc",
}

plot_defaults = {
    "markersize": 0.4,
    "linewidth": 0.7,
    "alpha": 0.6,
}


if __name__ == "__main__":
    __import__("pprint").pprint(colors)
