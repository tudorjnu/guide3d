from pathlib import Path

import seaborn as sns

test_dataset_path = Path.cwd() / "data"
dataset_path = Path.home() / "data" / "segment-real"

colors = {
    "matplotlib-colors": [
        *sns.palettes.color_palette("deep").as_hex(),
    ],
    "polyline-colors": "#ed0cbc",
}


if __name__ == "__main__":
    __import__("pprint").pprint(colors)
