import numpy as np
from pathlib import Path
import json
from src.configs.config import CLASS_NAMES

from src.dataset.dataloader import SlidePrediction

##############################################################
# Helper Functions
##############################################################

def softmax(x):

    e = np.exp(x - np.max(x))

    return e / e.sum()


##############################################################

def prediction_entropy(probabilities):

    p = np.asarray(probabilities)

    p = np.clip(
        p,
        1e-12,
        1.0
    )

    return float(
        -(p * np.log(p)).sum()
    )


##############################################################

def prediction_margin(probabilities):

    p = np.sort(probabilities)

    return float(
        p[-1] - p[-2]
    )


##############################################################

def build_prediction(
    probs
):

    pred = np.argmax(probs)

    return SlidePrediction(

        predicted_class=CLASS_NAMES[pred],

        confidence=float(probs[pred]),

        entropy=prediction_entropy(probs),

        margin=prediction_margin(probs),

        probabilities={

            c: float(p)

            for c, p in zip(
                CLASS_NAMES,
                probs
            )

        }

    )


##############################################################

def create_slide_folder(
    output_dir,
    slide_id
):

    folder = Path(output_dir) / slide_id

    folder.mkdir(
        parents=True,
        exist_ok=True
    )

    (folder / "patches").mkdir(exist_ok=True)

    (folder / "cellvit").mkdir(exist_ok=True)

    (folder / "reports").mkdir(exist_ok=True)

    return folder


##############################################################

def save_json(
    data,
    output_path
):

    with open(output_path, "w") as f:

        json.dump(
            data,
            f,
            indent=4
        )
