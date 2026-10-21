import pandas as pd
import torch


def load_model(
    model,
    checkpoint,
    device
):

    state = torch.load(
        checkpoint,
        map_location=device
    )

    if "state_dict" in state:
        state = state["state_dict"]

    # remove possible "module." prefix
    new_state = {}
    for k, v in state.items():
        if k.startswith("module."):
            k = k[7:]
        new_state[k] = v

    model.load_state_dict(new_state, strict=True)

    model.eval()

    return model

