from MIL_Lab.src.models.transmil import (
    TransMILModel,
    TransMILConfig,
)
import torch
import yaml


##############################################################
# Model
##############################################################

def load_model(
    checkpoint,
    device
):
    config_yaml = "MIL_Lab/src/model_configs/transmil"+ "base.yaml"
    with open(config_yaml, "r") as f:

        cfg_dict = yaml.safe_load(f)

    config = TransMILConfig(**cfg_dict)

    model = TransMILModel(
        config=config,
        in_dim=768,
        num_classes=5
    ).to(device)

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