from typing import Dict, List, Optional
from dataclasses import dataclass

##############################################################
# Dataclasses
##############################################################

@dataclass
class PatchEvidence:

    patch_id: str

    rank: int

    patch_index: int

    coord_x: float

    coord_y: float

    attention: float

    percentile: float

    image_path: Optional[str] = None

    cellvit_path: Optional[str] = None
    
    cellvit_cells: Optional[list] = None
    
    cellvit_summary: Optional[dict] = None

    cell_statistics: Optional[dict] = None


@dataclass
class SlidePrediction:

    predicted_class: str

    confidence: float

    entropy: float

    margin: float

    probabilities: Dict[str, float]
