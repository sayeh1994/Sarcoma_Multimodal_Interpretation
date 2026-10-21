from typing import Dict, List, Optional
import pandas as pd

##############################################################
# Metadata Manager
##############################################################

class MetadataManager:

    def __init__(
        self,
        tcga_tsv: Optional[str] = None,
        cptac_csv: Optional[str] = None,
    ):

        self.tcga = None
        self.cptac = None

        if tcga_tsv is not None:

            self.tcga = pd.read_csv(
                tcga_tsv,
                sep="\t"
            )

        if cptac_csv is not None:

            self.cptac = pd.read_csv(
                cptac_csv
            )

    ##########################################################

    def detect_dataset(
        self,
        slide_id: str
    ) -> str:

        slide_upper = slide_id.upper()

        if slide_upper.startswith("TCGA"):

            return "TCGA"

        if "C3" in slide_upper:

            return "CPTAC"

        return "OTHER"

    ##########################################################

    def get_patient_id(
        self,
        slide_id: str
    ):

        if slide_id.startswith("TCGA"):

            return "-".join(slide_id.split("-")[:4])

        return slide_id

    ##########################################################

    def lookup_organ(
        self,
        slide_id: str,
        manual_organ: Optional[str] = None
    ) -> str:

        dataset = self.detect_dataset(slide_id)

        ######################################################
        # TCGA
        ######################################################

        if dataset == "TCGA" and self.tcga is not None:

            patient = self.get_patient_id(slide_id)

            possible_columns = [
                "submitter_id",
                "case_submitter_id",
                "case_id"
            ]

            organ_columns = [
                "tissue_or_organ_of_origin",
                "primary_site",
                "site_of_resection_or_biopsy"
            ]

            for id_col in possible_columns:

                if id_col not in self.tcga.columns:
                    continue

                row = self.tcga[
                    self.tcga[id_col].astype(str).str.startswith(patient)
                ]

                if len(row) == 0:
                    continue

                for organ_col in organ_columns:

                    if organ_col in row.columns:

                        value = row.iloc[0][organ_col]

                        if pd.notna(value):

                            return str(value)

        ######################################################
        # CPTAC
        ######################################################

        if dataset == "CPTAC" and self.cptac is not None:

            for col in self.cptac.columns:

                values = self.cptac[col].astype(str)

                idx = values.str.contains(
                    slide_id,
                    case=False,
                    regex=False
                )

                if idx.any():

                    row = self.cptac[idx].iloc[0]

                    for organ_col in [
                        "Tumor_Site",
                        "Primary Site",
                        "PrimarySite",
                        "Tissue"
                    ]:

                        if organ_col in row.index:

                            value = row[organ_col]

                            if pd.notna(value):

                                return str(value)

        ######################################################
        # fallback
        ######################################################

        if manual_organ is not None:

            return manual_organ

        return "Unknown"
