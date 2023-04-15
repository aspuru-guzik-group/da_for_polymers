from rdkit import Chem
from rdkit.Chem import AllChem
import pkg_resources
import pandas as pd
import numpy as np

DFT_RAMPRASAD = pkg_resources.resource_filename(
    "da_for_polymers", "data/preprocess/DFT_Ramprasad/dft_exptresults_Egc.csv"
)


CFP_DFT_RAMPRASAD = pkg_resources.resource_filename(
    "da_for_polymers",
    "data/input_representation/DFT_Ramprasad/circular_fingerprint/dft_circular_fingerprint_Egc.csv",
)

# create a function that produces dimer, trimers, and polymer graph (RDKiT, add a new bond at the ends of the trimer)
