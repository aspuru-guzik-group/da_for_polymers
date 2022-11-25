from rdkit import Chem
from rdkit.Chem import AllChem

def fingerprints_equal(smi_1, smi_2):
    """
    Determines whether two fingerprints are identical using bit matching?
    """
    mol_1 = Chem.MolFromSmiles(smi_1)
    mol_2 = Chem.MolFromSmiles(smi_2)
    bitvector_1 = AllChem.GetMorganFingerprintAsBitVect(mol_1, 3, 512)
    bitvector_2 = AllChem.GetMorganFingerprintAsBitVect(mol_2, 3, 512)
    fp_list_1 = list(bitvector_1.ToBitString())
    fp_map_1 = map(int, fp_list_1)
    fp_1 = list(fp_map_1)
    fp_list_2 = list(bitvector_2.ToBitString())
    fp_map_2 = map(int, fp_list_2)
    fp_2 = list(fp_map_2)
    print(f"{fp_1=}, {fp_2=}")
    print(fp_1 == fp_2)


polymer_order_1 = "COCCCCOC(=O)CCC(C)=O"
polymer_order_2 = "CCC(=O)OCCCCOC(C)=O"

fingerprints_equal(polymer_order_1, polymer_order_2)