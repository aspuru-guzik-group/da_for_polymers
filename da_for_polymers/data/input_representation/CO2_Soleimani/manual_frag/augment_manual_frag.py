from rdkit import Chem
from rdkit.Chem import AllChem

smi_a = "CCCCC(CC)Cc1ccc(-c2c3cc(C)sc3c(-c3ccc(CC(CC)CCCC)s3)c3ccsc23)s1"
smi_b = "c1ccsc1"

smi_1 = smi_a + "." + smi_b
mol_1 = Chem.MolFromSmiles(smi_1)

smi_2 = smi_b + "." + smi_a
mol_2 = Chem.MolFromSmiles(smi_2)

bitvector_polymer = AllChem.GetMorganFingerprintAsBitVect(mol_1, 3, nBits=1024)
fp_list = list(bitvector_polymer.ToBitString())
fp_map = map(int, fp_list)
fp_1 = list(fp_map)

bitvector_polymer = AllChem.GetMorganFingerprintAsBitVect(mol_2, 3, nBits=1024)
fp_list = list(bitvector_polymer.ToBitString())
fp_map = map(int, fp_list)
fp_2 = list(fp_map)

print(fp_1 == fp_2)
