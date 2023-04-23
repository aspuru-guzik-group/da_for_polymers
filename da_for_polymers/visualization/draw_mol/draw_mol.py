from rdkit.Chem.Draw import rdMolDraw2D
from rdkit import Chem

smi = "[*]c7cccc(n6c(=O)c5ccc(Oc4ccc(C(C)(C)c3ccc(Oc2ccc1c(=O)n(*)c(=O)c1c2)cc3)cc4)cc5c6=O)c7"
mol = Chem.MolFromSmiles(smi)
asterick_idx = []
for atom in mol.GetAtoms():
    if atom.GetSymbol() == "*":
        asterick_idx.append(atom.GetIdx())
hit_ats = Chem.rdmolops.GetShortestPath(mol, asterick_idx[0], asterick_idx[1])
print(f"{hit_ats=}")
hit_bonds = []
atom_colors = {}
bond_colors = {}
# Get bonds from shortestpath
index = 0
while index < len(hit_ats) - 1:
    a1 = hit_ats[index]
    a2 = hit_ats[index + 1]
    bond = mol.GetBondBetweenAtoms(a1, a2)
    hit_bonds.append(bond.GetIdx())
    bond_colors[bond.GetIdx()] = (0, 1, 0)
    atom_colors[a1] = (0, 1, 0)
    atom_colors[a2] = (0, 1, 0)
    index += 1
d = rdMolDraw2D.MolDraw2DCairo(1000, 1000)  # or MolDraw2DCairo to get PNGs
rdMolDraw2D.PrepareAndDrawMolecule(
    d,
    mol,
    highlightAtoms=hit_ats,
    highlightBonds=hit_bonds,
    highlightBondColors=bond_colors,
    highlightAtomColors=atom_colors,
)
d.FinishDrawing()
d.WriteDrawingText("test.png")
