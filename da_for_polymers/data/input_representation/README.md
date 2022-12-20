# Molecular Representations

## Data augmentation via iterative rearrangements
1. Open `Dataset`, go to `manual_frag.py`
2. Each file will have this line `# ATTENTION: Fragmenting and Order for Data Augmentation`. The code directly below this comment will iterate through your inventory of polymers and proceed to ask for the bonds to fragment, and order of fragments. Please read the instructions in the command line. NOTE: the image of the polymer will be shown in `manual.png`, `manual_frag.png`, and `manual_rearranged.png`. Have these open when fragmenting! It will show you the indices of the atoms/bonds.
3. Only after fragmentation and augmentation can you preprocess the augmented data.

## Recombining augmented fragments
4. For recombination, you'll have to pay special attention to `manual_rearranged.png`. The workflow should automatically recombine fragments if the molecule was A) fragmented correctly along the linear backbone, and B) ordered correctly such that the chemical connectivity is preserved. Please refer to `Figure 2` in the paper or `TOC2.png` in the repository for reference.