# Data Augmentation of Polymer Molecules

## Data augmentation via iterative rearrangements
1. Open `CO2_Soleimani` or `PV_Wang` or `Swelling_Xu`, go to `manual_frag/manual_frag.py`
2. Each file will have this line `#ATTENTION: Fragmenting and Order for Data Augmentation`. The code directly below this comment will iterate through your inventory of polymers and proceed to ask for the bonds to fragment, and order of fragments. Please read the instructions in the command line. NOTE: the image of the current polymer you are fragmenting will be shown in `manual.png`, `manual_frag.png`, and `manual_rearranged.png`. Have these open when fragmenting! It will show you the indices of the atoms/bonds.
3. Only after fragmentation and augmentation can you preprocess the augmented data.

## Recombining augmented fragments
4. For recombination, you'll have to pay special attention to `manual_rearranged.png`. The workflow should automatically recombine fragments if the molecule was A) fragmented correctly along the linear backbone, and B) ordered correctly such that the chemical connectivity is preserved. Please refer to `Figure 2` in the paper or `TOC2.png` in the repository for reference.

## Augment your own molecules?
You will need to change how the data is loaded into the `manual_frag/manual_frag.py`
You will need to change:

    manual = manual_frag(PV_INVENTORY)
    # ATTENTION: Fragmenting and Order for Data Augmentation
    # iterate through donor and acceptor files
    manual_df = pd.read_csv(PV_INVENTORY)

Instead of it reading `PV_INVENTORY`, you can input your own `.csv` of molecules. Make sure you acquaint yourself with the inventory file in `da_for_polymers/data/preprocess/*_inventory.csv`. The column names will be important to keep track of for your file.