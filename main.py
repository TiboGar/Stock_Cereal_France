import famDataset
import os

fam = famDataset.famDataset(os.getcwd())
fam.download_raw_data()
fam.save_transformed_data()
