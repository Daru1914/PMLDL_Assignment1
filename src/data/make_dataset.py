# Unzipping the dataset in the raw directory:
import zipfile
import pandas as pd

with zipfile.ZipFile("../data/raw/filtered_paranmt.zip", mode="r") as archive:
    archive.printdir()

with zipfile.ZipFile("../data/raw/filtered_paranmt.zip", mode="r") as archive:
    dataset = archive.read("filtered.tsv").decode(encoding="utf-8")
    with open("../data/interim/filtered_paranmt.tsv", "w", encoding="utf-8") as f:
        f.write(dataset)

dataset = pd.read_csv("../data/interim/filtered_paranmt.tsv", delimiter='\t')
dataset = dataset.set_index(dataset.columns[0])
dataset.index.name = "Index"

# Create new columns, strictly toxic and non-toxic
dataset['toxic'] = dataset.apply(lambda row: row['translation'] if row['ref_tox'] < row['trn_tox'] else row['reference'], axis=1)
dataset['non-toxic'] = dataset.apply(lambda row: row['reference'] if row['ref_tox'] < row['trn_tox'] else row['translation'], axis=1)
dataset['old_toxicity'] = dataset.apply(lambda row: row['trn_tox'] if row['ref_tox'] < row['trn_tox'] else row['ref_tox'], axis=1)
dataset['new_toxicity'] = dataset.apply(lambda row: row['ref_tox'] if row['ref_tox'] < row['trn_tox'] else row['trn_tox'], axis=1)
dataset['toxic'] = dataset['toxic'].str.lower()
dataset['non-toxic'] = dataset['non-toxic'].str.lower()
dataset.drop(['reference', 'translation', 'similarity', 'lenght_diff', 'ref_tox', 'trn_tox'], axis=1, inplace=True)

dataset.to_csv("../data/interim/separated_tox.csv")