import pandas as pd

dataset = pd.read_csv("../data/interim/separated_tox.csv")
dataset = dataset.set_index(dataset.columns[0])
dataset.index.name = "Index"
dataset.head()

from sklearn.model_selection import train_test_split

train_dataset, test_dataset = train_test_split(dataset, test_size=0.2, random_state=36)

with open('../data/interim/toxic1.txt', 'w', encoding='utf-8') as file:
    for item in test_dataset["toxic"]:
        file.write("%s\n" % item)

train_dataset_1, test_dataset_1 = train_test_split(dataset, test_size=0.2, random_state=49)
reformed_td = test_dataset_1[:3000]

with open('../data/interim/toxic2.txt', 'w', encoding='utf-8') as file:
    for item in reformed_td["toxic"]:
        file.write("%s\n" % item)

# results of the fine-tuned CondBERT
!python ../detox/emnlp2021/metric/metric.py --inputs ../data/interim/toxic1.txt --preds ../data/results/results3.txt

# results of the paraGeDi
!python ../detox/emnlp2021/metric/metric.py --inputs ../data/interim/toxic2.txt --preds ../data/results/results2.txt

# results of the original CondBERT
!python ../detox/emnlp2021/metric/metric.py --inputs ../data/interim/toxic1.txt --preds ../data/results/results1.txt