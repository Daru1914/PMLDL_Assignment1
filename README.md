# PMLDL_Assignment1
This repository contains the solution to detoxification problem poised during the Practical Machine Learning and Deep Learning course in Innopolis University. Dataset used: ParaNMT-detox corpus (500K sentence pairs) (https://github.com/skoltech-nlp/detox/releases/download/emnlp2021/filtered_paranmt.zip).

Folder src contains scripts that should be useful to anyone wanting to use the models for prediction and training:
src/data/make_dataset.py is necessary to unpack from the archive and reform the original dataset;  
src/models/predict_condbert.py is necessary to predict with pre-trained CondBERT;  
src/models/predict_paragedi.py is necessary to predict with pre-trained paraGeDi;  
src/models/train_condbert.py is necessary to fine-tune and predict with pre-trained CondBERT;  
