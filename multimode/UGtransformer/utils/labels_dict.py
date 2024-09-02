import pandas as pd
import torch
import os
#print(os.listdir('../../DataProcess/labels/'))
def get_lables_dict(cacer_type):
    file_path = "../../DataProcess/labels/" + cacer_type + "_subtypes.csv"
    subtypes = pd.read_csv(
        file_path,
        header=0
    )

    labels_dict = dict(zip(subtypes['pan.samplesID'], subtypes['Subtype_mRNA']))
    return labels_dict

def label_mapping(label):
    map = {'Normal': 0,'LumA':1, 'LumB':2, 'Her2':3,'Basal':4}

    return map[label]
print(get_lables_dict("brca")['TCGA-C8-A12L-01A-11R-A115-07'])