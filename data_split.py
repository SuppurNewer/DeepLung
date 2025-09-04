import os
import pickle
from sklearn.model_selection import KFold
import numpy as np

path = r"data/atudio_data"
filename = [i.split('.')[0] for i in os.listdir(path)]
person_to_files = {}
for idex, lab in enumerate(filename):
    person_to_files.setdefault(lab.split('_')[1], []).append(lab)
   
all_names = np.array(list(person_to_files.keys()))

save_folder = f"data_split"
os.makedirs(save_folder, exist_ok=True)

kfold = KFold(n_splits=5, shuffle=True, random_state=31)
for fold_idx, (train_idx, val_idx) in enumerate(kfold.split(all_names)):
    trainset, testset = all_names[train_idx], all_names[val_idx]
    
    fold_data = {
    'train': trainset,
    'test': testset,
    'samples': person_to_files,
}

    print(f"Fold {fold_idx}: Train Size = {len(trainset)}, Validation Size = {len(testset)}")

    with open(os.path.join(save_folder, f"fold_{fold_idx}.pkl"), 'wb') as f:
        pickle.dump(fold_data, f)
# print(fold_data)
