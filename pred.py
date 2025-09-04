import torch
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix,classification_report
from tqdm import tqdm
import pickle
import numpy as np
from model import AudioStudent_vgg
from train import LungDataset


def validate_only(fold=0, ckpt_path=None):
    paths = {
        "lung_precess_path": "data/signal_data",
        "lung_CT_path": "data/ct_data",
        "lung_label_path": "data/label_data",
        "lung_idx_path": "data/basic_data",
        "lung_rep_path": "data/diagnosis_emd_data",
    }

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    with open(f"data_split/fold_{fold}.pkl", "rb") as file:
        data = pickle.load(file)

    val_files = [f for p in data["test"] for f in data["samples"][p]]
    val_set = LungDataset(val_files, **paths)
    val_loader = DataLoader(val_set, batch_size=8, shuffle=False, num_workers=4, pin_memory=True)

    model = AudioStudent_vgg().to(device)

    if ckpt_path:
        model.load_state_dict(torch.load(ckpt_path, map_location=device))
        print(f"Loaded weights from: {ckpt_path}")

    model.eval()
    preds, targets = [], []
    probs_list = []
    
    with torch.no_grad():
        for batch in tqdm(val_loader, desc="[Validation Only]"):
            x_stft, x_wave, x_mel, _, _, _, labels = [x.to(device) for x in batch]
            label_student = (labels != 0).long()

            logits, _ = model(x_stft, x_wave, x_mel)
            pred = logits.argmax(dim=1)

            preds.extend(pred.tolist())
            targets.extend(label_student.tolist())
            
    acc = accuracy_score(targets, preds)
    prec_macro = precision_score(targets, preds, average='macro')
    rec_macro = recall_score(targets, preds, average='macro')
    f1_macro = f1_score(targets, preds, average='macro')

    # 计算敏感度 (Disease Recall = 对正类的召回率)
    sensitivity = recall_score(targets, preds, pos_label=1)

    # 计算特异度 (Normal Recall = 对负类的召回率)
    # 也可以用混淆矩阵计算特异度
    tn, fp, fn, tp = confusion_matrix(targets, preds).ravel()
    specificity = tn / (tn + fp)
    print("\nClassification Report:")
    print(classification_report(targets, preds, target_names=["正常", "疾病"]))
    print(f"F1 Score: {f1_score(targets, preds, average='binary'):.4f}")

    auc_score = roc_auc_score(targets, probs_list)

    fold_results = {
    'Fold': fold,
    'Accuracy': acc,
    'Macro Precision': prec_macro,
    'Macro Recall': rec_macro,
    'Macro F1-score': f1_macro,
    'Sensitivity (Disease Recall)': sensitivity,
    'Specificity (Normal Recall)': specificity,
    'AUC': auc_score,
}
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Validate Student Model Only")
    parser.add_argument("--fold", type=int, default=0, help="Fold number")
    parser.add_argument("--ckpt", type=str, default=None, help="Checkpoint path")
    args = parser.parse_args()

    validate_only(fold=args.fold, ckpt_path=args.ckpt)
