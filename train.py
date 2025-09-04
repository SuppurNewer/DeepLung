import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch import optim
from sklearn.metrics import f1_score
from tqdm import tqdm
import numpy as np
import pickle
import tifffile
import os
from model import TeacherClassifier
from model import AudioStudent_vgg

class LungDataset(Dataset):
    def __init__(self, samples, lung_precess_path, lung_CT_path, lung_label_path, lung_idx_path, lung_rep_path):
        self.samples = samples
        self.paths = {
            'precess': lung_precess_path,
            'ct': lung_CT_path,
            'label': lung_label_path,
            'index': lung_idx_path,
            'rep': lung_rep_path,
        }

    def __getitem__(self, index):
        file_name = self.samples[index]
        patient_id = file_name.split('_')[1]
        label = self._load_label(f"{self.paths['label']}/{patient_id}.txt")

        stft = self._load_and_norm(f"{self.paths['precess']}/{file_name}_STFT.npy")
        wave = self._load_and_norm(f"{self.paths['precess']}/{file_name}_wavelet.npy")
        mel = self._load_and_norm(f"{self.paths['precess']}/{file_name}_mel.npy")

        if label == 0:
            index_data_ = torch.zeros(6, dtype=torch.float32)
            rep_ = torch.zeros(768, dtype=torch.float32)
            ct_data_ = torch.zeros(1, 64, 256, 256, dtype=torch.float32)
            return stft.unsqueeze(0), wave.unsqueeze(0), mel.unsqueeze(0), index_data_, ct_data_, rep_, label

        else:
            index_data = self._load_txt(f"{self.paths['index']}/{patient_id}.txt")
            rep = self._load_and_norm(f"{self.paths['rep']}/{patient_id}.npy")
            ct_data = self._load_ct(f"{self.paths['ct']}/{patient_id}.tif")
            return stft.unsqueeze(0), wave.unsqueeze(0), mel.unsqueeze(0), index_data, ct_data, rep, label

    def __len__(self):
        return len(self.samples)

    def _load_and_norm(self, path):
        data = np.load(path)
        return torch.tensor((data - data.min()) / (data.max() - data.min() + 1e-8), dtype=torch.float32)

    def _load_txt(self, path):
        values = np.loadtxt(path)
        values = (values - values.min()) / (values.max() - values.min() + 1e-8)
        return torch.tensor(values, dtype=torch.float32)

    def _load_ct(self, path):
        ct = tifffile.imread(path).astype(np.float32)
        ct = (ct - ct.min()) / (ct.max() - ct.min() + 1e-8)
        ct = torch.tensor(ct).unsqueeze(0).unsqueeze(0)
        ct = F.interpolate(ct, size=(64, 256, 256), mode='trilinear', align_corners=False)
        return ct.squeeze(0)

    def _load_label(self, path):
        return torch.tensor(int(open(path).read().strip().split()[0]), dtype=torch.long)


class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits, targets):
        # logits: (B, 2), targets: (B,) float
        bce_loss = F.binary_cross_entropy_with_logits(logits[:, 1], targets, reduction='none')  # 只取“正类”logit
        pt = torch.exp(-bce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * bce_loss
        return focal_loss.mean()

def distillation_loss(student_logits, teacher_probs, temperature=4.0):
    p = F.log_softmax(student_logits / temperature, dim=1)
    q = teacher_probs + 1e-8
    return F.kl_div(p, q, reduction='batchmean') * (temperature ** 2)


def validate(model, model2, val_loader, device):
    model.eval()
    model2.eval()
    student_preds, student_targets = [], []

    with torch.no_grad():
        for batch in tqdm(val_loader, desc="[Validation]"):
            x_stft, x_wave, x_mel, x_index, x_ct, x_rep, labels = [x.to(device) for x in batch]
            label_student = (labels != 0).long()

            logits, _ = model2(x_stft, x_wave, x_mel)
            preds = logits.argmax(dim=1)
            student_preds.extend(preds.tolist())
            student_targets.extend(label_student.tolist())

    f1 = f1_score(student_targets, student_preds, average='binary')
    print(f"Validation F1: {f1:.4f}")
    return f1


def train():
    paths = {
        "lung_precess_path": "data/signal_data",
        "lung_CT_path": "data/ct_data",
        "lung_label_path": "data/label_data",
        "lung_idx_path": "data/basic_data",
        "lung_rep_path": "data/diagnosis_emd_data",
    }

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    for fold in range(5):
        print(f"\n--- Starting Fold {fold} ---")
        with open(f"data_split/fold_{fold}.pkl", 'rb') as file:
            data = pickle.load(file)

        train_files = [f for p in data['train'] for f in data['samples'][p]]
        val_files = [f for p in data['test'] for f in data['samples'][p]]

        train_set = LungDataset(train_files, **paths)
        val_set = LungDataset(val_files, **paths)

        train_loader = DataLoader(train_set, batch_size=16, shuffle=True, num_workers=4, pin_memory=True)
        val_loader = DataLoader(val_set, batch_size=16, shuffle=False, num_workers=4, pin_memory=True)

        model = TeacherClassifier().to(device)
        model2 = AudioStudent_vgg().to(device)
        student_ckpt = f"best_student_vgg_fold_{fold}.pth"  # 你保存的路径
        if os.path.exists(student_ckpt):
            
            model2.load_state_dict(torch.load(student_ckpt, map_location=device))
        else:
            print(f">> No pretrained weights found at {student_ckpt}, training from scratch.")

        criterion_teacher_hard = nn.CrossEntropyLoss()
        criterion_student_hard = FocalLoss(alpha=1.5641,gamma=2.0)

        optimizer = optim.Adam(list(model.parameters()) + list(model2.parameters()), lr=1e-4, weight_decay=1e-5)
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[4], gamma=0.1)

        best_val_f1_student, patience, early_stop = 0.0, 20, 0

        for epoch in range(100):
            model.train()
            model2.train()
            train_losses = []
            student_preds, student_targets = [], []

            for batch in tqdm(train_loader, desc=f"Fold {fold} Epoch {epoch+1} [Train]"):
                optimizer.zero_grad()

                x_stft, x_wave, x_mel, x_index, x_ct, x_rep, label_teacher = [x.to(device) for x in batch]
                label_student = (label_teacher != 0).long()
                is_disease = label_teacher != 0
                is_normal = ~is_disease

                student_logits, student_feat = model2(x_stft, x_wave, x_mel)
                loss = criterion_student_hard(student_logits, label_student.float())

                if is_disease.sum() > 0:
                    teacher_logits, teacher_feat = model(
                        x_stft[is_disease], x_wave[is_disease], x_mel[is_disease],
                        x_index[is_disease], x_ct[is_disease], x_rep[is_disease])

                    loss += 0.5 * criterion_teacher_hard(teacher_logits, label_teacher[is_disease])

                    with torch.no_grad():
                        teacher_probs = F.softmax(teacher_logits[:, [0, 1]] / 4.0, dim=1)

                    loss += 0.5 * distillation_loss(student_logits[is_disease], teacher_probs, temperature=4.0)
                    loss += 0.5 * F.mse_loss(student_feat[is_disease], teacher_feat.detach())
                
                if is_normal.sum() > 0:
                    # 合成一个固定soft标签（0.95正常，0.05疾病）
                    soft_label_normal = torch.tensor([0.95, 0.05], device=device).unsqueeze(0).repeat(is_normal.sum(), 1)
                    loss += 0.1 * distillation_loss(student_logits[is_normal], soft_label_normal, temperature=1.0)
                
                loss.backward()
                optimizer.step()
                train_losses.append(loss.item())

                preds = student_logits.argmax(dim=1)
                student_preds.extend(preds.tolist())
                student_targets.extend(label_student.tolist())

            scheduler.step()
            train_f1 = f1_score(student_targets, student_preds, average='binary')
            print(f"[Epoch {epoch+1}] Train Loss: {np.mean(train_losses):.4f} | Train F1: {train_f1:.4f}")

            val_f1 = validate(model, model2, val_loader, device)

            if val_f1 > best_val_f1_student:
                best_val_f1_student = val_f1
                early_stop = 0
                torch.save(model2.state_dict(), f"best_fold_{fold}.pth")
                print(">> Saved best student model.")
            else:
                early_stop += 1
                if early_stop >= patience:
                    print(">> Early stopping.")
                    break


if __name__ == '__main__':
    train()
