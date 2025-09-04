import os
import torch
from transformers import BertTokenizer, BertModel
from tqdm import tqdm
import numpy as np

# 使用中文 BERT
tokenizer = BertTokenizer.from_pretrained("bert-base-chinese")
model = BertModel.from_pretrained("bert-base-chinese")
model.eval()  # 推理模式

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# 文本文件路径
txt_folder = r"H:\DeepLung\CT\diagnosis_data"
output_vectors = []

for file in tqdm(os.listdir(txt_folder)):
    if not file.endswith(".txt"):
        continue
    
    path = os.path.join(txt_folder, file)
    with open(path, "r", encoding="utf-8") as f:
        line = f.readline().strip()
        if not line:
            continue

    # 编码文本
    inputs = tokenizer(line, return_tensors="pt", truncation=True, max_length=128)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = model(**inputs)
        cls_embedding = outputs.last_hidden_state[:, 0, :]  # [CLS] token 的表示 (1, 768)
        vector = cls_embedding.squeeze().cpu().numpy()  # (768,)

        np.save(os.path.join("H:\DeepLung\CT\diagnosis_emd_data",f"{file.split('.')[0]}.npy"), vector)  # shape = (205, 768)

