import os
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm
import clip
from dataloader import CLIPDataset
from loss import triplet_loss
from clip_gcn_image_feature_integration import ModifiedCLIP
from evaluate import evaluate
import torch.nn.functional as F
from termcolor import colored

# 设置训练参数
DATASET_PATH = '../data/RSITMD/images'
TRAIN_FILENAME = '../data/RSITMD/train_filename.txt'
TRAIN_CAPS = '../data/RSITMD/train_caps.txt'
TRIPLES_FOLDER_PATH = './triples'
DET_JSON_FOLDER_PATH = './det_json'
BATCH_SIZE = 64
EPOCHS = 100
LR = 1e-4
MARGIN = 0.2
MAX_VIOLATION = False
GRAD_CLIP = 1.0
VAL_SPLIT = 0.2
IMAGE_ENCODER_RATIO = (1, 0)
TEXT_ENCODER_RATIO = (0.5, 0.5)
USE_PRETRAINED = True

BEST_MR = 0.0  # 用于保存最优模型
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# 加载模型
model = ModifiedCLIP(
    embed_dim=512,
    image_resolution=224,
    vision_layers=12,
    vision_width=768,
    vision_patch_size=32,
    context_length=77,
    vocab_size=49408,
    transformer_width=512,
    transformer_heads=8,
    transformer_layers=12,
    image_encoder_ratio=IMAGE_ENCODER_RATIO,
    text_encoder_ratio=TEXT_ENCODER_RATIO,
    use_pretrained=USE_PRETRAINED
).to(device)

# 数据变换
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711])
])

# 加载数据集
dataset = CLIPDataset(
    image_dir=DATASET_PATH,
    filename_path=TRAIN_FILENAME,
    caption_path=TRAIN_CAPS,
    triples_dir=TRIPLES_FOLDER_PATH,
    det_json_dir=DET_JSON_FOLDER_PATH,
    transform=transform
)

# 划分训练集和验证集
train_size = int((1 - VAL_SPLIT) * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

# 优化器
optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=LR)

# 开始训练
print("Starting training...")
for epoch in range(EPOCHS):
    model.train()
    total_loss = 0.0

    for batch_idx, batch in enumerate(tqdm(train_loader, desc=f"Epoch {epoch + 1}/{EPOCHS}")):
        anchor_images = batch['image'].to(device)
        positive_texts = [text for text in batch['text']]
        positive_texts = clip.tokenize(positive_texts).to(device)

        adjacency_matrices_image = batch['adj_matrix_image'].to(device).float()
        degree_matrices_image = batch['degree_matrix_image'].to(device).float()
        adjacency_matrices_text = batch['adj_matrix_text'].to(device).float()
        degree_matrices_text = batch['degree_matrix_text'].to(device).float()

        optimizer.zero_grad()

        image_features = model.encode_image(anchor_images, adjacency_matrices_image, degree_matrices_image)
        text_features = model.encode_text(positive_texts, adjacency_matrices_text, degree_matrices_text)

        image_features = F.normalize(image_features, dim=-1)
        text_features = F.normalize(text_features, dim=-1)

        scores = torch.matmul(image_features, text_features.T)

        loss = triplet_loss(scores, MARGIN, max_violation=MAX_VIOLATION)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        optimizer.step()

        total_loss += loss.item()
        if batch_idx % 10 == 0:
            print(f"Batch {batch_idx} - Loss: {loss.item():.4f}")

    avg_loss = total_loss / len(train_loader)
    print(f"Epoch {epoch + 1}, Average Loss: {avg_loss:.4f}")

    # 评估模型
    print(colored("\nEvaluating model...", "yellow"))
    recall_metrics, avg_mr = evaluate (model, val_loader, device)
    print(colored(f"Evaluation Metrics: {recall_metrics}", "cyan"))
    print(colored(f"Mean Recall (MR): {avg_mr:4f}", "cyan"))

    # 保存最优模型
    if avg_mr > BEST_MR:
        BEST_MR = avg_mr
        torch.save(model.state_dict(), "best_model.pth")
        print(colored(f"Best model saved with MR: {BEST_MR}", "green"))

print("Training completed.")
