import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from dataset import MSRVTTQADataset, DATA_DIR, FEAT_DIR, set_deterministic_seed
from models import MultimodalVideoQA

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def train():
    train_ds = MSRVTTQADataset(os.path.join(DATA_DIR, 'train_qa.json'), FEAT_DIR)
    train_loader = DataLoader(train_ds, batch_size=512, shuffle=True, num_workers=4, pin_memory=True)
    
    model = MultimodalVideoQA(num_classes=len(train_ds.ans2idx)).to(DEVICE)
    model.train()

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4, weight_decay=1e-4)
    
    num_epochs = 50 
    scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs)
    scaler = torch.amp.GradScaler('cuda')

    os.makedirs('./checkpoints', exist_ok=True)
    best_loss = float('inf') 

    print("BẮT ĐẦU HUẤN LUYỆN...")
    for epoch in range(num_epochs):
        total_loss, correct_preds, total_samples = 0.0, 0, 0
        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")

        for batch in loop:
            video_feats = batch['video_features'].to(DEVICE, non_blocking=True)
            input_ids = batch['input_ids'].to(DEVICE, non_blocking=True)
            attention_mask = batch['attention_mask'].to(DEVICE, non_blocking=True)
            labels = batch['label'].to(DEVICE, non_blocking=True)

            optimizer.zero_grad()
            with torch.amp.autocast('cuda'):
                logits = model(video_feats=video_feats, input_ids=input_ids, attention_mask=attention_mask)
                loss = criterion(logits, labels)
            
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            # BỔ SUNG: Chống nổ vi phân (Exploding Gradient)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item()
            preds = torch.argmax(logits, dim=1)
            correct_preds += (preds == labels).sum().item()
            total_samples += labels.size(0)

            loop.set_postfix(loss=f"{loss.item():.4f}", acc=f"{(correct_preds/total_samples)*100:.2f}%")
            
        scheduler.step()

        avg_loss = total_loss / len(train_loader)
        torch.save(model.state_dict(), './checkpoints/videoqa_latest.pth')
        
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), './checkpoints/videoqa_best.pth')

if __name__ == '__main__':
    set_deterministic_seed(42)
    train()