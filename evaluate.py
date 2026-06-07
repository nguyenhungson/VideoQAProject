import os
import json
import torch
import numpy as np
from tqdm.auto import tqdm
from dataset import get_shared_test_loader, set_deterministic_seed
from models import MultimodalVideoQA

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def evaluate_model():
    print("📦 Đang nạp chung một tập dữ liệu Test duy nhất...")
    test_loader, ans2idx = get_shared_test_loader(num_samples=1000)
    idx2ans_mapping = {v: k for k, v in ans2idx.items()}

    model = MultimodalVideoQA(num_classes=len(ans2idx)).to(DEVICE)
    
    ckpt_path = './checkpoints/videoqa_best.pth'
    if not os.path.exists(ckpt_path):
        ckpt_path = './checkpoints/videoqa_latest.pth'
    
    model.load_state_dict(torch.load(ckpt_path))
    model.eval()

    correct_preds, total_samples = 0, len(test_loader.dataset)
    latencies, error_logs = [], []
    starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)

    print("🔥 Khởi động Warm-up GPU...")
    with torch.inference_mode(), torch.amp.autocast('cuda', dtype=torch.bfloat16):
        for i, batch in enumerate(test_loader):
            if i >= 5: break
            _ = model(batch['video_features'].to(DEVICE), batch['input_ids'].to(DEVICE), batch['attention_mask'].to(DEVICE))

    print("⚡ Đang đánh giá Benchmark (Batch Size = 1)...")
    with torch.inference_mode():
        for batch in tqdm(test_loader, desc="Testing Baseline"):
            v_feat = batch['video_features'].to(DEVICE)
            t_ids = batch['input_ids'].to(DEVICE)
            mask = batch['attention_mask'].to(DEVICE)
            labels = batch['label'].to(DEVICE)

            starter.record()
            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                logits = model(video_feats=v_feat, input_ids=t_ids, attention_mask=mask)
            ender.record()
            torch.cuda.synchronize()

            latencies.append(starter.elapsed_time(ender))
            preds = torch.argmax(logits, dim=1)
            
            is_correct = (preds == labels).sum().item()
            correct_preds += is_correct

            if is_correct == 0:
                error_logs.append({
                    "question": batch['question'][0],
                    "ground_truth": idx2ans_mapping[labels.item()],
                    "prediction": idx2ans_mapping[preds.item()]
                })

    with open('error_analysis_baseline.json', 'w') as f:
        json.dump(error_logs, f, indent=4)

    print("\n" + "="*55)
    print("📈 KẾT QUẢ NGHIỆM THU (BASELINE CỦA BẠN)")
    print("="*55)
    print(f"🔹 Accuracy         : {(correct_preds / total_samples) * 100:.2f}%")
    print(f"🔹 Inference Latency: {np.mean(latencies):.2f} ms / câu")
    print(f"🔹 P99 Latency      : {np.percentile(latencies, 99):.2f} ms / câu")
    print("="*55)

if __name__ == '__main__':
    set_deterministic_seed(42)
    evaluate_model()