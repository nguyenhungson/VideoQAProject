import os
import json
import zipfile
import cv2
import numpy as np
import torch
import random
from torch.utils.data import Dataset, DataLoader, Subset
from tqdm.auto import tqdm
from transformers import CLIPProcessor, CLIPModel, BertTokenizer
from huggingface_hub import hf_hub_download

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
VIDEO_DIR = './videos'
FEAT_DIR =  "./data/MSRVTT-QA/video_features"
DATA_DIR = './data/MSRVTT-QA/MSRVTT-QA'
NUM_FRAMES = 8

def set_deterministic_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

class DataSetup:
    @staticmethod
    def initialize_directories():
        for d in [VIDEO_DIR, FEAT_DIR, DATA_DIR]:
            os.makedirs(d, exist_ok=True)

    @staticmethod
    def download_videos():
        if not any(f.endswith('.mp4') for f in os.listdir(VIDEO_DIR)):
            print("--- Đang tải Video từ HuggingFace... ---")
            zip_path = hf_hub_download(repo_id="friedrichor/MSR-VTT", filename="MSRVTT_Videos.zip", repo_type="dataset")
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(VIDEO_DIR)
            for root, _, files in os.walk(VIDEO_DIR):
                if root != VIDEO_DIR:
                    for f in files: os.rename(os.path.join(root, f), os.path.join(VIDEO_DIR, f))

class FeatureExtractor:
    def __init__(self, num_frames=NUM_FRAMES):
        self.num_frames = num_frames
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        self.model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(DEVICE, dtype=torch.float16).eval()

    def extract(self):
        video_files = sorted([f for f in os.listdir(VIDEO_DIR) if f.lower().endswith(('.mp4', '.avi'))])
        with torch.inference_mode():
            for v_name in tqdm(video_files, desc="Trích xuất CLIP Features"):
                v_id = os.path.splitext(v_name)[0]
                out_p = os.path.join(FEAT_DIR, f"{v_id}.pt")
                if os.path.exists(out_p): continue

                cap = cv2.VideoCapture(os.path.join(VIDEO_DIR, v_name))
                total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                if total <= 0: cap.release(); continue

                indices = np.linspace(0, total-1, self.num_frames, dtype=int)
                frames = []
                for idx in indices:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                    ret, frame = cap.read()
                    frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) if ret else np.zeros((224,224,3), dtype=np.uint8))
                cap.release()

                inputs = self.processor(images=frames, return_tensors="pt").to(DEVICE)
                inputs = {k: v.to(dtype=torch.float16) if v.is_floating_point() else v for k, v in inputs.items()}
                feats = self.model.get_image_features(**inputs)
                torch.save(feats.detach().cpu(), out_p)

class MSRVTTQADataset(Dataset):
    def __init__(self, qa_file, feat_dir, ans2idx=None):
        with open(qa_file, 'r') as f:
            self.raw = json.load(f)
        
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        self.data = []
        
        # Lọc dữ liệu hợp lệ: Video và Feature phải thực sự tồn tại
        for item in self.raw:
            vid = str(item['video_id'])
            v_name = vid if vid.startswith('video') else f"video{vid}"
            feat_path = os.path.join(feat_dir, f"{v_name}.pt")
            
            v_path = None
            for ext in ['.mp4', '.avi']:
                tmp_path = os.path.join(VIDEO_DIR, f"{v_name}{ext}")
                if os.path.exists(tmp_path):
                    v_path = tmp_path
                    break
                    
            if os.path.exists(feat_path) and v_path is not None:
                item['absolute_video_path'] = os.path.abspath(v_path)
                self.data.append(item)

        if ans2idx is None:
            unique_answers = sorted(list(set(x['answer'] for x in self.data)))
            self.ans2idx = {a: i for i, a in enumerate(unique_answers)}
        else:
            self.ans2idx = ans2idx
            self.data = [x for x in self.data if x['answer'] in self.ans2idx]

        self.idx2ans = {i: a for a, i in self.ans2idx.items()}

        self.video_features_cache = {}
        print("⚡ Đang nạp Video Features vào RAM...")
        for item in tqdm(self.data, desc="Caching"):
            v_name = os.path.basename(item['absolute_video_path']).split('.')[0]
            if v_name not in self.video_features_cache:
                self.video_features_cache[v_name] = torch.load(os.path.join(feat_dir, f"{v_name}.pt"), weights_only=True).float()

        questions = [item['question'] for item in self.data]
        self.tokenized_txt = self.tokenizer(questions, padding='max_length', truncation=True, max_length=32, return_tensors="pt")

    def __len__(self): 
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        v_name = os.path.basename(item['absolute_video_path']).split('.')[0]
        feats = self.video_features_cache[v_name]
        
        return {
            'question': item['question'],
            'answer': item['answer'],
            'video_path': item['absolute_video_path'],
            'video_features': feats,
            'input_ids': self.tokenized_txt['input_ids'][idx],
            'attention_mask': self.tokenized_txt['attention_mask'][idx],
            'label': torch.tensor(self.ans2idx[item['answer']])
        }

def get_shared_test_loader(num_samples=1000):
    train_ds = MSRVTTQADataset(os.path.join(DATA_DIR, 'train_qa.json'), FEAT_DIR)
    full_test_ds = MSRVTTQADataset(os.path.join(DATA_DIR, 'test_qa.json'), FEAT_DIR, ans2idx=train_ds.ans2idx)
    
    indices = list(range(len(full_test_ds)))
    random.seed(42)
    subset_indices = random.sample(indices, min(num_samples, len(full_test_ds)))
    
    test_ds = Subset(full_test_ds, subset_indices)
    return DataLoader(test_ds, batch_size=1, shuffle=False), train_ds.ans2idx

if __name__ == '__main__':
    set_deterministic_seed(42)
    DataSetup.initialize_directories()
    DataSetup.download_videos()
    extractor = FeatureExtractor()
    extractor.extract()