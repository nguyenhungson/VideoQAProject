import json
import torch
import numpy as np
import cv2
from PIL import Image
from tqdm.auto import tqdm
from dataset import get_shared_test_loader, set_deterministic_seed

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class Qwen2VLBenchmark:
    def __init__(self):
        print("🤖 Đang khởi tạo Qwen2-VL-2B-Instruct...")
        from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
        self.model = Qwen2VLForConditionalGeneration.from_pretrained(
            "Qwen/Qwen2-VL-2B-Instruct", torch_dtype=torch.bfloat16, device_map="auto"
        ).eval()
        self.processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-2B-Instruct")

    def run(self, test_loader):
        from qwen_vl_utils import process_vision_info
        correct_preds, latencies, error_logs = 0, [], []
        starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)

        print("🔥 Warm-up Qwen2-VL (5 samples)...")
        with torch.inference_mode():
            for i, batch in enumerate(test_loader):
                if i >= 5: break
                messages = [{"role": "user", "content": [{"type": "video", "video": batch["video_path"][0], "max_pixels": 360*360, "min_pixels": 56*56, "fps": 1.0}, {"type": "text", "text": "Question"}]}]
                text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                img_in, vid_in = process_vision_info(messages)
                inputs = self.processor(text=[text], images=img_in, videos=vid_in, padding=True, return_tensors="pt").to(DEVICE)
                _ = self.model.generate(**inputs, max_new_tokens=5, do_sample=False)

        print("⚡ Đang đánh giá chân thực Qwen2-VL...")
        with torch.inference_mode():
            for batch in tqdm(test_loader, desc="Qwen2-VL Testing"):
                ground_truth = batch["answer"][0].strip().lower()
                messages = [{"role": "user", "content": [{"type": "video", "video": batch["video_path"][0], "max_pixels": 360*360, "min_pixels": 56*56, "fps": 1.0}, {"type": "text", "text": f"Answer concisely in one or two words. Question: {batch['question'][0]}"}]}]
                
                text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                img_in, vid_in = process_vision_info(messages)
                inputs = self.processor(text=[text], images=img_in, videos=vid_in, padding=True, return_tensors="pt").to(DEVICE)

                starter.record()
                generated_ids = self.model.generate(**inputs, max_new_tokens=10, do_sample=False)
                ender.record()
                torch.cuda.synchronize()

                latencies.append(starter.elapsed_time(ender))
                gen_trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]
                pred_text = self.processor.batch_decode(gen_trimmed, skip_special_tokens=True)[0].strip().lower()

                if ground_truth in pred_text:
                    correct_preds += 1
                else:
                    error_logs.append({"question": batch["question"][0], "ground_truth": ground_truth, "prediction": pred_text})

        self._print_report("QWEN2-VL-2B", correct_preds, len(test_loader.dataset), latencies)

    def _print_report(self, name, correct, total, latencies):
        print("\n" + "="*55)
        print(f"📈 BÁO CÁO NGHIỆM THU MLLM: {name}")
        print("="*55)
        print(f"🔹 Accuracy         : {(correct / total) * 100:.2f}%")
        print(f"🔹 Inference Latency: {np.mean(latencies):.2f} ms / câu")
        print(f"🔹 P99 Latency      : {np.percentile(latencies, 99):.2f} ms / câu")
        print("="*55)


class InternVL2Benchmark:
    def __init__(self):
        print("🤖 Đang khởi tạo InternVL2-2B...")
        from transformers import AutoTokenizer, AutoModel
        self.model = AutoModel.from_pretrained(
            "OpenGVLab/InternVL2-2B", torch_dtype=torch.bfloat16, low_cpu_mem_usage=True, trust_remote_code=True
        ).eval().cuda()
        self.tokenizer = AutoTokenizer.from_pretrained("OpenGVLab/InternVL2-2B", trust_remote_code=True, use_fast=False)

    def _get_pixels(self, path):
        import torchvision.transforms as T
        transform = T.Compose([T.Resize((448, 448)), T.ToTensor(), T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
        cap = cv2.VideoCapture(path)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total <= 0: return torch.zeros((8, 3, 448, 448)).to(torch.bfloat16).cuda()
        
        frames = []
        for idx in np.linspace(0, total - 1, 8, dtype=int):
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
            ret, frame = cap.read()
            frames.append(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)) if ret else Image.new("RGB", (448, 448)))
        cap.release()
        return torch.stack([transform(f) for f in frames]).to(torch.bfloat16).cuda()

    def run(self, test_loader):
        correct_preds, latencies = 0, []
        starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)

        print("🔥 Warm-up InternVL2 (5 samples)...")
        with torch.inference_mode():
            for i, batch in enumerate(test_loader):
                if i >= 5: break
                pixels = self._get_pixels(batch["video_path"][0])
                _ = self.model.chat(self.tokenizer, pixels, "<image>\n"*8 + "Q", dict(max_new_tokens=5, do_sample=False), num_patches_list=[1]*8, history=None)

        print("⚡ Đang đánh giá chân thực InternVL2...")
        with torch.inference_mode():
            for batch in tqdm(test_loader, desc="InternVL2 Testing"):
                ground_truth = batch["answer"][0].strip().lower()
                pixels = self._get_pixels(batch["video_path"][0])
                prompt = "<image>\n"*8 + f"Answer concisely in one or two words. Question: {batch['question'][0]}"

                starter.record()
                response = self.model.chat(self.tokenizer, pixels, prompt, dict(max_new_tokens=10, do_sample=False), num_patches_list=[1]*8, history=None)
                ender.record()
                torch.cuda.synchronize()

                latencies.append(starter.elapsed_time(ender))
                if ground_truth in str(response).strip().lower():
                    correct_preds += 1

        print("\n" + "="*55)
        print("📈 BÁO CÁO NGHIỆM THU MLLM: INTERNVL2-2B")
        print("="*55)
        print(f"🔹 Accuracy         : {(correct_preds / len(test_loader.dataset)) * 100:.2f}%")
        print(f"🔹 Inference Latency: {np.mean(latencies):.2f} ms / câu")
        print(f"🔹 P99 Latency      : {np.percentile(latencies, 99):.2f} ms / câu")
        print("="*55)

if __name__ == '__main__':
    set_deterministic_seed(42)
    print("📦 Đang nạp chung tập dữ liệu Test duy nhất để đối chiếu công bằng...")
    test_loader, _ = get_shared_test_loader(num_samples=1000)
    
    # Qwen2VLBenchmark().run(test_loader)
    InternVL2Benchmark().run(test_loader)