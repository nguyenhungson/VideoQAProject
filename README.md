***

```markdown
# 🎥 Video Question Answering (Video QA) - MSR-VTT

Dự án này xây dựng một hệ thống Trả lời câu hỏi trên Video (Video QA) thuộc nhóm bài toán phân loại đa lớp. Hệ thống nhận đầu vào là một đoạn video ngắn cùng một câu hỏi văn bản, từ đó sử dụng cơ chế **Cross-Attention** để suy luận và chọn ra câu trả lời chính xác nhất.

Đặc biệt, dự án bao gồm một Pipeline nghiệm thu công bằng (**Fair Benchmark Protocol**) để đối chiếu trực tiếp hiệu năng của mô hình tự code (Baseline) với các siêu mô hình ngôn ngữ đa phương thức (MLLMs) SOTA hiện nay như **Qwen2-VL** và **InternVL2**.

---

## 🛠 Yêu cầu Hệ thống
- **Hệ điều hành**: Linux (Ubuntu 22.04+ được khuyến nghị).
- **Phần cứng**: GPU NVIDIA (Khuyến nghị VRAM >= 15GB để tải MLLMs).
- **Nền tảng**: Docker (Hỗ trợ CUDA 12.4).

## 📁 Cấu trúc Thư mục
```text
.
├── dataset.py            # Code xử lý dữ liệu, tokenize, cache RAM & trích xuất CLIP Features
├── models.py             # Kiến trúc mạng chính (Cross-Attention, VideoQAClassifier)
├── train.py              # Vòng lặp huấn luyện Baseline (Mixed Precision, CosineAnnealing)
├── evaluate.py           # Code nghiệm thu Baseline model (Tính Accuracy, Latency)
├── benchmark_mllms.py    # Code nghiệm thu MLLMs (Qwen2-VL, InternVL2)
├── setup_kaggle.py       # Tải tự động Dataset từ Kaggle
├── Dockerfile            # Môi trường Container (CUDA 12.4 + Flash-Attention)
├── requirements.txt      # Danh sách thư viện Python phụ thuộc
└── README.md             # Tài liệu kỹ thuật dự án
```

---

## 🚀 Hướng dẫn Cài đặt & Triển khai

### 1. Build Môi trường Docker
Chúng tôi sử dụng Base Image của NVIDIA với lõi CUDA 12.4 để biên dịch các thư viện cấp thấp một cách an toàn. Tại thư mục chứa dự án, chạy lệnh:
```bash
docker build -t video_qa_cu124 .
```

### 2. Khởi chạy Container
**Lưu ý:** Bắt buộc sử dụng flag `--shm-size=16g` để ngăn chặn lỗi tràn bộ nhớ (Out of Memory) của PyTorch DataLoader trên môi trường Linux khi chia batch lớn.
```bash
docker run --gpus all -it --rm \
    -v $(pwd):/app \
    --shm-size=16g \
    video_qa_cu124
```

---

## ⚙️ Quy trình Vận hành (Pipeline)

Toàn bộ các lệnh dưới đây được chạy **bên trong Docker Container**.

### Bước 1: Chuẩn bị Dữ liệu
Hệ thống sẽ tải toàn bộ tập annotations và Video thô (MSR-VTT) về máy.
```bash
python setup_kaggle.py
```

### Bước 2: Tiền xử lý & Trích xuất Đặc trưng (Feature Extraction)
Script sử dụng mô hình pre-trained `CLIP ViT-B/32` trích xuất đều 8 khung hình (frames) cho mỗi video và lưu thành các file Tensor (`.pt`).
```bash
python dataset.py
```

### Bước 3: Huấn luyện Mô hình Baseline
Quá trình huấn luyện áp dụng các kỹ thuật tối ưu hóa hiện đại: **Mixed Precision Training** (`torch.amp`), **Gradient Clipping** chống nổ vi phân, và **CosineAnnealingLR** scheduler. Trọng số có validation loss thấp nhất tự động được lưu vào `./checkpoints/videoqa_best.pth`.
```bash
python train.py
```

### Bước 4: Đánh giá Mô hình Baseline
Tiến hành đo lường **Độ chính xác (Accuracy)** và **Tốc độ suy luận (Inference Latency)** sử dụng cơ chế `torch.cuda.Event` với cấu hình thiết lập chuẩn Benchmark (`Batch Size = 1`).
```bash
python evaluate.py
```

### Bước 5: Đối chiếu (Benchmark) với MLLMs
Chạy nghiệm thu mô hình **Qwen2-VL-2B-Instruct** và **InternVL2-2B**. Cả Baseline và MLLMs đều được đánh giá trên **cùng một tập Subset dùng chung** (chia sẻ qua hàm `get_shared_test_loader`) nhằm loại bỏ hoàn toàn sai số chọn mẫu, đem lại sự công bằng kỹ thuật cao nhất.
```bash
python benchmark_mllms.py
```

---

## 🧠 Các Điểm Nhấn Kỹ Thuật Ghi Nhận (Key Technical Optimizations)

1. **Masked Mean Pooling**: Tính toán ma trận Attention Mask chính xác đến từng token, triệt tiêu nhiễu từ các padding token sinh ra bởi BertTokenizer.
2. **RAM Caching DataLoader**: Đưa toàn bộ hàng chục ngàn vector hình ảnh lên RAM từ lúc khởi tạo `Dataset`. Tốc độ đọc dữ liệu ($I/O$) rút ngắn tuyệt đối về mức $O(1)$.
3. **Cross Attention tự xây dựng (From Scratch)**: Module được code thủ công thông qua `torch.bmm`, liên kết chặt chẽ thông tin Tầm nhìn (Video Features) với Ngôn ngữ (Text Features).
4. **Fair Evaluation Protocol**: Một API truy xuất Dữ liệu dùng chung duy nhất cho mọi mô hình tham gia nghiệm thu, áp dụng bộ tạo ngẫu nhiên hạt giống cố định (Deterministic Seed).
```
