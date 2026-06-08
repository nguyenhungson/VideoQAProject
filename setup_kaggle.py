import json
import os
import subprocess
import zipfile

def setup_kaggle_on_runpod():
    print("Đang khởi tạo môi trường Kaggle trên RunPod...")

    # 1. Cấu hình credential
    kaggle_dir = os.path.expanduser("~/.kaggle")
    os.makedirs(kaggle_dir, exist_ok=True)

    kaggle_credentials = {
        "username": "snnguyn9898",
        "key": "eca27348b3843512ba2d1a35db01818e",
    }

    cred_path = os.path.join(kaggle_dir, "kaggle.json")

    with open(cred_path, "w") as f:
        json.dump(kaggle_credentials, f)

    # Đặt quyền bảo mật (bắt buộc cho Kaggle API)
    os.chmod(cred_path, 0o600)
    print("Đã thiết lập kaggle.json thành công.")

    # 2. Kiểm tra và tải Dataset
    dataset_name = "valerytamrazov/msrvttqa"
    data_dir = "./data"
    os.makedirs(data_dir, exist_ok=True)

    print(f"📥 Đang tải dataset: {dataset_name}...")

    try:
        subprocess.run(
            [
                "kaggle",
                "datasets",
                "download",
                "-d",
                dataset_name,
                "-p",
                data_dir,
            ],
            check=True,
        )

        zip_path = os.path.join(data_dir, "msrvttqa.zip")

        if os.path.exists(zip_path):
            print("Đang giải nén dữ liệu...")

            with zipfile.ZipFile(zip_path, "r") as zip_ref:
                zip_ref.extractall(data_dir)

            os.remove(zip_path)
            print(f"Dữ liệu đã sẵn sàng tại {data_dir}")
        else:
            print("Lỗi: Không tìm thấy file zip sau khi tải.")

    except subprocess.CalledProcessError as e:
        print(f"Lỗi khi tải Kaggle: {e}")
        print(
            "Gợi ý: Kiểm tra lại kết nối mạng hoặc quyền truy cập dataset trên Kaggle."
        )

if __name__ == "__main__":
    setup_kaggle_on_runpod()