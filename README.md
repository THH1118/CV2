# CV2
CVPDL HW2 – Aerial Object Detection (YOLOv9)

偵測 car / hov / person / motorcycle（空拍影像）。本 README 一次性說明：環境安裝 → 資料整理 → 訓練 → 推論 / 產生 submission。
最終成績：Private 0.28240、Public 0.25088。

目錄

環境安裝

資料準備

專案結構

訓練

推論與 Submission

結果摘要

可重現性

注意事項

授權

環境安裝
選項 A. venv（建議）
python -m venv .venv
# Windows
.\.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate

python -m pip install --upgrade pip wheel setuptools
pip install ultralytics opencv-python torch torchvision numpy pandas matplotlib tqdm PyYAML


若需 GPU，請依你的 CUDA 版本安裝相容的 torch/torchvision。

選項 B. Conda
conda create -n cvpd hw2 python=3.10 -y
conda activate cvpd
pip install ultralytics opencv-python torch torchvision numpy pandas matplotlib tqdm PyYAML

資料準備

下載競賽資料（Kaggle）

# 設定好 kaggle.json 後：
export KAGGLE_CONFIG_DIR=/path/to/your/kaggle   # Windows 可用 set
kaggle competitions download -c taica-cvpdl-2025-hw-2 -p ./data
unzip -q ./data/taica-cvpdl-2025-hw-2.zip -d ./data


整理成 YOLO 結構（示意）

data/
  CVPDL_hw2/
    images/
      train1280/
      val1280/
      test/              # 如主辦方提供
    labels/
      train1280/
      val1280/


標註需為 YOLO txt：class xc yc w h（相對比例；中心座標）。

建立 data.yaml

# hw2_data.yaml
path: ./data/CVPDL_hw2
train: images/train1280
val:   images/val1280
nc: 4
names: [car, hov, person, motorcycle]

專案結構
.
├─ hw2_data.yaml
├─ train.py                 # (可選) 你包好的訓練腳本；或直接用 yolo CLI
├─ infer_to_txt.py          # (可選) 推論並輸出 YOLO txt（含 conf）
├─ make_submission.py       # (可選) 將 txt 合併為比賽需要的 CSV
└─ data/
   └─ CVPDL_hw2/...

訓練

本作業採 Ultralytics YOLOv9，影像大小 imgsz=1280。

yolo detect train \
  model=yolov9m.yaml \
  data=hw2_data.yaml \
  epochs=300 imgsz=1280 batch=2 device=0 \
  optimizer=AdamW lr0=0.00075 lrf=0.01 cos_lr=True warmup_epochs=5 \
  mosaic=1.0 mixup=0.15 copy_paste=0.25 erasing=0.4 \
  degrees=10 scale=0.5 translate=0.1 close_mosaic=10 \
  pretrained=False \
  project=runs_hw2 name=yolov9_scratch


輸出權重：runs_hw2/detect/yolov9_scratch/weights/best.pt

推論與 Submission
1) 推論（Predict）
yolo detect predict \
  model=runs_hw2/detect/yolov9_scratch/weights/best.pt \
  source=./data/CVPDL_hw2/test \
  imgsz=1280 device=0 \
  save_txt=True save_conf=True


YOLO 會在 runs/detect/predict*/labels/ 產生每張圖的 *.txt（class xc yc w h conf）。

2) 產生 Submission（請依賽制欄位調整）

以下範例把 YOLO txt 合併為 CSV；若比賽要 像素座標 xyxy，請先依影像大小把相對座標還原（此處僅示範讀檔與合併，欄位你可調整）。

# make_submission.py
from pathlib import Path
import pandas as pd

LABELS_DIR = Path("runs/detect/predict/labels")  # 換成你的 predict 路徑
NAME_MAP = {0:"car", 1:"hov", 2:"person", 3:"motorcycle"}

rows = []
for p in sorted(LABELS_DIR.glob("*.txt")):
    image_name = p.stem + ".png"  # 若實際為 .jpg 就改 .jpg
    for line in p.read_text().strip().splitlines():
        cid, xc, yc, w, h, conf = map(float, line.split())
        rows.append([image_name, int(cid), xc, yc, w, h, conf])

df = pd.DataFrame(rows, columns=["image_name","class_id","xc","yc","w","h","score"])
df.to_csv("submission.csv", index=False)
print("Saved: submission.csv")

結果摘要

Validation（mAP@0.5 ≈ 0.696）
car 0.921, hov 0.850, motorcycle 0.628, person 0.387

主要觀察

car/hov 表現穩定；

person 為瓶頸（極小目標、遠景、召回不足）；

背景偶有 假陽性車框、密集區 多框殘留。

Kaggle Leaderboard

Private：0.28240

Public：0.25088

僅調「推論」時，建議使用分類別門檻：car 0.40 / hov 0.35 / motorcycle 0.33 / person 0.22，並將 NMS IoU ≈ 0.6，可在不重訓前提下稍微改善平衡。

可重現性
匯出完整環境（最穩）
pip freeze --all > requirements.txt

或在 Notebook 最後一格輸出「精簡需求」
import sys, importlib.metadata as md, pathlib
stdlib = set(getattr(sys, "stdlib_module_names", ()))
mods = {m.split(".",1)[0] for m in sys.modules if m and not m.startswith("_")}
mods = {m for m in mods if m not in stdlib}
alias = {"cv2":"opencv-python","yaml":"PyYAML","sklearn":"scikit-learn","skimage":"scikit-image","PIL":"Pillow","IPython":"ipython"}
pkg_map = md.packages_distributions()
pkgs, seen = [], set()
def add(d):
    if d and d not in seen:
        try: pkgs.append(f"{d}=={md.version(d)}"); seen.add(d)
        except md.PackageNotFoundError: pass
for m in sorted(mods):
    dists = pkg_map.get(m, [])
    if dists:
        for d in dists: add(d)
    else:
        add(alias.get(m, m))
path = pathlib.Path("requirements.txt"); path.write_text("\n".join(sorted(pkgs))+"\n", encoding="utf-8")
print("requirements.txt written to", path.resolve())

注意事項

imgsz=1280 有助於小物體，但顯存需求較高；顯存有限可改 1024 搭配較大 batch。

project / name 會影響權重與輸出路徑，請與 README 命令保持一致。

若你的比賽需要 像素座標，請在產生 submission 前先把相對座標換算為像素 xyxy。

授權

本倉庫僅供課程 / 研究使用。
資料集依競賽規範使用；模型與訓練程式基於 Ultralytics
 授權條款。
