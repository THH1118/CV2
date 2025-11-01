# CVPDL HW2 – Aerial Object Detection (YOLOv9)

偵測 `car / hov / person / motorcycle`（空拍影像）。本 README 一次性說明：**環境安裝 → 資料整理 → 訓練 → 推論 / 產生 submission**。  
**最終成績**：Private **0.28240**、Public **0.25088**。

---

## 目錄
- [環境安裝](#環境安裝)
  - [選項 A. venv（建議）](#選項-a-venv建議)
  - [選項 B. Conda](#選項-b-conda)
- [資料準備](#資料準備)
  - [下載競賽資料（Kaggle）](#下載競賽資料kaggle)
  - [整理成 YOLO 結構（示意）](#整理成-yolo-結構示意)
  - [建立 `data.yaml`](#建立-datayaml)
- [專案結構](#專案結構)
- [訓練](#訓練)
- [推論與 Submission](#推論與-submission)
  - [1) 推論（Predict）](#1-推論predict)
  - [2) 產生 Submission](#2-產生-submission)
- [結果摘要](#結果摘要)
  - [Validation 成績](#validation-成績)
  - [Kaggle Leaderboard](#kaggle-leaderboard)
  - [僅調「推論」時的建議](#僅調推論時的建議)
- [可重現性](#可重現性)
- [注意事項](#注意事項)
- [授權](#授權)

---

## 環境安裝

> 若需 GPU，請依你的 **CUDA 版本**安裝相容的 `torch/torchvision`。

### 選項 A. venv（建議）
```bash
python -m venv .venv
# Windows
.\.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate

python -m pip install --upgrade pip wheel setuptools
pip install ultralytics opencv-python torch torchvision numpy pandas matplotlib tqdm PyYAML
選項 B. Conda
bash
複製程式碼
conda create -n cvpd_hw2 python=3.10 -y
conda activate cvpd_hw2
pip install ultralytics opencv-python torch torchvision numpy pandas matplotlib tqdm PyYAML
資料準備
下載競賽資料（Kaggle）
bash
複製程式碼
# 設定好 kaggle.json 後：
export KAGGLE_CONFIG_DIR=/path/to/your/kaggle   # Windows 可用：set KAGGLE_CONFIG_DIR=...
kaggle competitions download -c taica-cvpdl-2025-hw-2 -p ./data
unzip -q ./data/taica-cvpdl-2025-hw-2.zip -d ./data
整理成 YOLO 結構（示意）
bash
複製程式碼
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
yaml
複製程式碼
# hw2_data.yaml
path: ./data/CVPDL_hw2
train: images/train1280
val:   images/val1280
nc: 4
names: [car, hov, person, motorcycle]
專案結構
bash
複製程式碼
.
├─ hw2_data.yaml
├─ train.py                 # (可選) 你包好的訓練腳本；或直接用 yolo CLI
├─ infer_to_txt.py          # (可選) 推論並輸出 YOLO txt（含 conf）
├─ make_submission.py       # (可選) 將 txt 合併為比賽需要的 CSV
└─ data/
   └─ CVPDL_hw2/...
訓練
本作業採 Ultralytics YOLOv9，影像大小 imgsz=1280。

bash
複製程式碼
yolo detect train \
  model=yolov9m.yaml \
  data=hw2_data.yaml \
  epochs=300 imgsz=1280 batch=2 device=0 \
  optimizer=AdamW lr0=0.00075 lrf=0.01 cos_lr=True warmup_epochs=5 \
  mosaic=1.0 mixup=0.15 copy_paste=0.25 erasing=0.4 \
  degrees=10 scale=0.5 translate=0.1 close_mosaic=10 \
  pretrained=False \
  project=runs_hw2 name=yolov9_scratch
輸出權重位置：

bash
複製程式碼
runs_hw2/detect/yolov9_scratch/weights/best.pt
推論與 Submission
1) 推論（Predict）
bash
複製程式碼
yolo detect predict \
  model=runs_hw2/detect/yolov9_scratch/weights/best.pt \
  source=./data/CVPDL_hw2/test \
  imgsz=1280 device=0 \
  save_txt=True save_conf=True
YOLO 會在 runs/detect/predict*/labels/ 產生每張圖的 *.txt（class xc yc w h conf）。

2) 產生 Submission
若比賽需要像素座標 xyxy，請先依影像大小把相對座標還原為像素座標（此處範例僅示範讀檔與合併，請依賽制定義欄位）。

python
複製程式碼
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
Validation 成績
mAP@0.5 ≈ 0.696

類別表現（mAP@0.5）：

car 0.921

hov 0.850

motorcycle 0.628

person 0.387

主要觀察

car / hov 表現穩定。

person 為瓶頸（極小目標、遠景、召回不足）。

背景偶有假陽性車框、密集區多框殘留。

Kaggle Leaderboard
Private：0.28240

Public：0.25088

僅調「推論」時的建議
分類別門檻（score threshold）：

car 0.40 / hov 0.35 / motorcycle 0.33 / person 0.22

NMS IoU：≈ 0.6

不重訓前提下可稍微改善 Precision/Recall 平衡（依實測再微調）。

可重現性
匯出完整環境（最穩）
bash
複製程式碼
pip freeze --all > requirements.txt
在 Notebook 最後一格輸出「精簡需求」
python
複製程式碼
import sys, importlib.metadata as md, pathlib
stdlib = set(getattr(sys, "stdlib_module_names", ()))
mods = {m.split(".",1)[0] for m in sys.modules if m and not m.startswith("_")}
mods = {m for m in mods if m not in stdlib}
alias = {"cv2":"opencv-python","yaml":"PyYAML","sklearn":"scikit-learn","skimage":"scikit-image","PIL":"Pillow","IPython":"ipython"}
pkg_map = md.packages_distributions()
pkgs, seen = [], set()
def add(d):
    if d and d not in seen:
        try:
            pkgs.append(f"{d}=={md.version(d)}"); seen.add(d)
        except md.PackageNotFoundError:
            pass
for m in sorted(mods):
    dists = pkg_map.get(m, [])
    if dists:
        for d in dists: add(d)
    else:
        add(alias.get(m, m))
path = pathlib.Path("requirements.txt")
path.write_text("\n".join(sorted(pkgs))+"\n", encoding="utf-8")
print("requirements.txt written to", path.resolve())
注意事項
imgsz=1280 有助於小物體，但顯存需求較高；顯存有限可改 1024 搭配較大 batch。

project / name 會影響權重與輸出路徑，請與 README 命令保持一致。

若比賽需像素座標，請在產生 submission 前先把相對座標換算為像素 xyxy。

授權
本倉庫僅供課程 / 研究使用。
資料集依競賽規範使用；模型與訓練程式基於 Ultralytics。
請遵循相關授權條款。
