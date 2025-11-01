# CVPDL HW2 – Aerial Object Detection (YOLOv9)

偵測 **car / hov / person / motorcycle**（空拍影像）。本 README 依序說明：**環境安裝 → 資料整理 → 訓練 → 推論 / 產生 submission**。  
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


## 環境安裝

> 若需 **GPU**，請依你的 **CUDA 版本**安裝相容的 `torch`/`torchvision`。可參考 PyTorch 官方安裝指引。

### 選項 A. venv（建議）
```bash
python -m venv .venv
# Windows
.\.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate

python -m pip install --upgrade pip wheel setuptools
pip install ultralytics opencv-python torch torchvision numpy pandas matplotlib tqdm PyYAML


## 資料準備

### 下載競賽資料（Kaggle）
```bash
# 先在本機放好 kaggle.json（API 金鑰），並設定環境變數：
export KAGGLE_CONFIG_DIR=/path/to/your/kaggle   # Windows 可用：set KAGGLE_CONFIG_DIR=...
kaggle competitions download -c taica-cvpdl-2025-hw-2 -p ./data
unzip -q ./data/taica-cvpdl-2025-hw-2.zip -d ./data


## 專案結構

```text
.
├─ hw2_data.yaml                # YOLO 訓練用資料設定
├─ train.py                     # (可選) 你包好的訓練腳本；或直接用 yolo CLI
├─ infer_to_txt.py              # (可選) 推論並輸出 YOLO txt（含 conf）
├─ make_submission.py           # (可選) 將 txt 合併為比賽需要的 CSV
└─ data/
   └─ CVPDL_hw2/
      ├─ images/
      │  ├─ train1280/
      │  ├─ val1280/
      │  └─ test/               # 如主辦方提供
      └─ labels/
         ├─ train1280/
         └─ val1280/


## 訓練

本作業採 **Ultralytics YOLOv9**，影像大小建議 `imgsz=1280`（小物體較友善，顯存較吃）。

```bash
yolo detect train \
  model=yolov9m.yaml \
  data=hw2_data.yaml \
  epochs=300 imgsz=1280 batch=2 device=0 \
  optimizer=AdamW lr0=0.00075 lrf=0.01 cos_lr=True warmup_epochs=5 \
  mosaic=1.0 mixup=0.15 copy_paste=0.25 erasing=0.4 \
  degrees=10 scale=0.5 translate=0.1 close_mosaic=10 \
  pretrained=False \
  project=runs_hw2 name=yolov9_scratch


## 推論與 Submission

### 1) 推論（Predict）
```bash
yolo detect predict \
  model=runs_hw2/detect/yolov9_scratch/weights/best.pt \
  source=./data/CVPDL_hw2/test \
  imgsz=1280 device=0 \
  save_txt=True save_conf=True


## 結果摘要

### Validation 成績
- **mAP@0.5 ≈ 0.696**
- 類別表現（mAP@0.5）：
  - `car` **0.921**
  - `hov` **0.850**
  - `motorcycle` **0.628**
  - `person` **0.387**

**主要觀察**
- `car / hov` 表現穩定。
- `person` 為瓶頸（極小目標、遠景、召回不足）。
- 背景偶有 **假陽性車框**、密集區 **多框殘留**。

### Kaggle Leaderboard
- **Private**：`0.28240`
- **Public**：`0.25088`

### 僅調「推論」時的建議（免重訓）
- **分類別門檻（score threshold）**：
  - `car 0.40` / `hov 0.35` / `motorcycle 0.33` / `person 0.22`
- **NMS IoU**：`≈ 0.6`  
> 有助於在不重訓的前提下，略為改善 Precision / Recall 的平衡（請依實測微調）。
