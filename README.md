# CV2
**CVPDL HW2 – Aerial Object Detection (YOLOv9)**

偵測 **car / hov / person / motorcycle**（空拍影像）。本 README 一次性說明：**環境安裝 → 資料整理 → 訓練 → 推論 / 產生 submission**。  
最終成績：**Private 0.28240**、**Public 0.25088**。

---

## 目錄
- [環境安裝](#環境安裝)
- [資料準備](#資料準備)
- [專案結構](#專案結構)
- [訓練](#訓練)
- [推論與 Submission](#推論與-submission)
- [結果摘要](#結果摘要)
- [可重現性](#可重現性)
- [注意事項](#注意事項)
- [授權](#授權)

---

## 環境安裝

### ✅ 選項 A. venv（建議）
```bash
python -m venv .venv
# Windows
.\.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate

python -m pip install --upgrade pip wheel setuptools
pip install ultralytics opencv-python torch torchvision numpy pandas matplotlib tqdm PyYAML
