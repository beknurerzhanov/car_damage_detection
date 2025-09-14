import os
import io
import torch
import numpy as np
import streamlit as st
from PIL import Image
from pathlib import Path
from collections import Counter
from ultralytics import YOLO

st.set_page_config(page_title="Car Damage", layout="wide")

# Жёстко используем модель рядом с файлом
MODEL_PATH = str(Path(__file__).with_name("best_saved.pt"))
if not os.path.exists(MODEL_PATH):
    st.error(f"Файл модели не найден: {MODEL_PATH}")
    st.stop()

# Параметры
CONF = st.sidebar.slider("Confidence", 0.05, 0.9, 0.25, 0.01)
IOU  = st.sidebar.slider("IoU (NMS)", 0.2, 0.9, 0.45, 0.01)
IMGSZ = st.sidebar.slider("imgsz", 320, 1280, 640, 32)
DEVICE = 0 if torch.cuda.is_available() else "cpu"
st.sidebar.caption(f"Device: **{DEVICE}**")

@st.cache_resource(show_spinner=True)
def load_model():
    return YOLO(MODEL_PATH)

model = load_model()

@st.cache_data(show_spinner=False)
def get_class_names():
    try:
        names = model.model.names if hasattr(model, "model") else model.names
        if isinstance(names, dict):
            return [names[i] for i in sorted(names.keys())]
        if isinstance(names, list):
            return names
    except Exception:
        pass
    return None

NAMES = get_class_names()

st.title("🚗🔧 YOLOv8 детекция повреждений вмятины")

def run_on_image(pil_img: Image.Image):
    img_np = np.array(pil_img.convert("RGB"))
    results = model.predict(
        source=img_np, imgsz=IMGSZ, conf=CONF, iou=IOU, device=DEVICE, verbose=False
    )
    r = results[0]
    # r.plot() -> BGR ndarray; конвертируем в RGB без OpenCV
    annotated_bgr = r.plot()
    annotated_rgb = annotated_bgr[:, :, ::-1]  # BGR -> RGB
    cls_ids = r.boxes.cls.cpu().numpy().astype(int) if r.boxes is not None else []
    return annotated_rgb, Counter(cls_ids)

def counts_md(counter: Counter):
    if not counter:
        return "Ничего не найдено."
    lines = []
    for cid, cnt in sorted(counter.items()):
        label = NAMES[cid] if NAMES and cid < len(NAMES) else str(cid)
        lines.append(f"- {label}: **{cnt}**")
    return "\n".join(lines)

uploads = st.file_uploader(
    "Загрузите изображение(я)", type=["jpg","jpeg","png","bmp","webp"], accept_multiple_files=True
)

if uploads:
    for up in uploads:
        st.divider()
        col1, col2 = st.columns([2, 1], vertical_alignment="top")
        try:
            img = Image.open(up)
        except Exception as e:
            st.error(f"Не удалось открыть файл {up.name}: {e}")
            continue

        with st.spinner(f"Инференс: {up.name}"):
            try:
                annotated_rgb, counter = run_on_image(img)
            except Exception as e:
                st.error(f"Ошибка инференса на {up.name}: {e}")
                continue

        with col1:
            st.image(annotated_rgb, caption=f"Аннотированное — {up.name}", use_container_width=True)
            buf = io.BytesIO()
            Image.fromarray(annotated_rgb).save(buf, format="JPEG", quality=90)
            buf.seek(0)
            st.download_button("⬇️ Скачать", data=buf,
                               file_name=f"annotated_{up.name}.jpg", mime="image/jpeg")
        with col2:
            st.subheader("Статистика по классам")
            st.markdown(counts_md(counter))

    st.caption(f"Параметры: imgsz={IMGSZ}, conf={CONF}, iou={IOU}")

