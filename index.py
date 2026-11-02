"""
Prediksi CNN berdasarkan foto daun selada (versi Python dari halaman CNN di index.html)

Cara pakai:
    pip install streamlit pillow numpy matplotlib requests
    streamlit run cnn_selada.py
"""
from __future__ import annotations  # agar anotasi list[float] | None aman di Python 3.8/3.9

import io
import math

import matplotlib

matplotlib.use("Agg")  # backend tanpa jendela, aman untuk Streamlit
import matplotlib.pyplot as plt
import numpy as np
import requests
import streamlit as st
from PIL import Image, ImageDraw, ImageOps

# ---------------------------------------------------------------- KONFIGURASI
CNN_URL = ""        # URL model CNN, kosongkan jika belum ada
FOV_CM = 20.0       # lebar bidang foto dalam cm
LEAF_MIN = 0.08     # minimal rasio area hijau agar dianggap selada
MAX_W = 420         # lebar maksimum foto saat dianalisis

CLS = [
    {"n": "Bibit (<3 cm)",            "c": 2.0,  "col": "#7c5cd6"},
    {"n": "Vegetatif awal (3-6 cm)",  "c": 4.5,  "col": "#2f7fd1"},
    {"n": "Vegetatif (6-10 cm)",      "c": 8.0,  "col": "#2f9e5b"},
    {"n": "Siap panen (>10 cm)",      "c": 12.0, "col": "#e8890c"},
]


# ---------------------------------------------------------------- LOGIKA
def cnn_score(w_cm: float) -> list[float]:
    """Peluang (%) tiap fase berdasarkan lebar daun."""
    e = [math.exp(-((w_cm - k["c"]) ** 2) / (2 * 1.8 * 1.8)) for k in CLS]
    t = sum(e) or 1.0
    return [v / t * 100 for v in e]


def analyze_image(img: Image.Image, fov_cm: float):
    """Segmentasi area hijau & ukur lebar daun.

    Return: (hasil_dict, gambar_pratinjau)
    hasil_dict: {"ok": bool, "ratio": float, "cm": float (jika ok)}
    """
    img = ImageOps.exif_transpose(img).convert("RGB")
    sc = min(1.0, MAX_W / img.width)
    w, h = max(1, round(img.width * sc)), max(1, round(img.height * sc))
    img = img.resize((w, h), Image.LANCZOS)

    a = np.asarray(img).astype(np.int32)
    r, g, b = a[..., 0], a[..., 1], a[..., 2]
    mask = (g > 50) & ((2 * g - r - b) > 30)

    n = int(mask.sum())
    ratio = n / (w * h)
    if n == 0 or ratio < LEAF_MIN:
        return {"ok": False, "ratio": ratio}, img

    hist = mask.sum(axis=0)            # jumlah piksel hijau per kolom
    rows = np.where(mask.any(axis=1))[0]
    y1, y2 = int(rows.min()), int(rows.max())

    cum = np.cumsum(hist)
    x1 = int(np.argmax(cum >= n * 0.01))
    cum_r = np.cumsum(hist[::-1])
    x2 = w - 1 - int(np.argmax(cum_r >= n * 0.01))

    # tandai area daun dengan warna merah muda
    out = a.astype(np.float64)
    pink = np.array([255, 61, 154], dtype=np.float64)
    out[mask] = out[mask] * 0.55 + pink * 0.45
    pre = Image.fromarray(np.clip(out, 0, 255).astype(np.uint8))
    ImageDraw.Draw(pre).rectangle([x1, y1, x2, y2], outline="#8bd450", width=2)

    cm = (x2 - x1) / w * fov_cm
    return {"ok": True, "ratio": ratio, "cm": cm}, pre


def send_to_model(url: str, file_bytes: bytes, filename: str):
    """Kirim foto ke model CNN. Respons JSON yang diharapkan:
    {"selada": true/false, "lebar_daun_cm": 7.5, "probabilitas": [p1,p2,p3,p4]}
    probabilitas bernilai 0..1.
    """
    resp = requests.post(url, files={"file": (filename, file_bytes)}, timeout=30)
    resp.raise_for_status()
    j = resp.json()
    if not isinstance(j, dict):
        raise ValueError("respons model harus berupa objek JSON")

    if j.get("selada") is False:
        return {"selada": False}

    w = j.get("lebar_daun_cm")
    w = float(w) if w is not None else None
    pr = j.get("probabilitas")
    pr = [float(x) * 100 for x in pr] if isinstance(pr, list) and len(pr) == 4 else None
    if pr is None and w is None:
        raise ValueError("respons tidak berisi lebar_daun_cm atau probabilitas")
    if pr is None:
        pr = cnn_score(w)
    return {"selada": j.get("selada"), "w": w, "pr": pr}


def plot_probs(pr: list[float] | None):
    fig, ax = plt.subplots(figsize=(7, 3.6))
    vals = pr if pr else [0, 0, 0, 0]
    m = vals.index(max(vals)) if pr else -1
    colors = [k["col"] if (i == m or not pr) else k["col"] + "55" for i, k in enumerate(CLS)]
    bars = ax.bar([k["n"] for k in CLS], vals, color=colors)
    ax.set_ylim(0, 105)
    ax.set_yticks(range(0, 101, 10))
    ax.set_yticklabels([f"{t}%" for t in range(0, 101, 10)])
    ax.set_ylabel("Peluang (%)")
    ax.grid(axis="y", alpha=0.3)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    if pr:
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, v + 1.5, f"{v:.1f}%",
                    ha="center", fontsize=9)
    plt.setp(ax.get_xticklabels(), fontsize=8)
    fig.tight_layout()
    return fig


def show_image(img: Image.Image, caption: str):
    """Kompatibel dengan berbagai versi Streamlit (parameter lebar berubah-ubah)."""
    for kw in ({"width": "stretch"}, {"use_container_width": True}, {"use_column_width": True}, {}):
        try:
            st.image(img, caption=caption, **kw)
            return
        except TypeError:
            continue


# ---------------------------------------------------------------- TAMPILAN
st.set_page_config(page_title="CNN Selada - Kelompok 7", page_icon="🥬", layout="wide")
st.title("🧠 Prediksi CNN berdasarkan foto daun")
st.caption("Unggah foto daun selada. Lebar daun diukur dari foto, lalu grafik batang "
           "menampilkan peluang tiap fase pertumbuhan.")

with st.sidebar:
    st.header("Konfigurasi")
    fov = st.number_input("Lebar bidang foto (cm)", 1.0, 200.0, FOV_CM, 0.5)
    url = st.text_input("URL model CNN (opsional)", CNN_URL).strip()

state = st.session_state
state.setdefault("model", None)  # hasil model CNN untuk foto tertentu: {"key", "result", "probs"}

left, right = st.columns([1, 2])
result = {"w": "-", "cls": "-", "conf": "-", "src": "-"}
probs = None

with left:
    up = st.file_uploader("Foto daun selada", type=["png", "jpg", "jpeg", "webp", "bmp"])
    send = st.button("Kirim foto ke model CNN")

    if up is None:
        state.model = None
        if send:
            st.warning("Pilih foto daun terlebih dahulu.")
    else:
        data = up.getvalue()
        file_key = (up.name, len(data))
        img = None
        try:
            img = Image.open(io.BytesIO(data))
            img.load()
        except Exception:  # noqa: BLE001
            st.error("File tidak bisa dibaca sebagai gambar.")

        if img is not None:
            info, preview = analyze_image(img, fov)
            show_image(preview, "Pratinjau deteksi daun")

            if not info["ok"]:
                st.error(f"⚠️ Foto bukan selada. Area daun hijau hanya "
                         f"{info['ratio']*100:.1f}% dari foto (minimal {LEAF_MIN*100:.0f}%). "
                         f"Unggah foto daun selada yang jelas.")
            else:
                w = round(info["cm"], 1)
                probs = cnn_score(w)
                m = probs.index(max(probs))
                result = {"w": f"{w} cm", "cls": CLS[m]["n"],
                          "conf": f"{probs[m]:.1f} %", "src": "Segmentasi foto"}
                st.success(f"✅ Selada terdeteksi. Area daun hijau mencakup "
                           f"{info['ratio']*100:.1f}% foto. Lebar daun sekitar {w} cm, "
                           f"fase {CLS[m]['n']}.")

            # kirim ke model hanya saat tombol ditekan; hasil disimpan agar tidak hilang saat rerun
            if send:
                if not url:
                    st.warning("URL model CNN belum diatur (isi di sidebar atau variabel CNN_URL).")
                elif not url.startswith(("http://", "https://")):
                    st.error("URL model harus diawali http:// atau https://")
                else:
                    try:
                        res = send_to_model(url, data, up.name)
                        if res["selada"] is False:
                            state.model = {"key": file_key, "notlettuce": True}
                        else:
                            pr = res["pr"]
                            mm = pr.index(max(pr))
                            state.model = {
                                "key": file_key, "probs": pr,
                                "result": {
                                    "w": f"{res['w']:.1f} cm" if res["w"] is not None else "-",
                                    "cls": CLS[mm]["n"], "conf": f"{pr[mm]:.1f} %",
                                    "src": "Model CNN"},
                            }
                    except Exception as e:  # noqa: BLE001
                        st.error(f"Gagal menghubungi model: {e}")

            mdl = state.model
            if mdl and mdl["key"] == file_key:
                if mdl.get("notlettuce"):
                    st.error("⚠️ Foto bukan selada menurut model CNN.")
                    result, probs = {"w": "-", "cls": "-", "conf": "-", "src": "-"}, None
                else:
                    st.success("Hasil diambil dari model CNN.")
                    result, probs = mdl["result"], mdl["probs"]

    st.markdown("---")
    for label, key in [("Lebar daun terdeteksi", "w"), ("Fase prediksi", "cls"),
                       ("Keyakinan", "conf"), ("Sumber", "src")]:
        c1, c2 = st.columns([3, 2])
        c1.write(label)
        c2.markdown(f"**{result[key]}**")

    st.caption("Tanpa model, lebar daun diperkirakan dari area hijau pada foto "
               "(bukan inferensi CNN sungguhan). Isi URL model CNN agar foto dikirim "
               "ke modelmu.")

with right:
    fig = plot_probs(probs)
    st.pyplot(fig)
    plt.close(fig)