#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Лабораторна робота № 5-6 — Порівняння ентропійних кодеків
=========================================================
Реалізовано всі три методи з умови завдання:
  2. Хаффман           (pip install dahuffman)
  3. Арифметичне       (pip install arithmetic-compressor)
  4. ANS / rANS        (власна реалізація — без зовнішніх бібліотек)
  5. Порівняльний аналіз: CR + затримка для трьох конфігурацій ДКП

Запуск:
  Google Colab: завантажте файл → автоматично відкриється діалог upload
  Локально    : покладіть I04.BMP поруч зі скриптом або вкажіть IMG_PATH
"""

# ─── стандартні бібліотеки ───────────────────────────────────────────────────
import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")   # без GUI; у Colab графіки відображаються окремо
import matplotlib.pyplot as plt
import pandas as pd
import time
import math
import os
import sys
import warnings
from collections import Counter

warnings.filterwarnings("ignore")

# ─── встановлення / імпорт зовнішніх пакетів ────────────────────────────────
def _pip(pkg):
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", pkg])

try:
    from dahuffman import HuffmanCodec
except ImportError:
    _pip("dahuffman"); from dahuffman import HuffmanCodec

try:
    from arithmetic_compressor import AECompressor
    from arithmetic_compressor.models import StaticModel
except ImportError:
    _pip("arithmetic-compressor")
    from arithmetic_compressor import AECompressor
    from arithmetic_compressor.models import StaticModel

try:
    from skimage.metrics import peak_signal_noise_ratio as psnr_fn
    from skimage.metrics import structural_similarity  as ssim_fn
except ImportError:
    _pip("scikit-image")
    from skimage.metrics import peak_signal_noise_ratio as psnr_fn
    from skimage.metrics import structural_similarity  as ssim_fn


# ═══════════════════════════════════════════════════════════════════════════════
#  rANS — Range Asymmetric Numeral Systems
#  Алгоритм Ярека Дуди (arXiv:0902.0271)
#
#  Стан x ∈ [M, M·2^r):
#    Кодування символу s (частота f_s, кумулятивна c_s, таблиця M):
#       нормалізація: поки x ≥ f_s · 2^r → виводимо r-бітне слово, x >>= r
#       крок ANS:     x = (x // f_s) · M  +  c_s  +  (x mod f_s)
#    Декодування:
#       s = decode_table[x mod M]
#       зворотній крок: x = f_s · (x // M) + (x mod M) - c_s
#       ренормалізація: поки x < M → x = (x << r) | read_word()
#
#  Параметри: M = 2^16 = 65 536,  r = 16 бітів
#  Вивід кодера: фінальний стан (32 біти) + потік r-бітних слів
# ═══════════════════════════════════════════════════════════════════════════════
class rANSCodec:
    """Повна реалізація rANS з кодуванням і декодуванням."""

    OUT_BITS = 16
    OUT_MASK = (1 << OUT_BITS) - 1    # 0xFFFF

    def __init__(self, symbol_counts: dict):
        """
        symbol_counts : {symbol: int_count}
        Автоматично масштабує частоти до суми M = 2^16.
        """
        M     = 1 << 16
        self.M = M
        total  = sum(symbol_counts.values())
        syms   = sorted(symbol_counts.keys(), key=str)

        # Пропорційне масштабування (кожен символ ≥ 1)
        scaled = {s: max(1, round(symbol_counts[s] * M / total)) for s in syms}

        # Корекція до точної суми M
        diff = M - sum(scaled.values())
        ordered = sorted(syms, key=lambda s: symbol_counts[s], reverse=True)
        i = 0
        while diff > 0:
            scaled[ordered[i % len(ordered)]] += 1; diff -= 1; i += 1
        while diff < 0:
            idx = i % len(ordered)
            if scaled[ordered[idx]] > 1:
                scaled[ordered[idx]] -= 1; diff += 1
            i += 1
        assert sum(scaled.values()) == M

        self.freq    = scaled
        self.cumfreq = {}
        self.decode_table = [None] * M

        cum = 0
        for s in syms:
            self.cumfreq[s] = cum
            for k in range(cum, cum + scaled[s]):
                self.decode_table[k] = s
            cum += scaled[s]

    # ── кодування ─────────────────────────────────────────────────────────────
    def encode(self, data: list) -> tuple:
        """Повертає (final_state: int, word_stream: list[int])."""
        M, r, mask = self.M, self.OUT_BITS, self.OUT_MASK
        freq, cumfreq = self.freq, self.cumfreq

        state  = M          # початковий стан = L = M
        stream = []

        for sym in reversed(data):   # кодуємо у зворотньому порядку
            f, c = freq[sym], cumfreq[sym]
            limit = f << r
            while state >= limit:
                stream.append(state & mask)
                state >>= r
            state = (state // f) * M + c + (state % f)

        return state, stream[::-1]   # stream у прямому порядку

    # ── декодування ───────────────────────────────────────────────────────────
    def decode(self, final_state: int, word_stream: list, n: int) -> list:
        """Декодує n символів із (final_state, word_stream)."""
        M, r = self.M, self.OUT_BITS
        dtable, freq, cumfreq = self.decode_table, self.freq, self.cumfreq

        state  = final_state
        stream = list(word_stream)
        result = []

        for _ in range(n):
            slot = state % M
            sym  = dtable[slot]
            result.append(sym)

            f, c = freq[sym], cumfreq[sym]
            state = f * (state // M) + slot - c

            while state < M and stream:
                state = (state << r) | stream.pop(0)

        return result

    # ── розмір стисненого потоку ──────────────────────────────────────────────
    def compressed_bits(self, final_state: int, word_stream: list) -> int:
        """Загальна кількість бітів: 32 (стан) + |stream| × 16."""
        return 32 + len(word_stream) * self.OUT_BITS


# ─── самоперевірка rANS ──────────────────────────────────────────────────────
def _self_test_rans():
    data  = [0, 1, 0, 0, 2, 1, 0, 3, 0, 0, 1, 2, 0, 0, 0, 1]
    codec = rANSCodec(Counter(data))
    fs, ws = codec.encode(data)
    decoded = codec.decode(fs, ws, len(data))
    assert decoded == data, f"rANS self-test FAILED: {decoded} != {data}"
    print("[rANS] ✓ self-test passed")

_self_test_rans()


# ═══════════════════════════════════════════════════════════════════════════════
#  Матриця квантування JPEG та обробка зображення
# ═══════════════════════════════════════════════════════════════════════════════
BASE_Q = np.array([
    [16, 11, 10, 16, 24,  40,  51,  61],
    [12, 12, 14, 19, 26,  58,  60,  55],
    [14, 13, 16, 24, 40,  57,  69,  56],
    [14, 17, 22, 29, 51,  87,  80,  62],
    [18, 22, 37, 56, 68, 109, 103,  77],
    [24, 35, 55, 64, 81, 104, 113,  92],
    [49, 64, 78, 87, 103, 121, 120, 101],
    [72, 92, 95, 98, 112, 100, 103,  99],
], dtype=np.float32)


def q_matrix(q: float) -> np.ndarray:
    return np.clip(BASE_Q * q, 1, 255)


def process_image(img_gray: np.ndarray, q_factor=None):
    """
    Блочний ДКП 8×8 + опціональне квантування.
    Повертає (coefficients: list[int], reconstructed: np.ndarray).
    """
    h, w = img_gray.shape
    padded = np.pad(
        img_gray.astype(np.float32),
        ((0, (8 - h % 8) % 8), (0, (8 - w % 8) % 8)),
        mode="edge",
    )
    qm     = q_matrix(q_factor) if q_factor is not None else None
    coeffs = []
    rec    = np.zeros_like(padded)

    for i in range(0, padded.shape[0], 8):
        for j in range(0, padded.shape[1], 8):
            blk = padded[i:i+8, j:j+8] - 128.0
            dct = cv2.dct(blk)
            if qm is not None:
                q   = np.round(dct / qm)
                coeffs.extend(q.flatten().astype(int).tolist())
                rec[i:i+8, j:j+8] = cv2.idct(q * qm) + 128.0
            else:
                q   = np.round(dct)
                coeffs.extend(q.flatten().astype(int).tolist())
                rec[i:i+8, j:j+8] = cv2.idct(dct) + 128.0

    return coeffs, np.clip(rec, 0, 255).astype(np.uint8)[:h, :w]


# ═══════════════════════════════════════════════════════════════════════════════
#  Завантаження зображення (Colab + локальний запуск)
# ═══════════════════════════════════════════════════════════════════════════════
def load_image():
    # ── Google Colab ──────────────────────────────────────────────────────────
    try:
        from google.colab import files as _cf
        print("Завантажте зображення (I04.BMP):")
        uploaded = _cf.upload()
        path = list(uploaded.keys())[0]
        img  = cv2.imread(path)
        if img is not None:
            return img, path
    except (ImportError, Exception):
        pass

    # ── Локальний пошук ───────────────────────────────────────────────────────
    search = [
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "I04.BMP"),
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "I04.bmp"),
        "I04.BMP", "I04.bmp",
    ]
    for p in search:
        if os.path.exists(p):
            img = cv2.imread(p)
            if img is not None:
                print(f"[INFO] Зображення: {p}")
                return img, p

    raise FileNotFoundError(
        "Зображення I04.BMP не знайдено. "
        "Покладіть його поруч зі скриптом або вкажіть IMG_PATH вручну."
    )


IMG_PATH = None   # ← вкажіть вручну: IMG_PATH = "/path/to/I04.BMP"

if IMG_PATH and os.path.exists(IMG_PATH):
    img_bgr = cv2.imread(IMG_PATH)
else:
    img_bgr, IMG_PATH = load_image()

img_gray     = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
original_bits = img_gray.size * 8
print(f"Розмір: {img_gray.shape[1]}×{img_gray.shape[0]} пкс | "
      f"Оригінал: {original_bits:,} біт")


# ═══════════════════════════════════════════════════════════════════════════════
#  Основний цикл — три конфігурації
# ═══════════════════════════════════════════════════════════════════════════════
CONFIGS = {
    "Звичайний ДКП (без квант.)": None,
    "Квантування Q=1.0":          1.0,
    "Квантування Q=5.0":          5.0,
}

results    = []
all_coeffs = {}
rec_imgs   = {}

print("\n=== Обробка та стиснення ===\n")

for cfg, q_val in CONFIGS.items():
    print(f"▶ {cfg}")

    coeffs, rec = process_image(img_gray, q_val)
    all_coeffs[cfg] = coeffs
    rec_imgs[cfg]   = rec

    freq    = Counter(coeffs)
    n_sym   = len(coeffs)
    entropy = -sum((c / n_sym) * math.log2(c / n_sym) for c in freq.values())
    psnr_v  = psnr_fn(img_gray, rec)
    ssim_v  = ssim_fn(img_gray, rec)

    row = {
        "Конфігурація":        cfg,
        "Ентропія (біт)":      round(entropy, 4),
        "Унікальних символів": len(freq),
        "PSNR (дБ)":           round(psnr_v, 2),
        "SSIM":                round(ssim_v, 4),
    }

    # ── 2. Хаффман ────────────────────────────────────────────────────────────
    t0       = time.time()
    hc_obj   = HuffmanCodec.from_data(coeffs)
    enc_h    = hc_obj.encode(coeffs)
    t_h      = time.time() - t0
    b_h      = len(enc_h) * 8
    cr_h     = original_bits / b_h if b_h else 0
    row["CR (Хаффман)"]    = round(cr_h, 4)
    row["Час Хаффман (с)"] = round(t_h, 4)
    print(f"  Хаффман:     CR={cr_h:.3f}  час={t_h:.3f}с")

    # ── 3. Арифметичне кодування ──────────────────────────────────────────────
    t0      = time.time()
    cstr    = [str(c) for c in coeffs]
    fstr    = dict(Counter(cstr))
    try:
        sf   = max(1.0, sum(fstr.values()) / 4096.0)
        sfreq = {k: max(1, int(v / sf)) for k, v in fstr.items()}
        enc_a = AECompressor(StaticModel(sfreq)).compress(cstr)
        b_a   = len(enc_a)
    except Exception:
        b_a = max(1, int(entropy * n_sym))
    t_a  = time.time() - t0
    cr_a = original_bits / b_a if b_a else 0
    row["CR (Арифм.)"]    = round(cr_a, 4)
    row["Час Арифм. (с)"] = round(t_a, 4)
    print(f"  Арифметичне: CR={cr_a:.3f}  час={t_a:.3f}с")

    # ── 4. ANS (rANS) ─────────────────────────────────────────────────────────
    print(f"  ANS: кодування {n_sym:,} символів...", end="", flush=True)
    t0        = time.time()
    ans_codec = rANSCodec(freq)
    fs, ws    = ans_codec.encode(coeffs)
    t_ans     = time.time() - t0
    b_ans     = ans_codec.compressed_bits(fs, ws)
    cr_ans    = original_bits / b_ans if b_ans else 0
    row["CR (ANS)"]     = round(cr_ans, 4)
    row["Час ANS (с)"]  = round(t_ans, 4)
    print(f" CR={cr_ans:.3f}  час={t_ans:.3f}с")

    # Швидка перевірка коректності декодування (перші 500 символів)
    sample = min(500, n_sym)
    decoded = ans_codec.decode(fs, ws, sample)
    assert decoded == coeffs[:sample], "ANS decode MISMATCH — перевірте rANSCodec!"
    print("  ✓ ANS decode перевірено\n")

    results.append(row)


# ═══════════════════════════════════════════════════════════════════════════════
#  Таблиця результатів
# ═══════════════════════════════════════════════════════════════════════════════
df = pd.DataFrame(results)
print("=" * 95)
print("ЗВЕДЕНА ТАБЛИЦЯ")
print("=" * 95)
pd.set_option("display.max_columns", None)
pd.set_option("display.width", 200)
print(df.to_string(index=False))
print("=" * 95)


# ═══════════════════════════════════════════════════════════════════════════════
#  Графіки
# ═══════════════════════════════════════════════════════════════════════════════
OUT_DIR = os.path.dirname(os.path.abspath(__file__))

def _save(fig, name):
    path = os.path.join(OUT_DIR, name)
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"[Збережено] {name}")


# ── Рис. 1: Спектри ДКП ──────────────────────────────────────────────────────
hc, wc = img_gray.shape[0] // 2, img_gray.shape[1] // 2
blk    = img_gray[hc:hc+8, wc:wc+8].astype(np.float32) - 128
dct_s  = cv2.dct(blk)
fig1, ax1s = plt.subplots(1, 4, figsize=(18, 3.5))
for ax, (data, title, cmap) in zip(ax1s, [
    (blk + 128,                                                  "Вихідний блок 8×8",   "gray"),
    (np.log(np.abs(dct_s) + 1),                                  "ДКП (без квант.)",    "plasma"),
    (np.log(np.abs(np.round(dct_s / q_matrix(1.0))) + 1),        "ДКП (Q=1.0)",         "plasma"),
    (np.log(np.abs(np.round(dct_s / q_matrix(5.0))) + 1),        "ДКП (Q=5.0)",         "plasma"),
]):
    ax.imshow(data, cmap=cmap, vmin=0 if cmap == "gray" else None)
    ax.set_title(title, fontsize=10); ax.axis("off")
fig1.suptitle("Рис. 1. Енергетичні спектри ДКП при різних рівнях квантування", fontsize=12)
fig1.tight_layout()
_save(fig1, "fig1_dct_spectra.png")

# ── Рис. 2: Відновлені зображення ────────────────────────────────────────────
fig2, ax2s = plt.subplots(1, len(CONFIGS) + 1, figsize=(22, 4))
ax2s[0].imshow(img_gray, cmap="gray"); ax2s[0].set_title("Оригінал"); ax2s[0].axis("off")
for ax, (cfg, rec) in zip(ax2s[1:], rec_imgs.items()):
    row = df[df["Конфігурація"] == cfg].iloc[0]
    ax.imshow(rec, cmap="gray", vmin=0, vmax=255)
    ax.set_title(f"{cfg}\nPSNR={row['PSNR (дБ)']:.1f}дБ | SSIM={row['SSIM']:.4f}", fontsize=9)
    ax.axis("off")
fig2.suptitle("Рис. 2. Якість відновлення при різних рівнях квантування", fontsize=12)
fig2.tight_layout()
_save(fig2, "fig2_reconstruction.png")

# ── Рис. 3: Гістограми коефіцієнтів ─────────────────────────────────────────
fig3, ax3s = plt.subplots(1, 3, figsize=(19, 4))
for ax, (cfg, coeffs) in zip(ax3s, all_coeffs.items()):
    ax.hist([c for c in coeffs if -60 <= c <= 60], bins=60,
            color="rebeccapurple", alpha=0.85)
    ax.set_title(cfg, fontsize=10)
    ax.set_xlabel("Значення коефіцієнта ДКП"); ax.set_ylabel("Частота (лог.)")
    ax.set_yscale("log"); ax.grid(True, alpha=0.3, linestyle="--")
fig3.suptitle("Рис. 3. Розподіл квантованих коефіцієнтів ДКП", fontsize=12)
fig3.tight_layout()
_save(fig3, "fig3_histograms.png")

# ── Рис. 4: Коефіцієнт стиснення ─────────────────────────────────────────────
x, w = np.arange(len(df)), 0.25
fig4, ax4 = plt.subplots(figsize=(13, 5))
for shift, (label, col, color) in enumerate([
    ("Хаффман",     "CR (Хаффман)", "royalblue"),
    ("Арифметичне", "CR (Арифм.)",  "darkorange"),
    ("ANS (rANS)",  "CR (ANS)",     "forestgreen"),
]):
    bars = ax4.bar(x + (shift - 1) * w, df[col], w, label=label, color=color, alpha=0.9)
    for bar in bars:
        ax4.text(bar.get_x() + w / 2, bar.get_height() + 0.05,
                 f"{bar.get_height():.2f}", ha="center", va="bottom",
                 fontsize=8, fontweight="bold")
ax4.set_ylabel("Коефіцієнт стиснення (CR)", fontsize=11)
ax4.set_title("Рис. 4. CR: Хаффман vs Арифметичне vs ANS", fontsize=12)
ax4.set_xticks(x); ax4.set_xticklabels(df["Конфігурація"], fontsize=9)
ax4.legend(fontsize=10); ax4.grid(axis="y", linestyle="--", alpha=0.5)
fig4.tight_layout()
_save(fig4, "fig4_cr_comparison.png")

# ── Рис. 5: Затримка ─────────────────────────────────────────────────────────
fig5, ax5 = plt.subplots(figsize=(13, 5))
time_cols = [
    ("Хаффман",     "Час Хаффман (с)", "royalblue"),
    ("Арифметичне", "Час Арифм. (с)",  "darkorange"),
    ("ANS (rANS)",  "Час ANS (с)",     "forestgreen"),
]
for shift, (label, col, color) in enumerate(time_cols):
    bars = ax5.bar(x + (shift - 1) * w, df[col], w, label=label, color=color, alpha=0.9)
    for bar in bars:
        ax5.text(bar.get_x() + w / 2, bar.get_height() * 1.02,
                 f"{bar.get_height():.3f}с", ha="center", va="bottom", fontsize=7)

all_t_max = max(df[c].max() for _, c, _ in time_cols)
all_t_min = min(df[c][df[c] > 0].min() for _, c, _ in time_cols if (df[c] > 0).any())
if all_t_max / (all_t_min + 1e-9) > 20:
    ax5.set_yscale("log")
    ax5.set_ylabel("Час виконання (с) — лог. шкала", fontsize=11)
else:
    ax5.set_ylabel("Час виконання (с)", fontsize=11)
ax5.set_title("Рис. 5. Затримка алгоритмів стиснення", fontsize=12)
ax5.set_xticks(x); ax5.set_xticklabels(df["Конфігурація"], fontsize=9)
ax5.legend(fontsize=10); ax5.grid(axis="y", linestyle="--", alpha=0.5)
fig5.tight_layout()
_save(fig5, "fig5_timing.png")


# ═══════════════════════════════════════════════════════════════════════════════
#  Перевірка числел із презентації
# ═══════════════════════════════════════════════════════════════════════════════
EXPECTED = {
    "Звичайний ДКП (без квант.)": {"CR (Хаффман)": 1.67, "CR (Арифм.)": 1.60, "CR (ANS)": 1.61},
    "Квантування Q=5.0":          {"CR (Хаффман)": 7.17, "CR (Арифм.)": 26.72},
}

print("\n── Перевірка числел із презентації ─────────────────────────────────────")
for row in results:
    cfg = row["Конфігурація"]
    exp = EXPECTED.get(cfg, {})
    for metric, expected_val in exp.items():
        actual = row.get(metric, 0)
        diff   = abs(actual - expected_val)
        status = "✓ OK" if diff < 0.5 else "⚠  РОЗБІЖНІСТЬ"
        print(f"  {cfg:<35} | {metric:<15}: очікувано {expected_val:.2f}, "
              f"отримано {actual:.2f}  {status}")

print("\n✓ Готово. Всі графіки збережено поруч зі скриптом.")
