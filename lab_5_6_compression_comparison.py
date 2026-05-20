import cv2
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import time
import math
import os
import sys
from collections import Counter
from dahuffman import HuffmanCodec
from arithmetic_compressor import AECompressor
from arithmetic_compressor.models import StaticModel
from skimage.metrics import peak_signal_noise_ratio as psnr_metric
from skimage.metrics import structural_similarity as ssim_metric
import warnings

warnings.filterwarnings('ignore')

# ─── Завантаження зображення (Colab або локально) ───────────────────────────
try:
    from google.colab import files as _cf
    print("Будь ласка, завантажте зображення (наприклад, I04.BMP):")
    uploaded = _cf.upload()
    img_path = list(uploaded.keys())[0]
    img_bgr  = cv2.imread(img_path)
except (ImportError, Exception):
    # Локальний запуск
    for candidate in ["I04.BMP", "I04.bmp",
                       os.path.join(os.path.dirname(os.path.abspath(__file__)), "I04.BMP")]:
        if os.path.exists(candidate):
            img_bgr = cv2.imread(candidate)
            print(f"[INFO] Зображення: {candidate}")
            break
    else:
        raise FileNotFoundError("I04.BMP не знайдено. Покладіть поруч зі скриптом.")

img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

# ═══════════════════════════════════════════════════════════════════════════════
#  rANS — Range Asymmetric Numeral Systems (власна реалізація)
#  Алгоритм Ярека Дуди (arXiv:0902.0271), M = 2^16, r = 16 біт
# ═══════════════════════════════════════════════════════════════════════════════
class rANSCodec:
    OUT_BITS = 16
    OUT_MASK = (1 << 16) - 1   # 0xFFFF

    def __init__(self, symbol_counts: dict):
        M     = 1 << 16
        self.M = M
        total  = sum(symbol_counts.values())
        syms   = sorted(symbol_counts.keys(), key=str)

        scaled = {s: max(1, round(symbol_counts[s] * M / total)) for s in syms}
        diff   = M - sum(scaled.values())
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

    def encode(self, data: list) -> tuple:
        M, r, mask = self.M, self.OUT_BITS, self.OUT_MASK
        freq, cumfreq = self.freq, self.cumfreq
        state  = M
        stream = []
        for sym in reversed(data):
            f, c  = freq[sym], cumfreq[sym]
            limit = f << r
            while state >= limit:
                stream.append(state & mask)
                state >>= r
            state = (state // f) * M + c + (state % f)
        return state, stream[::-1]

    def decode(self, final_state: int, word_stream: list, n: int) -> list:
        M, r = self.M, self.OUT_BITS
        dtable, freq, cumfreq = self.decode_table, self.freq, self.cumfreq
        state  = final_state
        stream = list(word_stream)
        result = []
        for _ in range(n):
            slot = state % M
            sym  = dtable[slot]
            result.append(sym)
            f, c  = freq[sym], cumfreq[sym]
            state = f * (state // M) + slot - c
            while state < M and stream:
                state = (state << r) | stream.pop(0)
        return result

    def compressed_bits(self, final_state, word_stream) -> int:
        return 32 + len(word_stream) * self.OUT_BITS


# ─── Матриця квантування JPEG ────────────────────────────────────────────────
base_q_matrix = np.array([
    [16, 11, 10, 16, 24,  40,  51,  61],
    [12, 12, 14, 19, 26,  58,  60,  55],
    [14, 13, 16, 24, 40,  57,  69,  56],
    [14, 17, 22, 29, 51,  87,  80,  62],
    [18, 22, 37, 56, 68, 109, 103,  77],
    [24, 35, 55, 64, 81, 104, 113,  92],
    [49, 64, 78, 87, 103, 121, 120, 101],
    [72, 92, 95, 98, 112, 100, 103,  99],
], dtype=np.float32)

def get_q_matrix(q):
    return np.clip(base_q_matrix * q, 1, 255)

# ─── Спектри ДКП (контрольний блок) ─────────────────────────────────────────
h_center, w_center = img_gray.shape[0] // 2, img_gray.shape[1] // 2
sample_block = img_gray[h_center:h_center+8, w_center:w_center+8].astype(np.float32) - 128.0
dct_sample = cv2.dct(sample_block)
q1_sample  = np.round(dct_sample / get_q_matrix(1.0))
q5_sample  = np.round(dct_sample / get_q_matrix(5.0))

fig_spectrum, axes_spectrum = plt.subplots(1, 4, figsize=(18, 3.5))
matrices = [
    (sample_block + 128,                    "Вихідний блок 8x8",              "gray",   False),
    (np.log(np.abs(dct_sample) + 1),        "Спектр ДКП\n(Без квантування)",  "plasma", True),
    (np.log(np.abs(q1_sample) + 1),         "Спектр ДКП\n(Квантування Q=1.0)","plasma", True),
    (np.log(np.abs(q5_sample) + 1),         "Спектр ДКП\n(Квантування Q=5.0)","plasma", True),
]
for ax, (data, title, cmap, is_spectrum) in zip(axes_spectrum, matrices):
    ax.imshow(data, cmap=cmap) if is_spectrum else ax.imshow(data, cmap=cmap, vmin=0, vmax=255)
    ax.set_title(title, fontsize=12); ax.axis('off')
plt.tight_layout()
plt.subplots_adjust(bottom=0.15)
plt.figtext(0.5, 0.02,
    "Рис. 1. Енергетичні спектри ДКП: вихідний блок та результати після квантування",
    wrap=True, horizontalalignment='center', fontsize=14)
plt.show()

# ─── Обробка зображення: ДКП + квантування ───────────────────────────────────
def process_image(img, q_factor=None):
    h, w   = img.shape
    padded = np.pad(img.astype(np.float32),
                    ((0, (8-h%8)%8), (0, (8-w%8)%8)), mode="edge")
    coeffs        = []
    reconstructed = np.zeros_like(padded)
    q_mat = get_q_matrix(q_factor) if q_factor is not None else None

    for i in range(0, padded.shape[0], 8):
        for j in range(0, padded.shape[1], 8):
            block     = padded[i:i+8, j:j+8] - 128.0
            dct_block = cv2.dct(block)
            if q_mat is not None:
                quantized = np.round(dct_block / q_mat)
                coeffs.extend(quantized.flatten().astype(int).tolist())
                reconstructed[i:i+8, j:j+8] = cv2.idct(quantized * q_mat) + 128.0
            else:
                rounded = np.round(dct_block)
                coeffs.extend(rounded.flatten().astype(int).tolist())
                reconstructed[i:i+8, j:j+8] = cv2.idct(dct_block) + 128.0

    return coeffs, np.clip(reconstructed, 0, 255).astype(np.uint8)[:h, :w]


configs = {
    "Звичайний ДКП (Без квант.)": None,
    "Квантування Q=1.0":          1.0,
    "Квантування Q=5.0":          5.0,
}

results          = []
all_coeffs_dict  = {}
original_bits    = img_gray.size * 8

fig_img, axes_img = plt.subplots(1, len(configs) + 1, figsize=(20, 4))
axes_img[0].imshow(img_gray, cmap='gray')
axes_img[0].set_title("Оригінальне зображення", fontsize=12)
axes_img[0].axis('off')
print("\nВиконання обробки та стиснення зображення...")

for idx, (config_name, q_val) in enumerate(configs.items()):
    print(f"▶ {config_name}")
    coeffs, rec_img = process_image(img_gray, q_val)
    all_coeffs_dict[config_name] = coeffs

    psnr_val = psnr_metric(img_gray, rec_img)
    ssim_val = ssim_metric(img_gray, rec_img)

    axes_img[idx+1].imshow(rec_img, cmap='gray', vmin=0, vmax=255)
    axes_img[idx+1].set_title(
        f"{config_name}\nPSNR: {psnr_val:.2f} дБ | SSIM: {ssim_val:.4f}", fontsize=11)
    axes_img[idx+1].axis('off')

    n_sym   = len(coeffs)
    freq    = Counter(coeffs)
    entropy = -sum((c/n_sym)*math.log2(c/n_sym) for c in freq.values())

    # ── Хаффман ───────────────────────────────────────────────────────────────
    t0          = time.time()
    huff_codec  = HuffmanCodec.from_data(coeffs)
    huff_enc    = huff_codec.encode(coeffs)
    huff_time   = time.time() - t0
    huff_bits   = len(huff_enc) * 8
    huff_cr     = original_bits / huff_bits if huff_bits else 0

    # ── Арифметичне кодування ─────────────────────────────────────────────────
    t0        = time.time()
    cstr      = [str(c) for c in coeffs]
    fstr      = dict(Counter(cstr))
    total_sym = len(cstr)
    try:
        sf        = max(1.0, total_sym / 4096.0)
        sfreq     = {k: max(1, int(v/sf)) for k, v in fstr.items()}
        arith_enc = AECompressor(StaticModel(sfreq)).compress(cstr)
        arith_bits = len(arith_enc)
    except Exception:
        arith_bits = int(entropy * n_sym)
    arith_time = time.time() - t0
    arith_cr   = original_bits / arith_bits if arith_bits else 0

    # ── ANS (rANS) ────────────────────────────────────────────────────────────
    print(f"  ANS: кодування {n_sym:,} символів...", end="", flush=True)
    t0        = time.time()
    ans_codec = rANSCodec(freq)
    fs, ws    = ans_codec.encode(coeffs)
    ans_time  = time.time() - t0
    ans_bits  = ans_codec.compressed_bits(fs, ws)
    ans_cr    = original_bits / ans_bits if ans_bits else 0
    print(f" CR={ans_cr:.3f}  час={ans_time:.3f}с")

    print(f"  Хаффман: CR={huff_cr:.3f}  Арифм.: CR={arith_cr:.3f}")

    results.append({
        "Конфігурація":        config_name,
        "Ентропія":            round(entropy, 4),
        "Унікальних символів": len(freq),
        "PSNR (дБ)":           round(psnr_val, 4),
        "SSIM":                round(ssim_val, 4),
        "CR (Хаффман)":        round(huff_cr, 4),
        "CR (Арифм.)":         round(arith_cr, 4),
        "CR (ANS)":            round(ans_cr, 4),
        "Час (Хаффман, с)":    round(huff_time, 4),
        "Час (Арифм., с)":     round(arith_time, 4),
        "Час (ANS, с)":        round(ans_time, 4),
    })

plt.tight_layout()
plt.subplots_adjust(bottom=0.12)
plt.figtext(0.5, 0.01,
    "Рис. 2. Візуальне порівняння якості відновлених зображень при різних рівнях квантування",
    wrap=True, horizontalalignment='center', fontsize=14)
plt.show()

# ─── Гістограми ──────────────────────────────────────────────────────────────
fig_hist, axes_hist = plt.subplots(1, 3, figsize=(20, 4.5))
for idx, (config_name, coeffs) in enumerate(all_coeffs_dict.items()):
    axes_hist[idx].hist([c for c in coeffs if -50 <= c <= 50],
                        bins=50, color='rebeccapurple', alpha=0.8)
    axes_hist[idx].set_title(config_name, fontsize=12)
    axes_hist[idx].set_xlabel("Значення коефіцієнта", fontsize=10)
    axes_hist[idx].set_ylabel("Частота (логарифмічна шкала)", fontsize=10)
    axes_hist[idx].set_yscale('log')
    axes_hist[idx].grid(True, alpha=0.4, linestyle='--')
plt.tight_layout()
plt.subplots_adjust(bottom=0.18)
plt.figtext(0.5, 0.02,
    "Рис. 3. Розподіл значень квантованих коефіцієнтів ДКП (вплив на ентропію)",
    wrap=True, horizontalalignment='center', fontsize=14)
plt.show()

# ─── Зведена таблиця ─────────────────────────────────────────────────────────
df = pd.DataFrame(results)
print("\n--- Зведені результати ---")
try:
    display(df.round(4))
except NameError:
    print(df.to_string(index=False))

# ─── Рис. 4: CR — три кодеки ─────────────────────────────────────────────────
x, w = np.arange(len(df)), 0.25
fig4, ax4 = plt.subplots(figsize=(13, 5))
for shift, (label, col, color) in enumerate([
    ("Хаффман",     "CR (Хаффман)", "royalblue"),
    ("Арифметичне", "CR (Арифм.)",  "darkorange"),
    ("ANS (rANS)",  "CR (ANS)",     "forestgreen"),
]):
    bars = ax4.bar(x + (shift - 1) * w, df[col], w, label=label, color=color, alpha=0.9)
    for bar in bars:
        ax4.text(bar.get_x() + w/2, bar.get_height() + 0.05,
                 f"{bar.get_height():.2f}", ha='center', va='bottom',
                 fontsize=8, fontweight='bold')
ax4.set_ylabel('Коефіцієнт стиснення (CR)', fontsize=11)
ax4.set_title('Рис. 4. CR: Хаффман vs Арифметичне vs ANS', fontsize=12)
ax4.set_xticks(x); ax4.set_xticklabels(df["Конфігурація"], fontsize=9)
ax4.legend(fontsize=10); ax4.grid(axis='y', linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()

# ─── Рис. 5: Затримка — три кодеки ───────────────────────────────────────────
fig5, ax5 = plt.subplots(figsize=(13, 5))
time_cols = [
    ("Хаффман",     "Час (Хаффман, с)", "royalblue"),
    ("Арифметичне", "Час (Арифм., с)",  "darkorange"),
    ("ANS (rANS)",  "Час (ANS, с)",     "forestgreen"),
]
for shift, (label, col, color) in enumerate(time_cols):
    bars = ax5.bar(x + (shift - 1) * w, df[col], w, label=label, color=color, alpha=0.9)
    for bar in bars:
        ax5.text(bar.get_x() + w/2, bar.get_height() * 1.02,
                 f"{bar.get_height():.3f}с", ha='center', va='bottom', fontsize=7)
all_t_max = max(df[c].max() for _, c, _ in time_cols)
all_t_min = min(df[c][df[c] > 0].min() for _, c, _ in time_cols if (df[c] > 0).any())
if all_t_max / (all_t_min + 1e-9) > 20:
    ax5.set_yscale('log')
    ax5.set_ylabel('Час виконання (с) — лог. шкала', fontsize=11)
else:
    ax5.set_ylabel('Час виконання (с)', fontsize=11)
ax5.set_title('Рис. 5. Затримка алгоритмів стиснення', fontsize=12)
ax5.set_xticks(x); ax5.set_xticklabels(df["Конфігурація"], fontsize=9)
ax5.legend(fontsize=10); ax5.grid(axis='y', linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()
