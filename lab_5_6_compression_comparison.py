import cv2
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import time
import math
from collections import Counter
from dahuffman import HuffmanCodec
from arithmetic_compressor import AECompressor
from arithmetic_compressor.models import StaticModel
from skimage.metrics import peak_signal_noise_ratio as psnr_metric
from skimage.metrics import structural_similarity as ssim_metric
from google.colab import files
import warnings

warnings.filterwarnings('ignore')

# Завантаження вхідного файлу
print("Будь ласка, завантажте зображення (наприклад, I04.BMP):")
uploaded = files.upload()
img_path = list(uploaded.keys())[0]
img_bgr = cv2.imread(img_path)
img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

# Стандартна матриця квантування JPEG
base_q_matrix = np.array([
    [16, 11, 10, 16, 24, 40, 51, 61],
    [12, 12, 14, 19, 26, 58, 60, 55],
    [14, 13, 16, 24, 40, 57, 69, 56],
    [14, 17, 22, 29, 51, 87, 80, 62],
    [18, 22, 37, 56, 68, 109, 103, 77],
    [24, 35, 55, 64, 81, 104, 113, 92],
    [49, 64, 78, 87, 103, 121, 120, 101],
    [72, 92, 95, 98, 112, 100, 103, 99]
], dtype=np.float32)

def get_q_matrix(q):
    return np.clip(base_q_matrix * q, 1, 255)

# Візуалізація спектрів для контрольного блоку
h_center, w_center = img_gray.shape[0] // 2, img_gray.shape[1] // 2
sample_block = img_gray[h_center:h_center+8, w_center:w_center+8].astype(np.float32) - 128.0

dct_sample = cv2.dct(sample_block)
q1_sample = np.round(dct_sample / get_q_matrix(1.0))
q5_sample = np.round(dct_sample / get_q_matrix(5.0))

fig_spectrum, axes_spectrum = plt.subplots(1, 4, figsize=(18, 3.5))

matrices = [
    (sample_block + 128, "Вихідний блок 8x8", "gray", False),
    (np.log(np.abs(dct_sample) + 1), "Спектр ДКП\n(Без квантування)", "plasma", True),
    (np.log(np.abs(q1_sample) + 1), "Спектр ДКП\n(Квантування Q=1.0)", "plasma", True),
    (np.log(np.abs(q5_sample) + 1), "Спектр ДКП\n(Квантування Q=5.0)", "plasma", True)
]

for ax, (data, title, cmap, is_spectrum) in zip(axes_spectrum, matrices):
    if is_spectrum:
        ax.imshow(data, cmap=cmap)
    else:
        ax.imshow(data, cmap=cmap, vmin=0, vmax=255)
    ax.set_title(title, fontsize=12)
    ax.axis('off')

plt.tight_layout()
plt.subplots_adjust(bottom=0.15)
plt.figtext(0.5, 0.02, "Рис. 1. Енергетичні спектри ДКП: вихідний блок та результати після квантування", wrap=True, horizontalalignment='center', fontsize=14)
plt.show()

# Обробка зображення: ДКП та квантування
def process_image(img, q_factor=None):
    h, w = img.shape
    pad_h = (8 - h % 8) % 8
    pad_w = (8 - w % 8) % 8
    padded = np.pad(img.astype(np.float32), ((0, pad_h), (0, pad_w)), mode="edge")
    
    coeffs = []
    reconstructed = np.zeros_like(padded)
    q_mat = get_q_matrix(q_factor) if q_factor is not None else None

    for i in range(0, padded.shape[0], 8):
        for j in range(0, padded.shape[1], 8):
            block = padded[i:i+8, j:j+8] - 128.0
            dct_block = cv2.dct(block)
            
            if q_mat is not None:
                quantized = np.round(dct_block / q_mat)
                coeffs.extend(quantized.flatten().astype(int).tolist())
                dequantized = quantized * q_mat
                rec_block = cv2.idct(dequantized)
            else:
                rounded = np.round(dct_block)
                coeffs.extend(rounded.flatten().astype(int).tolist())
                rec_block = cv2.idct(dct_block)
                
            reconstructed[i:i+8, j:j+8] = rec_block + 128.0
            
    return coeffs, np.clip(reconstructed, 0, 255).astype(np.uint8)[:h, :w]

# Налаштування параметрів тестування
configs = {
    "Звичайний ДКП (Без квант.)": None,
    "Квантування Q=1.0": 1.0,
    "Квантування Q=5.0": 5.0
}

results = []
all_coeffs_dict = {}
original_bits = img_gray.size * 8

# Основний цикл тестування та кодування
fig_img, axes_img = plt.subplots(1, len(configs) + 1, figsize=(20, 4))

axes_img[0].imshow(img_gray, cmap='gray')
axes_img[0].set_title("Оригінальне зображення", fontsize=12)
axes_img[0].axis('off')

print("\nВиконання обробки та стиснення зображення...")

for idx, (config_name, q_val) in enumerate(configs.items()):
    
    coeffs, rec_img = process_image(img_gray, q_val)
    all_coeffs_dict[config_name] = coeffs 
    
    psnr_val = psnr_metric(img_gray, rec_img)
    ssim_val = ssim_metric(img_gray, rec_img)
    
    axes_img[idx+1].imshow(rec_img, cmap='gray', vmin=0, vmax=255)
    axes_img[idx+1].set_title(f"{config_name}\nPSNR: {psnr_val:.2f} дБ | SSIM: {ssim_val:.4f}", fontsize=11)
    axes_img[idx+1].axis('off')
    
    # Вимірювання часу та стиснення Хаффмана
    start_t = time.time()
    huff_codec = HuffmanCodec.from_data(coeffs)
    huff_encoded = huff_codec.encode(coeffs)
    huff_time = time.time() - start_t
    huff_size_bits = len(huff_encoded) * 8
    
    # Вимірювання часу та арифметичне стиснення
    start_t = time.time()
    coeffs_str = [str(c) for c in coeffs]
    freq = dict(Counter(coeffs_str))
    total_symbols = len(coeffs_str)
    
    entropy = -sum((count / total_symbols) * math.log2(count / total_symbols) for count in freq.values())
    
    try:
        if len(freq) > 1:
            total_freq = sum(freq.values())
            scale_factor = max(1.0, total_freq / 4096.0)
            scaled_freq = {k: max(1, int(v / scale_factor)) for k, v in freq.items()}
            
            model = StaticModel(scaled_freq) 
            arith_coder = AECompressor(model)
            arith_encoded = arith_coder.compress(coeffs_str)
            arith_size_bits = len(arith_encoded)
        else:
            arith_size_bits = 0
    except Exception as e:
        arith_size_bits = int(entropy * total_symbols)
        
    arith_time = time.time() - start_t
    
    huff_cr = original_bits / huff_size_bits if huff_size_bits > 0 else 0
    arith_cr = original_bits / arith_size_bits if arith_size_bits > 0 else 0
    
    results.append({
        "Конфігурація": config_name,
        "Ентропія": entropy,
        "Унікальних символів": len(freq),
        "PSNR (дБ)": psnr_val,
        "SSIM": ssim_val,
        "CR (Хаффман)": huff_cr,
        "CR (Арифм.)": arith_cr,
        "Час (Хаффман, с)": huff_time,
        "Час (Арифм., с)": arith_time
    })

plt.tight_layout()
plt.subplots_adjust(bottom=0.12)
plt.figtext(0.5, 0.01, "Рис. 2. Візуальне порівняння якості відновлених зображень при різних рівнях квантування", wrap=True, horizontalalignment='center', fontsize=14)
plt.show()

# Гістограми розподілу значень
fig_hist, axes_hist = plt.subplots(1, 3, figsize=(20, 4.5))

for idx, (config_name, coeffs) in enumerate(all_coeffs_dict.items()):
    filtered_coeffs = [c for c in coeffs if -50 <= c <= 50]
    axes_hist[idx].hist(filtered_coeffs, bins=50, color='rebeccapurple', alpha=0.8)
    axes_hist[idx].set_title(f"{config_name}", fontsize=12)
    axes_hist[idx].set_xlabel("Значення коефіцієнта", fontsize=10)
    axes_hist[idx].set_ylabel("Частота (логарифмічна шкала)", fontsize=10)
    axes_hist[idx].set_yscale('log') 
    axes_hist[idx].grid(True, alpha=0.4, linestyle='--')

plt.tight_layout()
plt.subplots_adjust(bottom=0.18)
plt.figtext(0.5, 0.02, "Рис. 3. Розподіл значень квантованих коефіцієнтів ДКП (вплив на ентропію)", wrap=True, horizontalalignment='center', fontsize=14)
plt.show()

# Зведена таблиця результатів
df = pd.DataFrame(results)
print("\n--- Зведені результати порівняльного аналізу ---")
display(df.round(4))

# Порівняльні графіки ефективності
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 5))

x = np.arange(len(configs))
width = 0.35

# Коефіцієнт стиснення
ax1.bar(x - width/2, df["CR (Хаффман)"], width, label='Кодування Хаффмана', color='royalblue')
ax1.bar(x + width/2, df["CR (Арифм.)"], width, label='Арифметичне кодування', color='darkorange')
ax1.set_ylabel('Коефіцієнт стиснення (CR)', fontsize=11)
ax1.set_title('Ефективність стиснення', fontsize=12)
ax1.set_xticks(x)
ax1.set_xticklabels(df["Конфігурація"], fontsize=11)
ax1.legend(fontsize=10)
ax1.grid(axis='y', linestyle='--', alpha=0.7)
for i in range(len(configs)):
    ax1.text(i - width/2, df["CR (Хаффман)"].iloc[i], f'{df["CR (Хаффман)"].iloc[i]:.2f}', ha='center', va='bottom', fontweight='bold')
    ax1.text(i + width/2, df["CR (Арифм.)"].iloc[i], f'{df["CR (Арифм.)"].iloc[i]:.2f}', ha='center', va='bottom', fontweight='bold')

# Затримка алгоритмів
ax2.bar(x - width/2, df["Час (Хаффман, с)"], width, label='Кодування Хаффмана', color='royalblue')
ax2.bar(x + width/2, df["Час (Арифм., с)"], width, label='Арифметичне кодування', color='darkorange')
ax2.set_ylabel('Час виконання (секунди)', fontsize=11)
ax2.set_title('Затримка алгоритмів', fontsize=12)
ax2.set_xticks(x)
ax2.set_xticklabels(df["Конфігурація"], fontsize=11)
if df["Час (Арифм., с)"].max() > 10 * df["Час (Хаффман, с)"].max():
    ax2.set_yscale('log') 
    ax2.set_ylabel('Час виконання (секунди) - Логарифмічна шкала', fontsize=11)
ax2.legend(fontsize=10)
ax2.grid(axis='y', linestyle='--', alpha=0.7)
for i in range(len(configs)):
    ax2.text(i - width/2, df["Час (Хаффман, с)"].iloc[i], f'{df["Час (Хаффман, с)"].iloc[i]:.3f} с', ha='center', va='bottom', fontsize=9)
    ax2.text(i + width/2, df["Час (Арифм., с)"].iloc[i], f'{df["Час (Арифм., с)"].iloc[i]:.2f} с', ha='center', va='bottom', fontsize=9)

plt.tight_layout()
plt.subplots_adjust(bottom=0.15) 
plt.figtext(0.5, 0.02, "Рис. 4. Порівняльний аналіз ефективності стиснення та затримки ентропійних алгоритмів", wrap=True, horizontalalignment='center', fontsize=14)
plt.show()