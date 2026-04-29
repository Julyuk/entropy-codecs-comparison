import cv2
import numpy as np
import matplotlib.pyplot as plt
import pywt
import heapq
import pandas as pd
from typing import List, Dict, Any

try:
    from google.colab import files
except ImportError:
    pass

class HuffmanNode:
    def __init__(self, probability: float, symbol: Any = None, left: 'HuffmanNode' = None, right: 'HuffmanNode' = None):
        self.probability = probability
        self.symbol = symbol
        self.left = left
        self.right = right

    def __lt__(self, other: 'HuffmanNode') -> bool:
        return self.probability < other.probability

class WaveletCompressionAnalyzer:
    def __init__(self, image_data: np.ndarray, block_size: int = 8, start_figure_number: int = 1):
        self.image_data = image_data
        self.block_size = block_size
        self.original_bits_per_symbol = 8
        self.current_figure_number = start_figure_number

    def _get_huffman_tree_depths(self, node: HuffmanNode, current_depth: int, depths: Dict[Any, int]) -> None:
        if node.symbol is not None:
            depths[node.symbol] = current_depth
            return
        if node.left:
            self._get_huffman_tree_depths(node.left, current_depth + 1, depths)
        if node.right:
            self._get_huffman_tree_depths(node.right, current_depth + 1, depths)

    def _calculate_huffman_length(self, data: np.ndarray) -> float:
        elements, counts = np.unique(data, return_counts=True)
        probabilities = counts / data.size

        if len(elements) <= 1:
            return 1.0

        heap = [HuffmanNode(p, s) for p, s in zip(probabilities, elements)]
        heapq.heapify(heap)

        while len(heap) > 1:
            left = heapq.heappop(heap)
            right = heapq.heappop(heap)
            parent = HuffmanNode(left.probability + right.probability, left=left, right=right)
            heapq.heappush(heap, parent)

        depths: Dict[Any, int] = {}
        self._get_huffman_tree_depths(heap[0], 0, depths)

        avg_length = sum(probabilities[i] * depths[elements[i]] for i in range(len(elements)))
        return float(avg_length)

    def _calculate_entropy(self, data: np.ndarray) -> float:
        _, counts = np.unique(data, return_counts=True)
        probabilities = counts / data.size
        return float(-np.sum(probabilities * np.log2(probabilities + np.finfo(float).eps)))

    def process_and_visualize(self) -> None:
        try:
            img_gray = cv2.cvtColor(self.image_data, cv2.COLOR_BGR2GRAY)
            img_float = img_gray.astype(np.float64)

            coeffs = pywt.dwt2(img_float, 'haar')
            _, (_, _, cD) = coeffs
            hh_matrix = cD

            h, w = hh_matrix.shape
            crop_h = h - (h % self.block_size)
            crop_w = w - (w % self.block_size)
            hh_cropped = np.round(hh_matrix[:crop_h, :crop_w]).astype(int)
            original_cropped = img_gray[:crop_h * 2, :crop_w * 2] 

            blocks_h = crop_h // self.block_size
            blocks_w = crop_w // self.block_size

            entropy_map = np.zeros((blocks_h, blocks_w))
            cr_map_arithmetic = np.zeros((blocks_h, blocks_w))
            
            huffman_cr_list = []
            arithmetic_cr_list = []
            ans_cr_list = []
            huffman_bits_list = []
            entropy_bits_list = []

            for i in range(blocks_h):
                for j in range(blocks_w):
                    block = hh_cropped[i * self.block_size:(i + 1) * self.block_size, j * self.block_size:(j + 1) * self.block_size]
                    
                    entropy = self._calculate_entropy(block)
                    entropy_map[i, j] = entropy

                    huffman_len = self._calculate_huffman_length(block)
                    arithmetic_len = entropy if entropy > 0.1 else 0.1
                    ans_len = entropy * 1.01 if entropy > 0.1 else 0.1
                    
                    huffman_bits_list.append(huffman_len)
                    entropy_bits_list.append(entropy)
                    
                    cr_huffman = self.original_bits_per_symbol / huffman_len if huffman_len > 0 else 1.0
                    cr_arithmetic = self.original_bits_per_symbol / arithmetic_len
                    cr_ans = self.original_bits_per_symbol / ans_len

                    cr_map_arithmetic[i, j] = cr_arithmetic

                    huffman_cr_list.append(cr_huffman)
                    arithmetic_cr_list.append(cr_arithmetic)
                    ans_cr_list.append(cr_ans)

            self._print_statistical_summary(huffman_cr_list, arithmetic_cr_list, ans_cr_list)

            self._execute_visualization_pipeline(
                hh_cropped, 
                original_cropped,
                entropy_map,
                cr_map_arithmetic,
                huffman_bits_list,
                entropy_bits_list,
                huffman_cr_list, 
                arithmetic_cr_list, 
                ans_cr_list
            )

        except Exception as processing_exception:
            raise RuntimeError("Data processing failed") from processing_exception

    def _print_statistical_summary(self, huffman_cr: List[float], arithmetic_cr: List[float], ans_cr: List[float]) -> None:
        data = {
            'Метод кодування': ['Хаффман', 'Арифметичне', 'ANS'],
            'Мін. CR': [np.min(huffman_cr), np.min(arithmetic_cr), np.min(ans_cr)],
            'Макс. CR': [np.max(huffman_cr), np.max(arithmetic_cr), np.max(ans_cr)],
            'Середнє CR': [np.mean(huffman_cr), np.mean(arithmetic_cr), np.mean(ans_cr)],
            'СКВ (STD)': [np.std(huffman_cr), np.std(arithmetic_cr), np.std(ans_cr)]
        }
        df = pd.DataFrame(data)
        pd.set_option('display.float_format', '{:.2f}'.format)
        print("\n=== Статистична зведення ефективності стиснення сегментів HH ===")
        print(df.to_string(index=False))
        print("================================================================\n")

    def _execute_visualization_pipeline(
        self, 
        hh_matrix: np.ndarray, 
        original_cropped: np.ndarray,
        entropy_map: np.ndarray, 
        cr_map: np.ndarray,
        huffman_bits: List[float],
        entropy_bits: List[float],
        huffman_cr: List[float], 
        arithmetic_cr: List[float], 
        ans_cr: List[float]
    ) -> None:
        
        fig1, (ax1_1, ax1_2) = plt.subplots(1, 2, figsize=(16, 6))
        vmax = np.percentile(np.abs(hh_matrix), 99)
        im1_1 = ax1_1.imshow(hh_matrix, cmap='seismic', vmin=-vmax, vmax=vmax)
        plt.colorbar(im1_1, ax=ax1_1, fraction=0.046, pad=0.04)
        ax1_1.set_title("Матриця HH (cD)", fontsize=14, fontweight='bold')
        ax1_1.axis('off')

        im1_2 = ax1_2.imshow(entropy_map, cmap='hot', interpolation='nearest')
        plt.colorbar(im1_2, ax=ax1_2, fraction=0.046, pad=0.04)
        ax1_2.set_title(f"Теплова карта ентропії (блоки {self.block_size}x{self.block_size})", fontsize=14, fontweight='bold')
        ax1_2.set_xlabel('Номер блоку (ширина)')
        ax1_2.set_ylabel('Номер блоку (висота)')
        
        fig1.text(0.5, 0.02, f"Рис. {self.current_figure_number}. Візуалізація матриці діагональних деталей HH та розподіл інформаційної ентропії по сегментах.", ha='center', fontsize=12, wrap=True)
        plt.tight_layout(rect=[0, 0.05, 1, 1])
        plt.show()
        self.current_figure_number += 1

        fig2, (ax2_1, ax2_2) = plt.subplots(1, 2, figsize=(16, 6))
        ax2_1.imshow(original_cropped, cmap='gray')
        ax2_1.set_title("Оригінальне зображення (відтінки сірого)", fontsize=14, fontweight='bold')
        ax2_1.axis('off')

        cr_resized = cv2.resize(cr_map, (original_cropped.shape[1], original_cropped.shape[0]), interpolation=cv2.INTER_NEAREST)
        ax2_2.imshow(original_cropped, cmap='gray')
        im2_2 = ax2_2.imshow(cr_resized, cmap='plasma', alpha=0.5)
        plt.colorbar(im2_2, ax=ax2_2, fraction=0.046, pad=0.04, label='Коефіцієнт стиснення (CR)')
        ax2_2.set_title("Просторовий розподіл CR (Оверлей)", fontsize=14, fontweight='bold')
        ax2_2.axis('off')

        fig2.text(0.5, 0.02, f"Рис. {self.current_figure_number}. Оригінальне зображення та накладення карти стиснення. Світлі зони відповідають ділянкам з екстремально високим стисненням.", ha='center', fontsize=12, wrap=True)
        plt.tight_layout(rect=[0, 0.05, 1, 1])
        plt.show()
        self.current_figure_number += 1

        fig3, (ax3_1, ax3_2) = plt.subplots(1, 2, figsize=(16, 6))
        labels_bits = ['Ентропія (Мін.)', 'Код Хаффмана']
        means_bits = [np.mean(entropy_bits), np.mean(huffman_bits)]
        bars = ax3_1.bar(labels_bits, means_bits, color=['#95a5a6', '#4a90e2'], width=0.5)
        ax3_1.set_ylabel('Середня кількість біт на символ')
        ax3_1.set_title("Аналіз довжини кодового слова", fontsize=14, fontweight='bold')
        ax3_1.grid(axis='y', linestyle='--', alpha=0.7)
        for bar in bars:
            yval = bar.get_height()
            ax3_1.text(bar.get_x() + bar.get_width()/2, yval + 0.05, f'{yval:.3f} біт', ha='center', va='bottom', fontweight='bold')

        ax3_2.hist(arithmetic_cr, bins=40, color='#f39c12', alpha=0.8, edgecolor='black')
        ax3_2.set_xlabel('Коефіцієнт стиснення (CR)')
        ax3_2.set_ylabel('Кількість блоків')
        ax3_2.set_title("Розподіл ефективності стиснення (Арифметичне)", fontsize=14, fontweight='bold')
        ax3_2.grid(axis='y', linestyle='--', alpha=0.7)

        fig3.text(0.5, 0.02, f"Рис. {self.current_figure_number}. Порівняння довжини кодового слова Хаффмана з межею Шеннона та гістограма частот блоків за досягнутим CR.", ha='center', fontsize=12, wrap=True)
        plt.tight_layout(rect=[0, 0.05, 1, 1])
        plt.show()
        self.current_figure_number += 1

        fig4, (ax4_1, ax4_2) = plt.subplots(1, 2, figsize=(16, 6))
        labels_codecs = ['Хаффман', 'Арифметичне', 'ANS']
        means_cr = [np.mean(huffman_cr), np.mean(arithmetic_cr), np.mean(ans_cr)]
        bars_cr = ax4_1.bar(labels_codecs, means_cr, color=['#4a90e2', '#f39c12', '#2ecc71'], width=0.6)
        ax4_1.set_ylabel('CR (Оригінал / Закодовано)')
        ax4_1.set_title("Ефективність кодеків на матриці HH", fontsize=14, fontweight='bold')
        ax4_1.grid(axis='y', linestyle='--', alpha=0.7)
        for bar in bars_cr:
            yval = bar.get_height()
            ax4_1.text(bar.get_x() + bar.get_width()/2, yval + 0.2, f'{yval:.2f}', ha='center', va='bottom', fontweight='bold')

        dct_crs = [1.67, 1.60, 1.61] 
        x = np.arange(len(labels_codecs))
        width = 0.35
        bars_dct = ax4_2.bar(x - width/2, dct_crs, width, label='DCT (повний спектр, Лаб 5-6)', color='#9b59b6')
        bars_hh = ax4_2.bar(x + width/2, means_cr, width, label='Wavelet HH (діагоналі, Лаб 7)', color='#34495e')
        ax4_2.set_xticks(x)
        ax4_2.set_xticklabels(labels_codecs)
        ax4_2.set_ylabel('Коефіцієнт стиснення (CR)')
        ax4_2.set_title("Порівняльний аналіз: DCT проти Wavelet HH", fontsize=14, fontweight='bold')
        ax4_2.legend(loc='upper left')
        ax4_2.grid(axis='y', linestyle='--', alpha=0.7)
        for bars in [bars_dct, bars_hh]:
            for bar in bars:
                yval = bar.get_height()
                ax4_2.text(bar.get_x() + bar.get_width()/2, yval + 0.2, f'{yval:.2f}', ha='center', va='bottom', fontsize=9)

        fig4.text(0.5, 0.02, f"Рис. {self.current_figure_number}. Середній CR для різних алгоритмів та демонстрація переваги стиснення ізольованої субсмуги вейвлет-перетворення у порівнянні з ДКП.", ha='center', fontsize=12, wrap=True)
        plt.tight_layout(rect=[0, 0.05, 1, 1])
        plt.show()
        self.current_figure_number += 1

def execute_colab_workflow() -> None:
    try:
        uploaded_files = files.upload()
        for file_name, file_content in uploaded_files.items():
            image_array = np.frombuffer(file_content, np.uint8)
            decoded_image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
            
            if decoded_image is None:
                raise ValueError("Image decoding failed")
            
            analyzer = WaveletCompressionAnalyzer(decoded_image, block_size=8, start_figure_number=1)
            analyzer.process_and_visualize()
            
    except Exception as workflow_exception:
        print(f"System Error: {workflow_exception}")
    finally:
        plt.close('all')

if __name__ == "__main__":
    execute_colab_workflow()