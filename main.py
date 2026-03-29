import os
import io
import requests
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# =========================
# НАСТРОЙКИ
# =========================
ORIGIN = "https://www.slavcorpora.ru"
SAMPLE_ID = "b008ae91-32cf-4d7d-84e4-996144e4edb7"

# Порог для перевода полутонового изображения в монохромное
BIN_THRESHOLD = 128

# Коэффициент усиления разностного изображения
DIFF_CONTRAST = 10

RAW_DIR = "raw_images"
GRAY_DIR = "gray_bmp"
MONO_DIR = "mono_bmp"
FILTERED_GRAY_DIR = "filtered_gray_rank"
FILTERED_MONO_DIR = "filtered_mono_rank"
DIFF_DIR = "difference_images"
DIFF_X10_DIR = "difference_images_x10"
XOR_DIR = "xor_mono"
DEMO_DIR = "demo"

for folder in [
    RAW_DIR,
    GRAY_DIR,
    MONO_DIR,
    FILTERED_GRAY_DIR,
    FILTERED_MONO_DIR,
    DIFF_DIR,
    DIFF_X10_DIR,
    XOR_DIR,
    DEMO_DIR
]:
    os.makedirs(folder, exist_ok=True)


# =========================
# ПОЛУЧЕНИЕ СПИСКА ИЗОБРАЖЕНИЙ
# =========================
def get_image_urls(origin: str, sample_id: str):
    url = f"{origin}/api/samples/{sample_id}"
    response = requests.get(url, timeout=30)
    response.raise_for_status()
    sample_data = response.json()
    return [f"{origin}/images/{page['filename']}" for page in sample_data["pages"]]


# =========================
# ЗАГРУЗКА ИЗОБРАЖЕНИЯ
# =========================
def download_image(url: str) -> Image.Image:
    response = requests.get(url, timeout=60)
    response.raise_for_status()
    return Image.open(io.BytesIO(response.content))


# =========================
# ПЕРЕВОД В ПОЛУТОН
# Без convert('L') как готового решения
# Y = 0.299R + 0.587G + 0.114B
# =========================
def rgb_to_grayscale_manual(img: Image.Image) -> Image.Image:
    rgb_img = img.convert("RGB")
    arr = np.array(rgb_img, dtype=np.float32)

    r = arr[:, :, 0]
    g = arr[:, :, 1]
    b = arr[:, :, 2]

    gray = 0.299 * r + 0.587 * g + 0.114 * b
    gray = np.clip(gray, 0, 255).astype(np.uint8)

    return Image.fromarray(gray, mode="L")


# =========================
# ПЕРЕВОД В МОНОХРОМ
# Ручная пороговая обработка
# =========================
def grayscale_to_monochrome_manual(gray_img: Image.Image, threshold: int = 128) -> Image.Image:
    gray = np.array(gray_img, dtype=np.uint8)
    mono = np.where(gray >= threshold, 255, 0).astype(np.uint8)
    return Image.fromarray(mono, mode="L")


# =========================
# РАНГОВЫЙ ФИЛЬТР
# Вариант 6:
# Маска "косой крест"
# 1 0 1
# 0 1 0
# 1 0 1
#
# Ранг 3/5 = третий элемент после сортировки 5 значений
# Работает и для полутона, и для монохрома
# =========================
def rank_filter_sparse_diagonal_cross(img_l: Image.Image) -> Image.Image:
    arr = np.array(img_l, dtype=np.uint8)
    h, w = arr.shape

    padded = np.pad(arr, pad_width=1, mode="edge")
    filtered = np.zeros((h, w), dtype=np.uint8)

    mask_offsets = [(-1, -1), (-1, 1), (0, 0), (1, -1), (1, 1)]

    for y in range(h):
        for x in range(w):
            cy = y + 1
            cx = x + 1

            values = []
            for dy, dx in mask_offsets:
                values.append(int(padded[cy + dy, cx + dx]))

            values.sort()
            filtered[y, x] = values[2]  # ранг 3/5

    return Image.fromarray(filtered, mode="L")


# =========================
# МОДУЛЬ РАЗНОСТИ ДЛЯ ПОЛУТОНА
# |I - F|
# =========================
def make_difference_image(original_img: Image.Image, filtered_img: Image.Image) -> Image.Image:
    original_arr = np.array(original_img, dtype=np.int16)
    filtered_arr = np.array(filtered_img, dtype=np.int16)

    diff = np.abs(original_arr - filtered_arr)
    diff = np.clip(diff, 0, 255).astype(np.uint8)

    return Image.fromarray(diff, mode="L")


# =========================
# УСИЛЕНИЕ РАЗНОСТИ
# =========================
def amplify_difference_image(diff_img: Image.Image, factor: int = 10) -> Image.Image:
    diff_arr = np.array(diff_img, dtype=np.int16)
    amplified = diff_arr * factor
    amplified = np.clip(amplified, 0, 255).astype(np.uint8)
    return Image.fromarray(amplified, mode="L")


# =========================
# XOR ДЛЯ МОНОХРОМА
# 255 - если пиксели различаются
# 0   - если одинаковые
# =========================
def xor_monochrome_images(img1: Image.Image, img2: Image.Image) -> Image.Image:
    arr1 = np.array(img1, dtype=np.uint8)
    arr2 = np.array(img2, dtype=np.uint8)

    xor_result = np.where(arr1 != arr2, 255, 0).astype(np.uint8)
    return Image.fromarray(xor_result, mode="L")


# =========================
# СОХРАНЕНИЕ СРАВНЕНИЯ 2 КАРТИНОК
# =========================
def save_comparison(img1: Image.Image, img2: Image.Image,
                    title1: str, title2: str, out_path: str):
    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    if img1.mode == "L":
        plt.imshow(img1, cmap="gray", vmin=0, vmax=255, interpolation="nearest")
    else:
        plt.imshow(img1, interpolation="nearest")
    plt.title(title1)
    plt.axis("off")

    plt.subplot(1, 2, 2)
    if img2.mode == "L":
        plt.imshow(img2, cmap="gray", vmin=0, vmax=255, interpolation="nearest")
    else:
        plt.imshow(img2, interpolation="nearest")
    plt.title(title2)
    plt.axis("off")

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()


# =========================
# СОХРАНЕНИЕ СРАВНЕНИЯ 4 КАРТИНОК
# =========================
def save_quad_comparison(img1: Image.Image, img2: Image.Image, img3: Image.Image, img4: Image.Image,
                         out_path: str,
                         main_title: str,
                         title1: str,
                         title2: str,
                         title3: str,
                         title4: str):
    plt.figure(figsize=(18, 10))
    plt.suptitle(main_title, fontsize=16, y=0.98)

    images = [img1, img2, img3, img4]
    titles = [title1, title2, title3, title4]

    for i, (img, title) in enumerate(zip(images, titles), start=1):
        plt.subplot(2, 2, i)
        if img.mode == "L":
            plt.imshow(img, cmap="gray", vmin=0, vmax=255, interpolation="nearest")
        else:
            plt.imshow(img, interpolation="nearest")
        plt.title(title, fontsize=12)
        plt.axis("off")

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()


# =========================
# ОСНОВНАЯ ФУНКЦИЯ
# =========================
def process_images(max_images=None):
    image_urls = get_image_urls(ORIGIN, SAMPLE_ID)

    if max_images is not None:
        image_urls = image_urls[:max_images]

    print(f"Изображений для обработки: {len(image_urls)}")

    for i, url in enumerate(image_urls, start=1):
        print(f"[{i}/{len(image_urls)}] Загружается: {url}")

        try:
            img = download_image(url)
            base_name = f"image_{i:02d}"

            # 1. Исходное изображение
            raw_path = os.path.join(RAW_DIR, f"{base_name}.png")
            img.convert("RGB").save(raw_path, format="PNG")

            # 2. Полутоновое изображение
            gray_img = rgb_to_grayscale_manual(img)
            gray_path = os.path.join(GRAY_DIR, f"{base_name}_gray.bmp")
            gray_img.save(gray_path, format="BMP")

            # 3. Монохромное изображение
            mono_img = grayscale_to_monochrome_manual(gray_img, threshold=BIN_THRESHOLD)
            mono_path = os.path.join(MONO_DIR, f"{base_name}_mono.bmp")
            mono_img.save(mono_path, format="BMP")

            # 4. Ранговая фильтрация полутона
            filtered_gray_img = rank_filter_sparse_diagonal_cross(gray_img)
            filtered_gray_path = os.path.join(
                FILTERED_GRAY_DIR,
                f"{base_name}_filtered_gray_rank3of5.bmp"
            )
            filtered_gray_img.save(filtered_gray_path, format="BMP")

            # 5. Ранговая фильтрация монохрома
            filtered_mono_img = rank_filter_sparse_diagonal_cross(mono_img)
            filtered_mono_path = os.path.join(
                FILTERED_MONO_DIR,
                f"{base_name}_filtered_mono_rank3of5.bmp"
            )
            filtered_mono_img.save(filtered_mono_path, format="BMP")

            # 6. Разность изображения |I - F| для полутона
            diff_img = make_difference_image(gray_img, filtered_gray_img)
            diff_path = os.path.join(DIFF_DIR, f"{base_name}_difference.bmp")
            diff_img.save(diff_path, format="BMP")

            # 7. Разность изображения, умноженная на 10
            diff_x10_img = amplify_difference_image(diff_img, factor=DIFF_CONTRAST)
            diff_x10_path = os.path.join(DIFF_X10_DIR, f"{base_name}_difference_x{DIFF_CONTRAST}.bmp")
            diff_x10_img.save(diff_x10_path, format="BMP")

            # 8. XOR для монохрома
            xor_img = xor_monochrome_images(mono_img, filtered_mono_img)
            xor_path = os.path.join(XOR_DIR, f"{base_name}_xor_mono.bmp")
            xor_img.save(xor_path, format="BMP")

            # =========================
            # DEMO
            # =========================
            demo_gray_path = os.path.join(DEMO_DIR, f"{base_name}_gray_demo.png")
            save_comparison(
                img.convert("RGB"),
                gray_img,
                "Исходное изображение",
                "Полутоновое изображение",
                demo_gray_path
            )

            demo_mono_path = os.path.join(DEMO_DIR, f"{base_name}_mono_demo.png")
            save_comparison(
                gray_img,
                mono_img,
                "Полутоновое изображение",
                f"Монохромное изображение (threshold={BIN_THRESHOLD})",
                demo_mono_path
            )

            demo_filtered_gray_path = os.path.join(DEMO_DIR, f"{base_name}_filtered_gray_demo.png")
            save_comparison(
                gray_img,
                filtered_gray_img,
                "Полутоновое изображение",
                "Ранговая фильтрация полутона",
                demo_filtered_gray_path
            )

            demo_filtered_mono_path = os.path.join(DEMO_DIR, f"{base_name}_filtered_mono_demo.png")
            save_comparison(
                mono_img,
                filtered_mono_img,
                "Монохромное изображение",
                "Ранговая фильтрация монохрома",
                demo_filtered_mono_path
            )

            demo_diff_path = os.path.join(DEMO_DIR, f"{base_name}_diff_demo.png")
            save_comparison(
                gray_img,
                diff_img,
                "Полутоновое изображение",
                "Разность |I - F|",
                demo_diff_path
            )

            demo_diff_x10_path = os.path.join(DEMO_DIR, f"{base_name}_diff_x10_demo.png")
            save_comparison(
                diff_img,
                diff_x10_img,
                "Разность |I - F|",
                f"Разность ×{DIFF_CONTRAST}",
                demo_diff_x10_path
            )

            demo_xor_path = os.path.join(DEMO_DIR, f"{base_name}_xor_demo.png")
            save_comparison(
                mono_img,
                xor_img,
                "Монохромное изображение",
                "XOR монохрома",
                demo_xor_path
            )

            demo_gray_quad_path = os.path.join(DEMO_DIR, f"{base_name}_gray_pipeline.png")
            save_quad_comparison(
                gray_img,
                filtered_gray_img,
                diff_img,
                diff_x10_img,
                demo_gray_quad_path,
                main_title="Обработка полутонового изображения",
                title1="Полутон",
                title2="Ранговая фильтрация",
                title3="Разность",
                title4=f"Разность ×{DIFF_CONTRAST}"
            )

            demo_mono_quad_path = os.path.join(DEMO_DIR, f"{base_name}_mono_pipeline.png")
            save_quad_comparison(
                mono_img,
                filtered_mono_img,
                xor_img,
                img.convert("RGB"),
                demo_mono_quad_path,
                main_title="Обработка монохромного изображения",
                title1="Монохром",
                title2="Ранговая фильтрация",
                title3="XOR",
                title4="Исходное изображение"
            )

            print("  Сохранено:")
            print(f"    исходник:                 {raw_path}")
            print(f"    полутон:                  {gray_path}")
            print(f"    монохром:                 {mono_path}")
            print(f"    фильтр полутона:          {filtered_gray_path}")
            print(f"    фильтр монохрома:         {filtered_mono_path}")
            print(f"    разность:                 {diff_path}")
            print(f"    разность x{DIFF_CONTRAST}:          {diff_x10_path}")
            print(f"    XOR монохром:             {xor_path}")
            print(f"    demo gray:                {demo_gray_path}")
            print(f"    demo mono:                {demo_mono_path}")
            print(f"    demo filtered gray:       {demo_filtered_gray_path}")
            print(f"    demo filtered mono:       {demo_filtered_mono_path}")
            print(f"    demo diff:                {demo_diff_path}")
            print(f"    demo diff x{DIFF_CONTRAST}:           {demo_diff_x10_path}")
            print(f"    demo xor:                 {demo_xor_path}")
            print(f"    pipeline gray:            {demo_gray_quad_path}")
            print(f"    pipeline mono:            {demo_mono_quad_path}")

        except Exception as e:
            print(f"Ошибка при обработке {url}: {e}")


if __name__ == "__main__":
    process_images(max_images=5)   # можно поменять число
    # process_images()              # обработать все изображения
    print("\nГотово.")