"""
Лабораторна робота № 2. Фільтрація, покращення та сегментація зображень.

Скрипт послідовно виконує всі пункти завдання:
    1. Генерація шуму двох типів: гаусів (адитивний) та "сіль і перець" (імпульсний).
    2. Фільтрація: Gaussian filter, Median filter, Bilateral filter.
    3. Покращення контрасту: Histogram Equalization vs CLAHE.
    4. Сегментація/бінаризація: метод Otsu vs Adaptive Thresholding
       (у тому числі при нерівномірному освітленні).
    5. Дослідницький експеримент: три рівні шуму sigma = {0.05, 0.1, 0.2}
       для ОБОХ типів шуму (гаусів і сіль-перець), порівняння фільтрів,
       побудова залежності PSNR = f(sigma).
    6. Міні-дослідження: автоматичний підбір оптимальних параметрів фільтра
       (розмір ядра / діаметр) за критерієм максимуму PSNR.

Усі проміжні зображення, таблиці (CSV) та графіки (PNG) зберігаються у теку
"results/lab2".
"""

import os
import csv

import cv2
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from skimage import data
from skimage.metrics import structural_similarity as sk_ssim
from skimage.metrics import peak_signal_noise_ratio as sk_psnr

IMAGES_DIR = "images"
RESULTS_DIR = os.path.join("results", "lab2")
os.makedirs(IMAGES_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)


def result_path(filename: str) -> str:
    return os.path.join(RESULTS_DIR, filename)


def load_source_image() -> np.ndarray:
    """Те саме вихідне зображення, що й у ЛР № 1 (для узгодженості результатів)."""
    for name in ("my_image.png", "my_image.jpg", "my_image.jpeg", "my_image.bmp"):
        path = os.path.join(IMAGES_DIR, name)
        if os.path.exists(path):
            print(f"Використовую власне зображення: {path}")
            return cv2.imread(path, cv2.IMREAD_COLOR)

    print("Власне зображення не знайдено -> використовую вбудоване demo-зображення "
          "skimage.data.astronaut(). Покладіть своє фото у images/my_image.png, "
          "щоб лабораторна виконувалась саме на ньому.")
    rgb = data.astronaut()
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def mse(img1: np.ndarray, img2: np.ndarray) -> float:
    return float(np.mean((img1.astype(np.float64) - img2.astype(np.float64)) ** 2))


def psnr(img1: np.ndarray, img2: np.ndarray) -> float:
    return float(sk_psnr(img1, img2, data_range=255))


def ssim(img1: np.ndarray, img2: np.ndarray) -> float:
    if img1.ndim == 3:
        return float(sk_ssim(img1, img2, data_range=255, channel_axis=2))
    return float(sk_ssim(img1, img2, data_range=255))


# ---------------------------------------------------------------------------
# 1. Генерація шуму.
# ---------------------------------------------------------------------------
def add_gaussian_noise(img: np.ndarray, sigma: float, rng: np.random.Generator) -> np.ndarray:
    """
    Адитивний гаусів шум: I_noisy = I + N(0, sigma^2).
    sigma задається у відносних одиницях (частка від діапазону 0..255),
    тому реальне стандартне відхилення шуму = sigma * 255.
    """
    noise = rng.normal(0, sigma * 255.0, img.shape)
    noisy = img.astype(np.float64) + noise
    return np.clip(noisy, 0, 255).astype(np.uint8)


def add_salt_and_pepper_noise(img: np.ndarray, amount: float, rng: np.random.Generator) -> np.ndarray:
    """
    Імпульсний шум "сіль і перець": частка `amount` пікселів замінюється
    на 0 (перець) або 255 (сіль), порівну.
    """
    noisy = img.copy()
    mask = rng.random(img.shape[:2])

    salt_mask = mask < (amount / 2)
    pepper_mask = (mask >= (amount / 2)) & (mask < amount)

    if noisy.ndim == 3:
        noisy[salt_mask] = 255
        noisy[pepper_mask] = 0
    else:
        noisy[salt_mask] = 255
        noisy[pepper_mask] = 0
    return noisy


def part1_noise(img_bgr: np.ndarray) -> None:
    print("\n=== 1. Створення шуму ===")
    rng = np.random.default_rng(seed=42)

    gaussian_noisy = add_gaussian_noise(img_bgr, sigma=0.1, rng=rng)
    sp_noisy = add_salt_and_pepper_noise(img_bgr, amount=0.05, rng=rng)

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    for ax, img, title in zip(
            axes,
            [img_bgr, gaussian_noisy, sp_noisy],
            ["Оригінал", "Гаусів шум, sigma=0.1", "Сіль і перець, d=5%"],
    ):
        ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        ax.set_title(title)
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(result_path("01_noise_examples.png"), dpi=130)
    plt.close(fig)


# ---------------------------------------------------------------------------
# 2. Фільтрація: Gaussian, Median, Bilateral.
# ---------------------------------------------------------------------------
def part2_filters(img_bgr: np.ndarray) -> None:
    print("\n=== 2. Фільтри: Gaussian, Median, Bilateral ===")
    rng = np.random.default_rng(seed=7)

    gaussian_noisy = add_gaussian_noise(img_bgr, sigma=0.1, rng=rng)
    sp_noisy = add_salt_and_pepper_noise(img_bgr, amount=0.05, rng=rng)

    gaussian_filtered = cv2.GaussianBlur(gaussian_noisy, (5, 5), sigmaX=1.5)
    bilateral_filtered = cv2.bilateralFilter(gaussian_noisy, d=9, sigmaColor=75, sigmaSpace=75)
    median_on_sp = cv2.medianBlur(sp_noisy, 5)
    median_on_gaussian = cv2.medianBlur(gaussian_noisy, 5)

    results = {
        "Шум без фільтра (гаусів)": psnr(img_bgr, gaussian_noisy),
        "Gaussian filter": psnr(img_bgr, gaussian_filtered),
        "Bilateral filter": psnr(img_bgr, bilateral_filtered),
        "Median на сіль/перець": psnr(img_bgr, median_on_sp),
        "Median на гаусів шум": psnr(img_bgr, median_on_gaussian),
    }
    for name, value in results.items():
        print(f"PSNR: {name} = {value:.2f} дБ")

    fig, axes = plt.subplots(2, 3, figsize=(13, 8))
    images = [
        (gaussian_noisy, "Гаусів шум (до)"),
        (gaussian_filtered, "Gaussian filter"),
        (bilateral_filtered, "Bilateral filter"),
        (sp_noisy, "Сіль і перець (до)"),
        (median_on_sp, "Median filter"),
        (median_on_gaussian, "Median на гаусовому шумі"),
    ]
    for ax, (img, title) in zip(axes.ravel(), images):
        ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        ax.set_title(title)
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(result_path("02_filters_compare.png"), dpi=130)
    plt.close(fig)

    with open(result_path("02_filters_psnr.csv"), "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Варіант", "PSNR, дБ"])
        for name, value in results.items():
            writer.writerow([name, f"{value:.2f}"])


# ---------------------------------------------------------------------------
# 3. Покращення контрасту: Histogram Equalization vs CLAHE.
# ---------------------------------------------------------------------------
def part3_contrast(img_bgr: np.ndarray) -> None:
    print("\n=== 3. Покращення контрасту: Histogram Equalization vs CLAHE ===")

    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    equalized = cv2.equalizeHist(gray)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    clahe_result = clahe.apply(gray)

    fig, axes = plt.subplots(2, 3, figsize=(13, 8))
    images = [(gray, "Оригінал (grayscale)"), (equalized, "Histogram Equalization"),
              (clahe_result, "CLAHE")]
    for ax, (img, title) in zip(axes[0], images):
        ax.imshow(img, cmap="gray")
        ax.set_title(title)
        ax.axis("off")

    for ax, (img, title) in zip(axes[1], images):
        ax.hist(img.ravel(), bins=256, range=(0, 255), color="gray")
        ax.set_title(f"Гістограма: {title}")
        ax.set_xlim(0, 255)

    fig.tight_layout()
    fig.savefig(result_path("03_contrast_compare.png"), dpi=130)
    plt.close(fig)

    print("Histogram Equalization працює глобально й може перепідсилити контраст "
          "там, де він і без того нормальний. CLAHE діє локально (по плитках 8x8) "
          "з обмеженням підсилення (clipLimit), тому не підсилює шум так сильно.")


# ---------------------------------------------------------------------------
# 4. Бінаризація: Otsu vs Adaptive Thresholding.
# ---------------------------------------------------------------------------
def part4_thresholding(img_bgr: np.ndarray) -> None:
    print("\n=== 4. Бінаризація: Otsu vs Adaptive Thresholding ===")

    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape

    # Метод Otsu автоматично шукає один глобальний поріг, що мінімізує
    # внутрішньокласову дисперсію (добре при рівномірному освітленні).
    _, otsu_uniform = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    adaptive_uniform = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, blockSize=15, C=5
    )

    # Симулюємо нерівномірне освітлення: додаємо градієнт яскравості зліва направо.
    gradient = np.tile(np.linspace(-80, 80, w), (h, 1))
    uneven = np.clip(gray.astype(np.float64) + gradient, 0, 255).astype(np.uint8)

    _, otsu_uneven = cv2.threshold(uneven, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    adaptive_uneven = cv2.adaptiveThreshold(
        uneven, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, blockSize=15, C=5
    )

    fig, axes = plt.subplots(2, 3, figsize=(13, 8))
    row1 = [(gray, "Рівномірне освітлення"), (otsu_uniform, "Otsu"),
            (adaptive_uniform, "Adaptive")]
    row2 = [(uneven, "Нерівномірне освітлення"), (otsu_uneven, "Otsu -> псує результат"),
            (adaptive_uneven, "Adaptive -> тримається краще")]
    for ax, (img, title) in zip(axes[0], row1):
        ax.imshow(img, cmap="gray")
        ax.set_title(title)
        ax.axis("off")
    for ax, (img, title) in zip(axes[1], row2):
        ax.imshow(img, cmap="gray")
        ax.set_title(title)
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(result_path("04_thresholding_compare.png"), dpi=130)
    plt.close(fig)

    print("Otsu шукає один поріг для всього зображення -- добре при рівномірному "
          "освітленні, але помиляється при градієнті яскравості. Adaptive "
          "Thresholding обчислює поріг локально для кожної області, тому краще "
          "тримається при нерівномірному освітленні.")


# ---------------------------------------------------------------------------
# 5. Дослідницький експеримент: sigma = {0.05, 0.1, 0.2}, обидва типи шуму,
#    декілька фільтрів, PSNR = f(sigma), SSIM = f(sigma).
# ---------------------------------------------------------------------------
def part5_research_experiment(img_bgr: np.ndarray) -> list:
    print("\n=== 5. Дослідницький експеримент: PSNR = f(sigma) ===")

    sigmas = [0.05, 0.1, 0.2]
    rng = np.random.default_rng(seed=123)
    rows = []

    for sigma in sigmas:
        # ---- Гаусів шум ----
        noisy = add_gaussian_noise(img_bgr, sigma=sigma, rng=rng)
        gaussian_filtered = cv2.GaussianBlur(noisy, (5, 5), sigmaX=1.5)
        median_filtered = cv2.medianBlur(noisy, 5)
        bilateral_filtered = cv2.bilateralFilter(noisy, d=9, sigmaColor=75, sigmaSpace=75)

        variants = {
            "без фільтра": noisy,
            "Gaussian": gaussian_filtered,
            "Median": median_filtered,
            "Bilateral": bilateral_filtered,
        }
        for filter_name, result_img in variants.items():
            rows.append({
                "noise_type": "гаусів",
                "sigma": sigma,
                "filter": filter_name,
                "mse": mse(img_bgr, result_img),
                "psnr": psnr(img_bgr, result_img),
                "ssim": ssim(img_bgr, result_img),
            })

        # ---- Сіль і перець (amount пропорційний sigma, щоб мати три рівні
        # інтенсивності шуму так само, як для гаусового) ----
        sp_noisy = add_salt_and_pepper_noise(img_bgr, amount=sigma, rng=rng)
        gaussian_filtered_sp = cv2.GaussianBlur(sp_noisy, (5, 5), sigmaX=1.5)
        median_filtered_sp = cv2.medianBlur(sp_noisy, 5)
        bilateral_filtered_sp = cv2.bilateralFilter(sp_noisy, d=9, sigmaColor=75, sigmaSpace=75)

        variants_sp = {
            "без фільтра": sp_noisy,
            "Gaussian": gaussian_filtered_sp,
            "Median": median_filtered_sp,
            "Bilateral": bilateral_filtered_sp,
        }
        for filter_name, result_img in variants_sp.items():
            rows.append({
                "noise_type": "сіль-перець",
                "sigma": sigma,
                "filter": filter_name,
                "mse": mse(img_bgr, result_img),
                "psnr": psnr(img_bgr, result_img),
                "ssim": ssim(img_bgr, result_img),
            })

    for row in rows:
        print(f"[{row['noise_type']:12s}] sigma={row['sigma']:.2f} | "
              f"{row['filter']:12s} | MSE={row['mse']:8.2f} | "
              f"PSNR={row['psnr']:6.2f} дБ | SSIM={row['ssim']:.4f}")

    with open(result_path("05_research_experiment.csv"), "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Тип шуму", "sigma", "Фільтр", "MSE", "PSNR, дБ", "SSIM"])
        for row in rows:
            writer.writerow([row["noise_type"], row["sigma"], row["filter"],
                             f"{row['mse']:.2f}", f"{row['psnr']:.2f}", f"{row['ssim']:.4f}"])

    # Графіки PSNR = f(sigma) окремо для кожного типу шуму.
    for noise_type in ["гаусів", "сіль-перець"]:
        fig, ax = plt.subplots(figsize=(7, 5))
        for filter_name in ["без фільтра", "Gaussian", "Median", "Bilateral"]:
            xs = [r["sigma"] for r in rows if r["noise_type"] == noise_type and r["filter"] == filter_name]
            ys = [r["psnr"] for r in rows if r["noise_type"] == noise_type and r["filter"] == filter_name]
            ax.plot(xs, ys, marker="o", label=filter_name)
        ax.set_title(f"PSNR = f(sigma), шум: {noise_type}")
        ax.set_xlabel("sigma (рівень шуму)")
        ax.set_ylabel("PSNR, дБ")
        ax.grid(True, alpha=0.3)
        ax.legend()
        fig.tight_layout()
        fig.savefig(result_path(f"05_psnr_vs_sigma_{noise_type}.png"), dpi=130)
        plt.close(fig)

    return rows


# ---------------------------------------------------------------------------
# 6. Міні-дослідження: автоматичний підбір оптимальних параметрів фільтра.
#
# Ідея: для фіксованого рівня шуму перебираємо різні розміри ядра
# (Gaussian/Median) або діаметр (Bilateral) і обираємо параметр,
# що дає максимальний PSNR відносно оригіналу.
# ---------------------------------------------------------------------------
def part6_auto_tune(img_bgr: np.ndarray) -> None:
    print("\n=== 6. Міні-дослідження: автопідбір параметрів фільтра ===")

    rng = np.random.default_rng(seed=99)
    noisy = add_gaussian_noise(img_bgr, sigma=0.1, rng=rng)

    best = {"filter": None, "param": None, "psnr": -np.inf}
    rows = []

    # Gaussian: перебираємо розмір ядра (непарні числа).
    for k in [3, 5, 7, 9, 11]:
        filtered = cv2.GaussianBlur(noisy, (k, k), sigmaX=0)
        value = psnr(img_bgr, filtered)
        rows.append({"filter": "Gaussian", "param": f"kernel={k}", "psnr": value})
        if value > best["psnr"]:
            best = {"filter": "Gaussian", "param": k, "psnr": value}

    # Median: перебираємо розмір ядра.
    for k in [3, 5, 7, 9, 11]:
        filtered = cv2.medianBlur(noisy, k)
        value = psnr(img_bgr, filtered)
        rows.append({"filter": "Median", "param": f"kernel={k}", "psnr": value})
        if value > best["psnr"]:
            best = {"filter": "Median", "param": k, "psnr": value}

    # Bilateral: перебираємо діаметр сусідства.
    for d in [5, 9, 13, 17]:
        filtered = cv2.bilateralFilter(noisy, d=d, sigmaColor=75, sigmaSpace=75)
        value = psnr(img_bgr, filtered)
        rows.append({"filter": "Bilateral", "param": f"d={d}", "psnr": value})
        if value > best["psnr"]:
            best = {"filter": "Bilateral", "param": d, "psnr": value}

    print(f"Найкращий результат: фільтр={best['filter']}, параметр={best['param']}, "
          f"PSNR={best['psnr']:.2f} дБ")

    with open(result_path("06_auto_tune.csv"), "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Фільтр", "Параметр", "PSNR, дБ"])
        for row in rows:
            writer.writerow([row["filter"], row["param"], f"{row['psnr']:.2f}"])
        writer.writerow(["НАЙКРАЩИЙ", f"{best['filter']} ({best['param']})", f"{best['psnr']:.2f}"])

    fig, ax = plt.subplots(figsize=(7, 5))
    for filter_name in ["Gaussian", "Median", "Bilateral"]:
        xs = list(range(sum(1 for r in rows if r["filter"] == filter_name)))
        ys = [r["psnr"] for r in rows if r["filter"] == filter_name]
        labels = [r["param"] for r in rows if r["filter"] == filter_name]
        ax.plot(labels, ys, marker="o", label=filter_name)
    ax.set_title("Підбір параметрів фільтра (гаусів шум, sigma=0.1)")
    ax.set_xlabel("Параметр фільтра")
    ax.set_ylabel("PSNR, дБ")
    ax.tick_params(axis="x", rotation=45)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(result_path("06_auto_tune.png"), dpi=130)
    plt.close(fig)


def main() -> None:
    img_bgr = load_source_image()

    part1_noise(img_bgr)
    part2_filters(img_bgr)
    part3_contrast(img_bgr)
    part4_thresholding(img_bgr)
    part5_research_experiment(img_bgr)
    part6_auto_tune(img_bgr)

    print("\nЛабораторна робота № 2 виконана. Усі результати у теці:", RESULTS_DIR)


if __name__ == "__main__":
    main()
