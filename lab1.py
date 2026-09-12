"""
Лабораторна робота № 1. Первинна обробка та представлення мультимедійних даних.

Скрипт послідовно виконує всі пункти завдання:
    1. Завантаження зображення та збереження його у форматах JPG, PNG, BMP.
    2. Дослідження колірних просторів RGB, Grayscale, HSV, YCbCr.
    3. Геометричні перетворення: масштабування, поворот, обрізання.
    4. Реалізація метрик якості MSE, RMSE, PSNR, SSIM (власноруч і через scikit-image).
    5. Дослідницький експеримент із JPEG-компресією: Q = {10, 20, 40, 60, 80, 95},
       побудова залежностей PSNR = f(Q) та SSIM = f(Q).

Усі проміжні зображення, таблиці (CSV) та графіки (PNG) зберігаються у теку
"results/lab1", щоб їх потім можна було вставити у звіт.
"""

import os
import csv

import cv2
import numpy as np
import matplotlib

# Не показуємо вікна графіків, а одразу зберігаємо їх у файли (зручно для запуску без GUI).
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from skimage import data
from skimage.metrics import structural_similarity as sk_ssim
from skimage.metrics import mean_squared_error as sk_mse
from skimage.metrics import peak_signal_noise_ratio as sk_psnr

# ---------------------------------------------------------------------------
# Допоміжні шляхи. Усі результати лабораторної складаємо в окрему теку,
# щоб не змішувати їх із вихідним кодом.
# ---------------------------------------------------------------------------
IMAGES_DIR = "images"
RESULTS_DIR = os.path.join("results", "lab1")
os.makedirs(IMAGES_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)


def result_path(filename: str) -> str:
    """Формує повний шлях до файлу результату лабораторної № 1."""
    return os.path.join(RESULTS_DIR, filename)


# ---------------------------------------------------------------------------
# 0. Підготовка вихідного зображення.
#
# За умовою лабораторної кожен студент працює зі своїм зображенням згідно
# з варіантом. Якщо власне зображення не задане (не покладене у теку
# "images"), скрипт автоматично використовує тестове зображення
# astronaut() із бібліотеки scikit-image, щоб приклад можна було запустити
# "з коробки". Для реальної здачі лабораторної достатньо покласти своє фото
# у файл images/my_image.png (або .jpg) — скрипт підхопить саме його.
# ---------------------------------------------------------------------------
def load_source_image() -> np.ndarray:
    """Завантажує вихідне зображення у форматі BGR (як зберігає OpenCV)."""
    for name in ("my_image.png", "my_image.jpg", "my_image.jpeg", "my_image.bmp"):
        path = os.path.join(IMAGES_DIR, name)
        if os.path.exists(path):
            print(f"Використовую власне зображення: {path}")
            return cv2.imread(path, cv2.IMREAD_COLOR)

    print("Власне зображення не знайдено -> використовую вбудоване demo-зображення "
          "skimage.data.astronaut(). Покладіть своє фото у images/my_image.png, "
          "щоб лабораторна виконувалась саме на ньому.")
    rgb = data.astronaut()  # зображення у форматі RGB, 512x512x3, uint8
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)  # OpenCV очікує порядок каналів BGR
    return bgr


# ---------------------------------------------------------------------------
# Власні реалізації метрик якості (пункт 4 завдання). Спочатку рахуємо
# формули "руками" на numpy, а потім звіряємо зі scikit-image, щоб
# переконатися, що розбіжність — лише похибка округлення.
# ---------------------------------------------------------------------------
def mse_manual(img1: np.ndarray, img2: np.ndarray) -> float:
    """MSE = (1 / (M*N)) * sum((I - K)^2) -- середньоквадратична помилка."""
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    return float(np.mean((img1 - img2) ** 2))


def rmse_manual(img1: np.ndarray, img2: np.ndarray) -> float:
    """RMSE = sqrt(MSE) -- корінь із середньоквадратичної помилки."""
    return float(np.sqrt(mse_manual(img1, img2)))


def psnr_manual(img1: np.ndarray, img2: np.ndarray, max_value: float = 255.0) -> float:
    """PSNR = 10 * log10(MAX^2 / MSE) -- пікове відношення сигнал/шум, дБ."""
    mse_value = mse_manual(img1, img2)
    if mse_value == 0:
        return float("inf")  # зображення ідентичні -> шуму немає
    return float(10.0 * np.log10((max_value ** 2) / mse_value))


def ssim_manual_gray(img1: np.ndarray, img2: np.ndarray, win_size: int = 7) -> float:
    """
    Власноруч реалізований SSIM (Structural Similarity Index) для
    напівтонового зображення за класичною формулою Ванга та ін., з
    обчисленням статистик у ковзному вікні win_size x win_size (як у
    scikit-image за замовчуванням) -- тому результат має збігатися з
    eталонною реалізацією scikit-image з точністю до похибок округлення.

    SSIM(x, y) = ((2*mu_x*mu_y + C1) * (2*sigma_xy + C2)) /
                 ((mu_x^2 + mu_y^2 + C1) * (sigma_x^2 + sigma_y^2 + C2))

    Локальні статистики (mu, sigma) рахуються через box-filter (просте
    ковзне середнє) по вікну, так само, як у стандартній реалізації без
    гаусових ваг.
    """
    x = img1.astype(np.float64)
    y = img2.astype(np.float64)

    def local_mean(a: np.ndarray) -> np.ndarray:
        return cv2.boxFilter(a, ddepth=-1, ksize=(win_size, win_size),
                              borderType=cv2.BORDER_REPLICATE)

    mu_x = local_mean(x)
    mu_y = local_mean(y)
    mu_x2 = mu_x ** 2
    mu_y2 = mu_y ** 2
    mu_xy = mu_x * mu_y

    # Незміщена (sample) оцінка дисперсії/коваріації: N/(N-1) поправка,
    # як робить scikit-image (use_sample_covariance=True за замовчуванням).
    n_pixels = win_size ** 2
    cov_norm = n_pixels / (n_pixels - 1)

    sigma_x2 = cov_norm * (local_mean(x * x) - mu_x2)
    sigma_y2 = cov_norm * (local_mean(y * y) - mu_y2)
    sigma_xy = cov_norm * (local_mean(x * y) - mu_xy)

    l_range = 255.0
    c1 = (0.01 * l_range) ** 2
    c2 = (0.03 * l_range) ** 2

    numerator = (2 * mu_xy + c1) * (2 * sigma_xy + c2)
    denominator = (mu_x2 + mu_y2 + c1) * (sigma_x2 + sigma_y2 + c2)
    ssim_map = numerator / denominator

    # Як і scikit-image, відкидаємо рамку шириною win_size//2 (крайовий ефект).
    pad = win_size // 2
    if pad > 0:
        ssim_map = ssim_map[pad:-pad, pad:-pad]
    return float(ssim_map.mean())


def to_gray(img_bgr: np.ndarray) -> np.ndarray:
    """Швидкий переклад BGR -> Grayscale через OpenCV (для метрик потрібен 1 канал)."""
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)


# ---------------------------------------------------------------------------
# 1. Завантаження зображення та формати файлів (BMP / PNG / JPG).
# ---------------------------------------------------------------------------
def part1_formats(img_bgr: np.ndarray) -> None:
    print("\n=== 1. Завантаження та формати файлів ===")

    h, w, c = img_bgr.shape
    print(f"Розмір (H, W, каналів): ({h}, {w}, {c})")
    print(f"Тип даних: {img_bgr.dtype} -> 8 біт на канал")

    bmp_path = result_path("demo.bmp")
    png_path = result_path("demo.png")
    jpg_path = result_path("demo.jpg")

    # BMP -- зберігання без стиснення (кожен піксель записаний "як є").
    cv2.imwrite(bmp_path, img_bgr)
    # PNG -- стиснення без втрат (DEFLATE), розмір файлу менший за BMP,
    # але після розпакування пікселі будуть побітово ідентичні оригіналу.
    cv2.imwrite(png_path, img_bgr)
    # JPEG -- стиснення з втратами (дискретне косинусне перетворення +
    # квантування), розмір файлу значно менший ціною втрати частини деталей.
    cv2.imwrite(jpg_path, img_bgr, [cv2.IMWRITE_JPEG_QUALITY, 90])

    sizes_kb = {
        "BMP": os.path.getsize(bmp_path) / 1024,
        "PNG": os.path.getsize(png_path) / 1024,
        "JPG (Q=90)": os.path.getsize(jpg_path) / 1024,
    }
    for name, size in sizes_kb.items():
        print(f"{name}: {size:.1f} КБ")

    # Теоретичний розмір нестиснутого зображення за формулою S = M*N*c*b/8 байт,
    # де M, N -- розміри у пікселях, c -- кількість каналів, b -- біт на канал.
    theoretical_bytes = h * w * c * 8 / 8
    theoretical_kb = theoretical_bytes / 1024
    print(f"Теоретичний розмір (формула S = M*N*c*b/8): {theoretical_kb:.1f} КБ")
    print("-> саме такий обсяг займає BMP-файл, бо він не стискає дані.")

    with open(result_path("01_formats.csv"), "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Формат", "Розмір, КБ"])
        for name, size in sizes_kb.items():
            writer.writerow([name, f"{size:.1f}"])
        writer.writerow(["Теоретичний (без стиснення)", f"{theoretical_kb:.1f}"])


# ---------------------------------------------------------------------------
# 2. Колірні простори: RGB, Grayscale, HSV, YCbCr.
# ---------------------------------------------------------------------------
def part2_color_spaces(img_bgr: np.ndarray) -> None:
    print("\n=== 2. Колірні простори: RGB, Grayscale, HSV, YCbCr ===")

    b, g, r = cv2.split(img_bgr)

    # Grayscale за формулою з лекції (застосовується до каналів у порядку R, G, B):
    # Y = 0.299*R + 0.587*G + 0.114*B
    gray_manual = (0.299 * r.astype(np.float64)
                   + 0.587 * g.astype(np.float64)
                   + 0.114 * b.astype(np.float64))
    gray_manual = np.clip(gray_manual, 0, 255).astype(np.uint8)

    gray_cv2 = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    diff = np.abs(gray_manual.astype(np.int16) - gray_cv2.astype(np.int16))
    mean_diff = float(diff.mean())
    print(f"Середня різниця між ручною формулою і cv2.cvtColor: {mean_diff:.4f} "
          "-> лише похибка округлення при переведенні у uint8.")

    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    h_ch, s_ch, v_ch = cv2.split(hsv)

    ycbcr = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2YCrCb)  # у OpenCV порядок каналів Y, Cr, Cb
    y_ch, cr_ch, cb_ch = cv2.split(ycbcr)

    # Перевірка оборотності переходу RGB -> YCbCr -> RGB (з округленнями).
    bgr_back = cv2.cvtColor(ycbcr, cv2.COLOR_YCrCb2BGR)
    mse_roundtrip = mse_manual(img_bgr, bgr_back)
    print(f"MSE після циклу RGB -> YCbCr -> RGB: {mse_roundtrip:.4f}")

    fig, axes = plt.subplots(3, 3, figsize=(12, 12))
    channels = [
        (r, "Канал R"), (g, "Канал G"), (b, "Канал B"),
        (h_ch, "H (відтінок)"), (s_ch, "S (насиченість)"), (v_ch, "V (яскравість)"),
        (y_ch, "Y (яскравість)"), (cr_ch, "Cr"), (cb_ch, "Cb"),
    ]
    for ax, (channel, title) in zip(axes.ravel(), channels):
        ax.imshow(channel, cmap="gray")
        ax.set_title(title)
        ax.axis("off")
    fig.suptitle("Канали колірних просторів RGB, HSV, YCbCr")
    fig.tight_layout()
    fig.savefig(result_path("02_color_channels.png"), dpi=130)
    plt.close(fig)

    fig2, axes2 = plt.subplots(1, 3, figsize=(12, 4))
    axes2[0].imshow(gray_manual, cmap="gray")
    axes2[0].set_title("Grayscale: формула вручну")
    axes2[1].imshow(gray_cv2, cmap="gray")
    axes2[1].set_title("Grayscale: cv2.cvtColor")
    im = axes2[2].imshow(diff, cmap="hot")
    axes2[2].set_title(f"Різниця (макс={diff.max()})")
    for ax in axes2:
        ax.axis("off")
    fig2.colorbar(im, ax=axes2[2], fraction=0.046)
    fig2.tight_layout()
    fig2.savefig(result_path("02_grayscale_compare.png"), dpi=130)
    plt.close(fig2)

    with open(result_path("02_color_spaces.csv"), "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Показник", "Значення"])
        writer.writerow(["Середня різниця Grayscale (ручна формула vs cv2)", f"{mean_diff:.4f}"])
        writer.writerow(["MSE після RGB->YCbCr->RGB", f"{mse_roundtrip:.4f}"])


# ---------------------------------------------------------------------------
# 3. Геометричні перетворення: масштабування, поворот, обрізання.
# ---------------------------------------------------------------------------
def part3_geometry(img_bgr: np.ndarray) -> dict:
    print("\n=== 3. Геометричні перетворення ===")

    h, w = img_bgr.shape[:2]

    # Поворот на 30 градусів навколо центра зображення без обрізання країв.
    center = (w / 2, h / 2)
    rotation_matrix = cv2.getRotationMatrix2D(center, angle=30, scale=1.0)
    rotated = cv2.warpAffine(img_bgr, rotation_matrix, (w, h))

    # Обрізання (crop) центральної частини зображення розміром 60% x 60%.
    crop_w, crop_h = int(w * 0.6), int(h * 0.6)
    x0 = (w - crop_w) // 2
    y0 = (h - crop_h) // 2
    cropped = img_bgr[y0:y0 + crop_h, x0:x0 + crop_w]

    # Масштабування (scaling) -- окрема операція, не плутати з циклом
    # "зменшити -> збільшити" нижче, який лише порівнює методи інтерполяції.
    # Тут просто зменшуємо зображення вдвічі та збільшуємо у 1.5 раза
    # відносно оригіналу (білінійна інтерполяція).
    scaled_down = cv2.resize(img_bgr, (w // 2, h // 2), interpolation=cv2.INTER_LINEAR)
    scaled_up = cv2.resize(img_bgr, (int(w * 1.5), int(h * 1.5)), interpolation=cv2.INTER_LINEAR)

    cv2.imwrite(result_path("03_rotated_30deg.png"), rotated)
    cv2.imwrite(result_path("03_cropped_center.png"), cropped)
    cv2.imwrite(result_path("03_scaled_down_0.5x.png"), scaled_down)
    cv2.imwrite(result_path("03_scaled_up_1.5x.png"), scaled_up)
    print(f"Масштабування: оригінал {w}x{h} -> зменшено до {w // 2}x{h // 2} "
          f"-> збільшено до {int(w * 1.5)}x{int(h * 1.5)}")

    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    for ax, img, title in zip(
        axes,
        [img_bgr, scaled_down, rotated, cropped],
        ["Оригінал", "Масштабування 0.5x", "Поворот на 30°", "Обрізання (центр)"],
    ):
        ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        ax.set_title(title)
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(result_path("03_geometry.png"), dpi=130)
    plt.close(fig)

    # Порівняння методів інтерполяції при циклі "зменшити -> збільшити":
    # чим менше втрачається інформації, тим вищий PSNR після повернення
    # до початкового розміру.
    interpolations = {
        "NEAREST": cv2.INTER_NEAREST,
        "LINEAR": cv2.INTER_LINEAR,
        "CUBIC": cv2.INTER_CUBIC,
        "LANCZOS4": cv2.INTER_LANCZOS4,
    }
    small_size = (w // 4, h // 4)

    psnr_by_interp = {}
    resized_images = {}
    for name, flag in interpolations.items():
        small = cv2.resize(img_bgr, small_size, interpolation=flag)
        back = cv2.resize(small, (w, h), interpolation=flag)
        psnr_value = psnr_manual(img_bgr, back)
        psnr_by_interp[name] = psnr_value
        resized_images[name] = back
        print(f"{name}: PSNR після зменшення/збільшення = {psnr_value:.2f} дБ")

    fig2, axes2 = plt.subplots(1, len(interpolations), figsize=(4 * len(interpolations), 4))
    for ax, (name, img) in zip(axes2, resized_images.items()):
        ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        ax.set_title(f"{name}\nPSNR={psnr_by_interp[name]:.2f} дБ")
        ax.axis("off")
    fig2.tight_layout()
    fig2.savefig(result_path("03_interpolation_compare.png"), dpi=130)
    plt.close(fig2)

    with open(result_path("03_interpolation_psnr.csv"), "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Метод інтерполяції", "PSNR після зменшення/збільшення, дБ"])
        for name, value in psnr_by_interp.items():
            writer.writerow([name, f"{value:.2f}"])

    return psnr_by_interp


# ---------------------------------------------------------------------------
# 4. Метрики якості: MSE, RMSE, PSNR, SSIM -- власна реалізація vs scikit-image.
# ---------------------------------------------------------------------------
def part4_metrics(img_bgr: np.ndarray) -> None:
    print("\n=== 4. Метрики якості MSE / RMSE / PSNR / SSIM ===")

    gray = to_gray(img_bgr)

    # Створюємо спотворену версію зображення (додаємо гаусів шум), щоб було
    # що порівнювати з оригіналом -- на ідентичних зображеннях всі метрики
    # тривіальні (MSE=0, PSNR=inf, SSIM=1).
    rng = np.random.default_rng(seed=42)
    noise = rng.normal(0, 15, gray.shape)
    noisy_gray = np.clip(gray.astype(np.float64) + noise, 0, 255).astype(np.uint8)

    mse_own = mse_manual(gray, noisy_gray)
    rmse_own = rmse_manual(gray, noisy_gray)
    psnr_own = psnr_manual(gray, noisy_gray)
    ssim_own = ssim_manual_gray(gray, noisy_gray)

    mse_sk = sk_mse(gray, noisy_gray)
    psnr_sk = sk_psnr(gray, noisy_gray, data_range=255)
    rmse_sk = float(np.sqrt(mse_sk))
    ssim_sk, ssim_map = sk_ssim(gray, noisy_gray, data_range=255, full=True)

    print(f"Власна реалізація | MSE={mse_own:.2f} RMSE={rmse_own:.2f} "
          f"PSNR={psnr_own:.2f} дБ SSIM={ssim_own:.4f}")
    print(f"scikit-image      | MSE={mse_sk:.2f} RMSE={rmse_sk:.2f} "
          f"PSNR={psnr_sk:.2f} дБ SSIM={ssim_sk:.4f}")

    fig, axes = plt.subplots(1, 2, figsize=(9, 4.5))
    axes[0].imshow(noisy_gray, cmap="gray")
    axes[0].set_title("Зображення з шумом")
    im = axes[1].imshow(ssim_map, cmap="viridis")
    axes[1].set_title("Карта SSIM (локально), scikit-image")
    for ax in axes:
        ax.axis("off")
    fig.colorbar(im, ax=axes[1], fraction=0.046)
    fig.tight_layout()
    fig.savefig(result_path("04_metrics.png"), dpi=130)
    plt.close(fig)

    with open(result_path("04_metrics.csv"), "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Метрика", "Власна реалізація", "scikit-image"])
        writer.writerow(["MSE", f"{mse_own:.2f}", f"{mse_sk:.2f}"])
        writer.writerow(["RMSE", f"{rmse_own:.2f}", f"{rmse_sk:.2f}"])
        writer.writerow(["PSNR, дБ", f"{psnr_own:.2f}", f"{psnr_sk:.2f}"])
        writer.writerow(["SSIM", f"{ssim_own:.4f}", f"{ssim_sk:.4f}"])


# ---------------------------------------------------------------------------
# 5. Дослідницький експеримент: JPEG-компресія Q = {10,20,40,60,80,95}.
#    Будуємо залежності PSNR = f(Q) та SSIM = f(Q).
# ---------------------------------------------------------------------------
def part5_jpeg_experiment(img_bgr: np.ndarray) -> list:
    print("\n=== 5. Дослідницький експеримент: JPEG-компресія ===")

    quality_levels = [10, 20, 40, 60, 80, 95]
    rows = []

    for q in quality_levels:
        jpg_path = result_path(f"05_jpeg_q{q}.jpg")
        cv2.imwrite(jpg_path, img_bgr, [cv2.IMWRITE_JPEG_QUALITY, q])
        size_kb = os.path.getsize(jpg_path) / 1024

        decoded = cv2.imread(jpg_path, cv2.IMREAD_COLOR)
        psnr_value = sk_psnr(img_bgr, decoded, data_range=255)
        ssim_value = sk_ssim(img_bgr, decoded, data_range=255, channel_axis=2)

        rows.append({"Q": q, "size_kb": size_kb, "psnr": psnr_value, "ssim": ssim_value})
        print(f"Q={q:3d} | розмір={size_kb:6.1f} КБ | PSNR={psnr_value:6.2f} дБ | "
              f"SSIM={ssim_value:.4f}")

    with open(result_path("05_jpeg_experiment.csv"), "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Q", "Розмір, КБ", "PSNR, дБ", "SSIM"])
        for row in rows:
            writer.writerow([row["Q"], f"{row['size_kb']:.1f}",
                              f"{row['psnr']:.2f}", f"{row['ssim']:.4f}"])

    qs = [row["Q"] for row in rows]
    psnrs = [row["psnr"] for row in rows]
    ssims = [row["ssim"] for row in rows]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    axes[0].plot(qs, psnrs, marker="o")
    axes[0].set_title("PSNR = f(Q)")
    axes[0].set_xlabel("Q (якість JPEG)")
    axes[0].set_ylabel("PSNR, дБ")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(qs, ssims, marker="o", color="orange")
    axes[1].set_title("SSIM = f(Q)")
    axes[1].set_xlabel("Q (якість JPEG)")
    axes[1].set_ylabel("SSIM")
    axes[1].grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(result_path("05_jpeg_psnr_ssim.png"), dpi=130)
    plt.close(fig)

    # Наочна демонстрація артефактів компресії на прикладі найнижчої якості.
    q_low = quality_levels[0]
    low_q_img = cv2.imread(result_path(f"05_jpeg_q{q_low}.jpg"), cv2.IMREAD_COLOR)
    diff = cv2.absdiff(img_bgr, low_q_img)
    diff_amplified = np.clip(diff.astype(np.int32) * 5, 0, 255).astype(np.uint8)

    fig2, axes2 = plt.subplots(1, 2, figsize=(9, 4.5))
    axes2[0].imshow(cv2.cvtColor(low_q_img, cv2.COLOR_BGR2RGB))
    axes2[0].set_title(f"JPEG, Q={q_low}")
    axes2[1].imshow(cv2.cvtColor(diff_amplified, cv2.COLOR_BGR2RGB))
    axes2[1].set_title("Різниця (оригінал - стиснуте), x5")
    for ax in axes2:
        ax.axis("off")
    fig2.tight_layout()
    fig2.savefig(result_path("05_jpeg_artifacts.png"), dpi=130)
    plt.close(fig2)

    # Додаткове дослідницьке питання: як впливає роздільна здатність на якість
    # при однаковому рівні компресії Q=40. Зменшуємо зображення в 1, 2 і 4 рази
    # і дивимось на PSNR відносно зображення такого ж (зменшеного) розміру.
    resolution_rows = []
    h, w = img_bgr.shape[:2]
    for scale in [1, 2, 4]:
        resized = cv2.resize(img_bgr, (w // scale, h // scale), interpolation=cv2.INTER_AREA)
        path = result_path(f"05_resolution_scale{scale}_q40.jpg")
        cv2.imwrite(path, resized, [cv2.IMWRITE_JPEG_QUALITY, 40])
        decoded = cv2.imread(path, cv2.IMREAD_COLOR)
        psnr_value = sk_psnr(resized, decoded, data_range=255)
        resolution_rows.append({"scale": scale, "psnr": psnr_value})
        print(f"Масштаб 1/{scale}, Q=40 -> PSNR={psnr_value:.2f} дБ")

    with open(result_path("05_resolution_vs_quality.csv"), "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Масштаб зменшення", "PSNR при Q=40, дБ"])
        for row in resolution_rows:
            writer.writerow([f"1/{row['scale']}", f"{row['psnr']:.2f}"])

    return rows


def main() -> None:
    img_bgr = load_source_image()

    part1_formats(img_bgr)
    part2_color_spaces(img_bgr)
    part3_geometry(img_bgr)
    part4_metrics(img_bgr)
    part5_jpeg_experiment(img_bgr)

    print("\nЛабораторна робота № 1 виконана. Усі результати у теці:", RESULTS_DIR)


if __name__ == "__main__":
    main()
