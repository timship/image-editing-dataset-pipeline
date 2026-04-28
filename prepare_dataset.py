import json
import random
from pathlib import Path

from diffusers import AutoPipelineForImage2Image
from datasets import load_dataset
import torch
from PIL import Image
from tqdm import tqdm


# =========================
# 1. ОСНОВНЫЕ НАСТРОЙКИ
# =========================

# Используем датасет beans с Hugging Face как источник исходных изображений.
DATASET_NAME = "beans"
SPLIT = "train"

# Количество изображений, которые будут выбраны из датасета и обработаны моделью.
LIMIT = 20

# Локальная image-to-image модель: получает исходное изображение и prompt, затем генерирует изменённую версию изображения.
MODEL_NAME = "stabilityai/sd-turbo"
# Папки проекта.
DATA_DIR = Path("data")
RAW_DIR = DATA_DIR / "raw"
EDITED_DIR = DATA_DIR / "edited"
METADATA_DIR = DATA_DIR / "metadata"

# Итоговый json-файл.
OUTPUT_JSON_PATH = METADATA_DIR / "dataset.json"


# =========================
# 2. ОПИСАНИЯ И ИНСТРУКЦИИ
# =========================

# Инструкции для редактирования.
# Модель будет менять изображение согласно этим промптам.
EDIT_INSTRUCTIONS = [
    "a realistic photo of a green plant leaf with water droplets",
    "a realistic botanical photo of a plant leaf in warm sunlight",
    "a close-up photo of a plant leaf on a clean white background",
    "a bright colorful photo of a green plant leaf in a garden",
    "a realistic plant leaf photo with enhanced colors and contrast",
]


# =========================
# 3. ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ
# =========================

def create_folders():
    """
    Создаём папки для исходных изображений,
    отредактированных изображений и metadata.
    """
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    EDITED_DIR.mkdir(parents=True, exist_ok=True)
    METADATA_DIR.mkdir(parents=True, exist_ok=True)


def prepare_image(image: Image.Image) -> Image.Image:
    """
    Приводим изображение к RGB и размеру 512x512.
    SD-Turbo лучше всего использовать на 512x512.
    """
    image = image.convert("RGB")
    image = image.resize((512, 512))
    return image


def save_json(records, path: Path):
    """
    Сохраняем список словарей в JSON-файл.
    ensure_ascii=False нужен, чтобы русский текст сохранялся нормально.
    """
    with open(path, "w", encoding="utf-8") as file:
        json.dump(records, file, ensure_ascii=False, indent=2)


def get_device():
    """
    Возвращает устройство для запуска модели.

    В проекте используется CPU, потому что на моей GTX 1650 Ti
    генерация в float16 на GPU давала чёрные изображения.
    """
    return "cpu"

def load_editing_model(device: str):
    """
    Загружаем локальную diffusion-модель для image-to-image генерации.

    SD-Turbo принимает исходное изображение и prompt,
    после чего генерирует изменённую версию изображения.
    """

    dtype = torch.float32

    pipe = AutoPipelineForImage2Image.from_pretrained(
        MODEL_NAME,
        torch_dtype=dtype,
        variant="fp16" if device == "cuda" else None,
        safety_checker=None,
        requires_safety_checker=False,
    )

    pipe = pipe.to(device)

    pipe.enable_attention_slicing()
    pipe.vae.enable_slicing()

    return pipe


def edit_image(pipe, image: Image.Image, instruction: str, device: str) -> Image.Image:
    """
    Редактируем изображение через локальную img2img-модель.

    strength — насколько сильно менять исходное изображение
    Для SD-Turbo лучше использовать небольшой guidance_scale и мало steps
    """

    generator = torch.Generator(device=device).manual_seed(42)

    result = pipe(
        prompt=instruction,
        image=image,
        strength=0.5,
        guidance_scale=0.0,
        num_inference_steps=4,
        generator=generator,
    )

    return result.images[0]


# =========================
# 4. ОСНОВНОЙ PIPELINE
# =========================

def main():
    create_folders()

    device = get_device()
    print(f"Using device: {device}")

    print("Loading dataset...")
    dataset = load_dataset(DATASET_NAME, split=SPLIT)

    # Перемешиваем индексы, чтобы взять случайные изображения
    all_indices = list(range(len(dataset)))
    random.seed(42)
    random.shuffle(all_indices)
    selected_indices = all_indices[:LIMIT]

    print("Loading image editing model...")
    pipe = load_editing_model(device)

    records = []

    print("Processing images...")

    for number, dataset_index in enumerate(tqdm(selected_indices), start=1):
        item = dataset[dataset_index]

        # Берём изображение из датасета
        source_image = prepare_image(item["image"])

        # Формируем id
        source_image_id = f"source_{number:04d}"
        final_image_id = f"edited_{number:04d}"

        # Пути сохранения
        source_image_path = RAW_DIR / f"{source_image_id}.jpg"
        final_image_path = EDITED_DIR / f"{final_image_id}.jpg"

        # Выбираем промпт для редактирования
        edit_instruction = random.choice(EDIT_INSTRUCTIONS)

        # Сохраняем исходное изображение
        source_image.save(source_image_path)

        # Получаем отредактированное изображение
        edited_image = edit_image(
            pipe=pipe,
            image=source_image,
            instruction=edit_instruction,
            device=device,
        )

        # Сохраняем финальное изображение
        edited_image.save(final_image_path)

        # Сохраняем metadata по 4 этапам
        records.append(
            {
                "source_image_id": source_image_id,
                "edit_prompt": edit_instruction,
                "edited_image_id": final_image_id,
            }
        )

    save_json(records, OUTPUT_JSON_PATH)

    print("Done!")
    print(f"Saved source images to: {RAW_DIR}")
    print(f"Saved edited images to: {EDITED_DIR}")
    print(f"Saved metadata to: {OUTPUT_JSON_PATH}")


if __name__ == "__main__":
    main()