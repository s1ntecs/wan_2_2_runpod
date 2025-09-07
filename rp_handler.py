import os
import base64
import traceback
import uuid
import torch
import tempfile
import requests
import numpy as np

from diffusers import WanImageToVideoPipeline
from diffusers.utils import export_to_video, load_image

import runpod
from runpod.serverless.utils.rp_download import file
from runpod.serverless.modules.rp_logger import RunPodLogger

from styles import STYLE_URLS, STYLE_NAMES  # ваши словари


# -------------------------------------------------------------
#  Схема входных данных для валидации
# -------------------------------------------------------------

device = "cuda"
model_id = "Wan-AI/Wan2.2-I2V-A14B-Diffusers"
DTYPE = torch.bfloat16
MAX_AREA = 480 * 832

# Будем хранить текущий активный стиль и путь
CURRENT_LORA_NAME = "./loras/wan_SmNmRmC.safetensors"


def calculate_frames(duration, frame_rate):
    raw_frames = round(duration * frame_rate)
    nearest_multiple_of_4 = round(raw_frames / 4) * 4
    return min(nearest_multiple_of_4 + 1, 81)


class Predictor():
    def setup(self):
        """ 
        Загружаем CLIPVisionModel, VAE и сам WanImageToVideoPipeline. 
        Вызывается один раз перед первым predict.
        """
        try:
            self.pipe = WanImageToVideoPipeline.from_pretrained(
                model_id, torch_dtype=DTYPE)

            self.pipe.enable_model_cpu_offload()
        except Exception as e:
            print("Error loading pipeline:", str(e))
            raise RuntimeError(f"Failed to load pipeline: {str(e)}")

    def _get_local_lora_path(self, lora_style: str) -> str:
        """
        Сформировать локальный путь к файлу LoRA по ключу стиля:
          - ищем в STYLE_NAMES имя файла,
          - проверяем, лежит ли он в ./loras/,
          - если нет — возвращаем None.
        """
        if not lora_style:
            return None

        filename = STYLE_NAMES.get(lora_style)
        if filename is None:
            return None

        # считаем, что локальные лоры лежат в папке ./loras/
        local_path = os.path.join("./loras", filename)
        if os.path.isfile(local_path):
            return local_path
        return None

    def _download_lora_if_needed(self, lora_style: str) -> str:
        """
        Если у нас уже есть локально — вернём путь.
        Если нет и есть ссылка в STYLE_URLS — скачиваем в ./loras/
            и вернём путь.
        """
        # 1) Узнаём файл по ключу
        filename = STYLE_NAMES.get(lora_style)
        if filename is None:
            raise RuntimeError(f"Unknown LORA style: {lora_style}")

        target_dir = "./loras"
        os.makedirs(target_dir, exist_ok=True)
        local_path = os.path.join(target_dir, filename)

        # Если файл уже скачан — сразу возвращаем
        if os.path.isfile(local_path):
            return local_path

        # Иначе — скачиваем по URL
        url = STYLE_URLS.get(lora_style)
        if url is None:
            raise RuntimeError(f"No URL found for LORA style: {lora_style}")

        print(f"Downloading LoRA '{lora_style}' from {url} into {local_path}")
        # Если ссылка ведёт на HF «blob»-вид, нужно чуть подправить URL
        if "huggingface.co" in url and "/blob/" in url:
            url = url.replace("/blob/", "/resolve/")
        resp = requests.get(url)
        if resp.status_code != 200:
            raise RuntimeError(f"Failed to download LORA from {url}: "
                               f"HTTP {resp.status_code}")

        with open(local_path, "wb") as f:
            f.write(resp.content)

        print(f"Successfully saved LoRA to {local_path}")
        return local_path

    def load_lora(self, lora_style: str, lora_strength: float = 1.0):
        """
        Верхнеуровневая функция «установки» LoRA:
          - Проверяем, меняется ли стиль (global CURRENT_LORA_STYLE).
          - Если нет, то выходим (уже загружен нужный).
          - Если да, то вызываем pipe.unload_lora_weights(),
                скачиваем/загружаем новую.
        """
        global CURRENT_LORA_NAME

        # Если стиль не передан — ничего не делаем
        if not lora_style:
            return

        # Скачиваем (или берём локальный) нужный файл
        local_path = self._download_lora_if_needed(lora_style)

        # Если стиль не поменялся — нет смысла перезагрузки
        if CURRENT_LORA_NAME == local_path:
            return

        try:
            self.pipe.unload_lora_weights()
        except Exception:
            # возможно, раньше pipe был без LoRA, пропускаем
            pass
        # Устанавливаем через diffusers
        print(f"Loading LoRA weights from local_path = {local_path}"
              f"(style={lora_style}, strength={lora_strength})")
        self.pipe.load_lora_weights(local_path, multiplier=lora_strength)
        print("LoRA applied.")

        # Обновляем глобальное состояние
        CURRENT_LORA_NAME = local_path

    def predict(
        self,
        image: str,
        prompt: str,
        negative_prompt: str = "low quality, bad quality, blurry, pixelated",
        num_frames: int = 81,
        guidance_scale: float = 5.0,
        num_inference_steps: int = 28,
        seed: int = None,
        fps: int = 16
    ) -> str:
        """
        Запускаем генерацию видео и возвращаем Base64.
        """

        # 3) Собираем генератор
        if seed is not None:
            torch.manual_seed(seed)
            generator = torch.Generator(device=device).manual_seed(seed)
        else:
            seed = np.random.randint(0, 2**30)
            generator = torch.Generator(device=device).manual_seed(seed)

        # 4) Загружаем изображение
        try:
            input_image = load_image(str(image))
        except Exception as e:
            raise RuntimeError(f"Failed to load input image: {str(e)}")

        aspect_ratio = input_image.height / input_image.width
        mod_value = self.pipe.vae_scale_factor_spatial * \
            self.pipe.transformer.config.patch_size[1]

        height = round(np.sqrt(MAX_AREA * aspect_ratio)
                       ) // mod_value * mod_value
        width = round(np.sqrt(MAX_AREA / aspect_ratio)
                      ) // mod_value * mod_value

        # Гарантируем, что делится на 16
        if height % 16 != 0 or width % 16 != 0:
            height = (height // 16) * 16
            width = (width // 16) * 16

        input_image = input_image.resize((width, height))

        # 6) Генерируем кадры
        output = self.pipe(
            image=input_image,
            prompt=prompt,
            negative_prompt=negative_prompt,
            height=height,
            width=width,
            num_frames=num_frames,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
            generator=generator
        ).frames[0]

        # 7) Сохраняем в MP4 и кодируем в Base64
        local_video_path = tempfile.mkdtemp() + "/" + str(uuid.uuid4()) + ".mp4"  # noqa
        export_to_video(output, str(local_video_path), fps=fps)

        with open(local_video_path, "rb") as f:
            video_bytes = f.read()
        video_b64 = base64.b64encode(video_bytes).decode("utf-8")

        os.remove(local_video_path)
        return video_b64


# -------------------------------------------------------------
#  RunPod Handler
# -------------------------------------------------------------
logger = RunPodLogger()

if torch.cuda.is_available():
    print("Current CUDA device:", torch.cuda.current_device())
    print("Device name:", torch.cuda.get_device_name(
        torch.cuda.current_device()))
    predictor = Predictor()
    predictor.setup()


def handler(job):
    global predictor
    try:
        payload = job.get("input", {})
        if predictor is None:
            predictor = Predictor()
            predictor.setup()

        # Скачиваем входное изображение
        image_url = payload["image_url"]
        image_obj = file(image_url)
        image_path = image_obj["file_path"]

        prompt = payload["prompt"]
        negative_prompt = payload.get("negative_prompt", "")
        num_frames = payload.get("num_frames", 81)
        fps = payload.get("fps", 16)
        guidance_scale = payload.get("guidance_scale", 5.0)
        num_inference_steps = payload.get("num_inference_steps", 28)
        seed = payload.get("seed", None)

        video_b64 = predictor.predict(
            image=image_path,
            prompt=prompt,
            negative_prompt=negative_prompt,
            num_frames=num_frames,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
            seed=seed,
            fps=fps
        )

        return {"video_base64": video_b64}

    except Exception as e:
        logger.error(f"An exception was raised: {e}")
        return {
            "error": str(e),
            "output": traceback.format_exc(),
            "refresh_worker": True
        }


if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})
