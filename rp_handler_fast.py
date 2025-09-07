###############################################################################
# RunPod handler for Wan2.1-I2V-14B + TeaCache                             #
#                                                                           #
# • Поддержка пресетов качества: fast / normal / best                       #
# • Без LoRA и Diffusers — чистый Wan SDK                                   #
# • TeaCache патч берётся из официального TeaCache4Wan2.1                   #
# • Возвращает base64-MP4                                                   #
###############################################################################

from __future__ import annotations

import os, uuid, base64, tempfile, traceback, math, random, gc
from contextlib import contextmanager
from typing import Literal

import torch
import numpy as np
from PIL import Image
import torchvision.transforms.functional as TF
import torch.cuda.amp as amp
from tqdm import tqdm

import wan
from wan.src.wan.configs import WAN_CONFIGS, MAX_AREA_CONFIGS
from wan.src.wan.utils.utils import cache_video
from wan.src.wan.utils.fm_solvers import FlowDPMSolverMultistepScheduler, get_sampling_sigmas, retrieve_timesteps
from wan.src.wan.utils.fm_solvers_unipc import FlowUniPCMultistepScheduler
from wan.src.wan.modules.model import sinusoidal_embedding_1d

import runpod
from runpod.serverless.utils.rp_download import file
from runpod.serverless.modules.rp_logger import RunPodLogger

# --------------------------------------------------------------------------------
# TeaCache patched forward (из TeaCache4Wan2.1, слегка упрощён)
# --------------------------------------------------------------------------------

def teacache_forward(self, x, t, context, seq_len, clip_fea=None, y=None):
    """Заменяет исходный forward DiT, добавляя пропуски слоёв по TeaCache."""
    device = self.patch_embedding.weight.device
    if self.freqs.device != device:
        self.freqs = self.freqs.to(device)

    # объединяем условный фрейм (y) с текущими скрытыми x
    if y is not None:
        x = [torch.cat([u, v], dim=0) for u, v in zip(x, y)]

    # Patch-embed
    x = [self.patch_embedding(u.unsqueeze(0)) for u in x]
    grid_sizes = torch.stack([torch.tensor(u.shape[2:], dtype=torch.long) for u in x])
    x = [u.flatten(2).transpose(1, 2) for u in x]
    seq_lens = torch.tensor([u.size(1) for u in x], dtype=torch.long)
    x = torch.cat([
        torch.cat([u, u.new_zeros(1, seq_len - u.size(1), u.size(2))], dim=1)
        for u in x
    ])

    # Временные эмбеддинги
    with amp.autocast(dtype=torch.float32):
        e  = self.time_embedding(sinusoidal_embedding_1d(self.freq_dim, t).float())
        e0 = self.time_projection(e).unflatten(1, (6, self.dim))

    # Текстовые эмбеддинги
    context = self.text_embedding(torch.stack([
        torch.cat([u, u.new_zeros(self.text_len - u.size(0), u.size(1))]) for u in context
    ]))

    # Добавляем CLIP-фичи для I2V
    if clip_fea is not None:
        context = torch.cat([self.img_emb(clip_fea), context], dim=1)

    kwargs = dict(
        e=e0,
        seq_lens=seq_lens,
        grid_sizes=grid_sizes,
        freqs=self.freqs,
        context=context,
        context_lens=None,
    )

    # --- TeaCache: решаем, вычислять блок или пропускать --------------------
    mod_inp = e0 if self.use_ret_steps else e  # что сравниваем между шагами
    even = (self.cnt % 2 == 0)
    if even:
        acc_attr, prev_attr, res_attr = 'acc_even', 'prev_e_even', 'res_even'
    else:
        acc_attr, prev_attr, res_attr = 'acc_odd', 'prev_e_odd', 'res_odd'

    accumulated = getattr(self, acc_attr)
    prev_e = getattr(self, prev_attr)

    if self.cnt < self.ret_steps or self.cnt >= self.cutoff_steps:
        compute = True
        accumulated = 0.0
    else:
        poly = np.poly1d(self.coefficients)
        accumulated += poly(((mod_inp - prev_e).abs().mean() / prev_e.abs().mean()).cpu().item())
        compute = accumulated >= self.teacache_thresh
        if compute:
            accumulated = 0.0
    setattr(self, acc_attr, accumulated)
    setattr(self, prev_attr, mod_inp.clone())

    if compute:
        ori = x.clone()
        for block in self.blocks:
            x = block(x, **kwargs)
        setattr(self, res_attr, x - ori)
    else:
        x += getattr(self, res_attr)

    # --- Head & unpatchify ---------------------------------------------------
    x = self.head(x, e)
    x = self.unpatchify(x, grid_sizes)

    self.cnt = (self.cnt + 1) % self.num_steps
    return [u.float() for u in x]

# --------------------------------------------------------------------------------
# Custom generate for I2V – берём из примера Alibaba, только без ненужных опций
# --------------------------------------------------------------------------------

def i2v_generate(
    self,                # WanI2V instance
    prompt: str,
    img: Image.Image,
    *,
    max_area: int,
    frame_num: int,
    shift: float,
    sample_solver: Literal['unipc', 'dpm++'],
    sampling_steps: int,
    guide_scale: float,
    n_prompt: str,
    seed: int,
    offload_model: bool = True,
):
    img_t = TF.to_tensor(img).sub_(0.5).div_(0.5).to(self.device)
    F = frame_num
    h, w = img_t.shape[1:]
    ar = h / w
    lat_h = round(math.sqrt(max_area * ar) // self.vae_stride[1] // self.patch_size[1] * self.patch_size[1])
    lat_w = round(math.sqrt(max_area / ar) // self.vae_stride[2] // self.patch_size[2] * self.patch_size[2])
    h, w = lat_h * self.vae_stride[1], lat_w * self.vae_stride[2]

    max_seq_len = int(math.ceil((((F - 1) // self.vae_stride[0] + 1) * lat_h * lat_w / (self.patch_size[1] * self.patch_size[2])) / self.sp_size)) * self.sp_size

    seed = seed if seed >= 0 else random.randint(0, 2 ** 31 - 1)
    g = torch.Generator(device=self.device).manual_seed(seed)

    noise = torch.randn(
        self.vae.model.z_dim,
        (F - 1) // self.vae_stride[0] + 1,
        lat_h,
        lat_w,
        dtype=torch.float32,
        device=self.device,
        generator=g,
    )

    msk = torch.ones(1, F, lat_h, lat_w, device=self.device)
    msk[:, 1:] = 0
    msk = torch.cat([torch.repeat_interleave(msk[:, :1], repeats=4, dim=1), msk[:, 1:]], dim=1)
    msk = msk.view(1, msk.shape[1] // 4, 4, lat_h, lat_w).transpose(1, 2)[0]

    if n_prompt == "":
        n_prompt = self.sample_neg_prompt

    # --- текстовые и CLIP-контексты -----------------------------------------
    if not self.t5_cpu:
        self.text_encoder.model.to(self.device)
        ctx = self.text_encoder([prompt], self.device)
        ctx_null = self.text_encoder([n_prompt], self.device)
        if offload_model:
            self.text_encoder.model.cpu()
    else:
        ctx = self.text_encoder([prompt], torch.device('cpu'))
        ctx_null = self.text_encoder([n_prompt], torch.device('cpu'))
        ctx, ctx_null = [t.to(self.device) for t in ctx], [t.to(self.device) for t in ctx_null]

    self.clip.model.to(self.device)
    clip_ctx = self.clip.visual([img_t[:, None, :, :]])
    if offload_model:
        self.clip.model.cpu()

    y = self.vae.encode([
        torch.cat([
            TF.resize(img, (h, w)).transpose(0, 1),
            torch.zeros(3, F - 1, h, w),
        ], dim=1).to(self.device)
    ])[0]
    y = torch.cat([msk, y])

    @contextmanager
    def no_sync():
        yield

    with amp.autocast(dtype=self.param_dtype), torch.no_grad(), no_sync():
        if sample_solver == 'unipc':
            sched = FlowUniPCMultistepScheduler(self.num_train_timesteps, shift=1, use_dynamic_shifting=False)
            sched.set_timesteps(sampling_steps, device=self.device, shift=shift)
            timesteps = sched.timesteps
        else:
            sched = FlowDPMSolverMultistepScheduler(self.num_train_timesteps, shift=1, use_dynamic_shifting=False)
            sigmas = get_sampling_sigmas(sampling_steps, shift)
            timesteps, _ = retrieve_timesteps(sched, device=self.device, sigmas=sigmas)

        latent = noise
        arg_c   = {'context': [ctx[0]], 'clip_fea': clip_ctx, 'seq_len': max_seq_len, 'y': [y]}
        arg_nil = {'context': ctx_null,  'clip_fea': clip_ctx, 'seq_len': max_seq_len, 'y': [y]}

        if offload_model:
            torch.cuda.empty_cache()
        self.model.to(self.device)
        for t in tqdm(timesteps, desc="sampling"):
            li = [latent.to(self.device)]
            ts = torch.stack([t]).to(self.device)
            cond = self.model(li, t=ts, **arg_c)[0].to(torch.device('cpu') if offload_model else self.device)
            if offload_model: torch.cuda.empty_cache()
            uncond = self.model(li, t=ts, **arg_nil)[0].to(torch.device('cpu') if offload_model else self.device)
            if offload_model: torch.cuda.empty_cache()
            noise_pred = uncond + guide_scale * (cond - uncond)
            latent = sched.step(noise_pred.unsqueeze(0), t, latent.unsqueeze(0), return_dict=False, generator=g)[0].squeeze(0)

        if offload_model:
            self.model.cpu(); torch.cuda.empty_cache()
        video = self.vae.decode([latent.to(self.device)])[0]

    if offload_model:
        gc.collect(); torch.cuda.synchronize()
    return video

# --------------------------------------------------------------------------------
# Presets quality → TeaCache params
# --------------------------------------------------------------------------------
QUALITY_PRESETS = {
    "fast":   {"steps": 28, "thresh": 0.30, "use_ret": False},
    "normal": {"steps": 40, "thresh": 0.15, "use_ret": False},
    "best":   {"steps": 56, "thresh": 0.10, "use_ret": True},
}

# --------------------------------------------------------------------------------
# Predictor wrapper
# --------------------------------------------------------------------------------
DEVICE_ID = int(os.getenv("LOCAL_RANK", 0))
CKPT_DIR = os.getenv("WAN_CKPT_DIR", "./models/Wan-AI/Wan2.1-I2V-14B-480P")

logger = RunPodLogger()

class Predictor:
    def __init__(self):
        cfg = WAN_CONFIGS["i2v-14B"]
        self.pipe = wan.WanI2V(config=cfg, checkpoint_dir=CKPT_DIR, device_id=DEVICE_ID, rank=0)
        # подвязываем кастомные функции один раз
        self.pipe.__class__.generate = i2v_generate
        self._apply_preset("normal")
        logger.info("Wan2.1-I2V + TeaCache инициализирован.")

    # ------------------ TeaCache preset patch --------------------------------
    def _apply_preset(self, preset: str):
        p = QUALITY_PRESETS[preset]
        m = self.pipe.model.__class__  # type: ignore[attr-defined]
        m.enable_teacache = True
        m.forward = teacache_forward
        m.cnt = 0
        m.num_steps = p["steps"] * 2  # каждая итерация модальн.+немодальн.
        m.teacache_thresh = p["thresh"]
        m.use_ref_steps = p["use_ret"]
        # служебные ресеты
        m.acc_even = m.acc_odd = 0.0
        m.prev_e_even = m.prev_e_odd = None
        m.res_even = m.res_odd = None
        if p["use_ret"]:
            m.coefficients = [2.57151496e+05, -3.54229917e+04, 1.40286849e+03, -1.35890334e+01, 1.32517977e-01]
            m.ret_steps = 10
            m.cutoff_steps = m.num_steps
        else:
            m.coefficients = [-3.02331670e+02, 2.23948934e+02, -5.25463970e+01, 5.87348440e+00, -2.01973289e-01]
            m.ret_steps = 2
            m.cutoff_steps = m.num_steps - 2

    # ------------------ inference -------------------------------------------
    def __call__(
        self,
        *,
        image_path: str,
        prompt: str,
        negative_prompt: str,
        duration: float,
        fps: int,
        guidance: float,
        preset: str,
        seed: int | None,
    ) -> str:
        self._apply_preset(preset)
        img = Image.open(image_path).convert("RGB")
        base_seed = seed if seed is not None else random.randrange(2 ** 31)
        frames = max(5, round(duration * fps / 4) * 4 + 1)  # 4n+1

        video = self.pipe.generate(
            prompt,
            img,
            max_area=MAX_AREA_CONFIGS["480*832"],
            frame_num=frames,
            shift=3.0,
            sample_solver='unipc',
            sampling_steps=QUALITY_PRESETS[preset]["steps"],
            guide_scale=guidance,
            n_prompt=negative_prompt,
            seed=base_seed,
            offload_model=True,
        )

        tmp = tempfile.mkdtemp()
        out_path = os.path.join(tmp, f"{uuid.uuid4()}.mp4")
        cache_video(video[None], save_file=out_path, fps=fps, nrow=1, normalize=True, value_range=(-1, 1))
        with open(out_path, "rb") as f:
            b64 = base64.b64encode(f.read()).decode()
        return b64

# --------------------------------------------------------------------------------
# RunPod handler
# --------------------------------------------------------------------------------

predictor: Predictor | None = None


def handler(job):
    global predictor
    try:
        if predictor is None:
            predictor = Predictor()

        payload = job.get("input", {})
        img_file = file(payload["image_url"])

        video_b64 = predictor(
            image_path=img_file["file_path"],
            prompt=payload["prompt"],
            negative_prompt=payload.get("negative_prompt", "low quality, bad quality"),
            duration=float(payload.get("duration", 3.0)),
            fps=int(payload.get("fps", 16)),
            guidance=float(payload.get("guidance_scale", 5.0)),
            preset=payload.get("quality_preset", "normal"),
            seed=payload.get("seed"),
        )

        return {"video_base64": video_b64}

    except Exception as e:
        logger.error(f"Generation error: {e}")
        return {"error": str(e), "output": traceback.format_exc(), "refresh_worker": True}


if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})
