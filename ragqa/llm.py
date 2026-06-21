"""Local Qwen model loading and response parsing."""

from __future__ import annotations

import json
import os
import platform
import re
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Callable


class LocalQwen3:
    """Small adapter around a locally loaded, 4-bit Qwen3-8B model."""

    TRANSFORMERS_MODEL = "Qwen/Qwen3-8B"
    MLX_MODEL = "Qwen/Qwen3-8B-MLX-4bit"

    def __init__(
        self,
        model_id: str | None = None,
        progress_callback: Callable[[float, str], None] | None = None,
    ):
        self.backend = (
            "mlx"
            if platform.system() == "Darwin" and platform.machine() == "arm64"
            else "transformers"
        )
        selected_model = model_id or os.getenv(
            "QWEN_MODEL_ID",
            self.MLX_MODEL if self.backend == "mlx" else self.TRANSFORMERS_MODEL,
        )
        self.model_id = selected_model
        model_path = self._ensure_downloaded(selected_model, progress_callback)

        if self.backend == "mlx":
            from mlx_lm import load

            self.model, self.tokenizer = load(model_path)
        else:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

            quantization = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
            )
            self.tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path,
                device_map="auto",
                quantization_config=quantization,
                local_files_only=True,
            )

    @staticmethod
    def _ensure_downloaded(
        model_id: str,
        progress_callback: Callable[[float, str], None] | None,
    ) -> str:
        if Path(model_id).expanduser().exists():
            return str(Path(model_id).expanduser().resolve())

        from huggingface_hub import HfApi, snapshot_download
        from huggingface_hub.constants import HF_HUB_CACHE
        from huggingface_hub.file_download import repo_folder_name

        storage = Path(HF_HUB_CACHE) / repo_folder_name(repo_id=model_id, repo_type="model")
        marker = storage / ".ragqa_download_complete"
        if marker.exists():
            cached_path = Path(marker.read_text().strip())
            if cached_path.is_dir():
                if progress_callback:
                    progress_callback(1.0, "Qwen3 found in cache — no download needed.")
                return str(cached_path)

        if progress_callback:
            progress_callback(0.0, "Checking Qwen3 download size…")
        info = HfApi().model_info(model_id, files_metadata=True)
        total_bytes = sum(file.size or 0 for file in (info.siblings or []))

        def download() -> str:
            return snapshot_download(model_id, revision=info.sha, max_workers=4)

        with ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(download)
            while not future.done():
                downloaded = (
                    sum(
                        file.stat().st_size
                        for file in (storage / "blobs").glob("*")
                        if file.is_file()
                    )
                    if (storage / "blobs").exists()
                    else 0
                )
                fraction = min(downloaded / total_bytes, 0.99) if total_bytes else 0.0
                if progress_callback:
                    progress_callback(
                        fraction,
                        f"Downloading Qwen3: {downloaded / 1e9:.2f} / {total_bytes / 1e9:.2f} GB",
                    )
                time.sleep(0.25)
            cached_path = future.result()

        storage.mkdir(parents=True, exist_ok=True)
        marker.write_text(cached_path)
        if progress_callback:
            progress_callback(1.0, "Qwen3 download complete. Loading model…")
        return cached_path

    def generate(
        self,
        messages: list[dict[str, str]],
        temperature: float = 0.0,
        max_new_tokens: int = 512,
    ) -> str:
        template_args = dict(tokenize=False, add_generation_prompt=True)
        try:
            prompt = self.tokenizer.apply_chat_template(
                messages, enable_thinking=False, **template_args
            )
        except TypeError:
            prompt = self.tokenizer.apply_chat_template(messages, **template_args)

        if self.backend == "mlx":
            from mlx_lm import generate

            return generate(
                self.model,
                self.tokenizer,
                prompt=prompt,
                max_tokens=max_new_tokens,
                verbose=False,
            ).strip()

        import torch

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        do_sample = temperature > 0
        kwargs = {
            "max_new_tokens": max_new_tokens,
            "do_sample": do_sample,
            "pad_token_id": self.tokenizer.eos_token_id,
        }
        if do_sample:
            kwargs.update(temperature=temperature, top_p=0.9)
        with torch.inference_mode():
            output = self.model.generate(**inputs, **kwargs)
        generated = output[0, inputs["input_ids"].shape[1] :]
        return self.tokenizer.decode(generated, skip_special_tokens=True).strip()


def parse_json(text: str) -> dict:
    """Parse JSON even when a local model adds think tags or surrounding prose."""
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if not match:
            raise
        return json.loads(match.group(0))
