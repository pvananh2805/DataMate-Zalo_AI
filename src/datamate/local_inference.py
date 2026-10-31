"""CUDA-only local Transformers backend with offline, deterministic generation."""

from __future__ import annotations

import os
import threading
from pathlib import Path
from typing import Any

from .question_understanding import QuestionUnderstandingError


class LocalModelError(QuestionUnderstandingError):
    """Raised when a local inference checkpoint cannot be loaded or used safely."""


class LocalTransformersBackend:
    """A lazy-loaded, single-model backend for the model-independent planner adapter."""

    def __init__(
        self,
        model_path: Path | str,
        *,
        precision: str = "bf16",
        max_new_tokens: int = 1024,
        max_prompt_tokens: int = 16_384,
        parameter_cap: int = 15_000_000_000,
        cpu_threads: int = 2,
    ) -> None:
        self.model_path = Path(model_path).resolve()
        if not self.model_path.is_dir():
            raise LocalModelError("model_path must be a local checkpoint directory")
        if precision not in {"fp32", "fp16", "bf16"}:
            raise ValueError("precision must be fp32, fp16, or bf16")
        if not 1 <= max_new_tokens <= 4096:
            raise ValueError("max_new_tokens must be between 1 and 4096")
        if max_prompt_tokens < 1 or parameter_cap < 1 or cpu_threads < 1:
            raise ValueError("token limits, parameter cap, and threads must be positive")
        self.precision = precision
        self.max_new_tokens = max_new_tokens
        self.max_prompt_tokens = max_prompt_tokens
        self.parameter_cap = parameter_cap
        self.cpu_threads = cpu_threads
        self.model_name = self.model_path.name
        self.parameter_count: int | None = None
        self.model_storage_bytes: int | None = None
        self.cuda_free_bytes_before_transfer: int | None = None
        self.cuda_total_bytes: int | None = None
        self._model: Any = None
        self._tokenizer: Any = None
        self._torch: Any = None
        self._lock = threading.Lock()

    def _load(self) -> None:
        if self._model is not None:
            return
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        try:
            import torch
        except ImportError as exc:
            raise LocalModelError(
                "PyTorch is required for local inference; no remote API fallback exists"
            ) from exc
        if not torch.cuda.is_available():
            raise LocalModelError(
                "Local Transformers inference is configured for CUDA; no CPU fallback"
            )
        if self.precision == "bf16" and not torch.cuda.is_bf16_supported():
            raise LocalModelError("Selected GPU does not support bf16")
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:
            raise LocalModelError(
                "Transformers is required; install the selected model's runtime dependencies"
            ) from exc

        torch.set_num_threads(self.cpu_threads)
        dtype = {
            "fp32": torch.float32,
            "fp16": torch.float16,
            "bf16": torch.bfloat16,
        }[self.precision]
        tokenizer = AutoTokenizer.from_pretrained(
            str(self.model_path),
            local_files_only=True,
            trust_remote_code=False,
        )
        if tokenizer.pad_token_id is None:
            if tokenizer.eos_token_id is None:
                raise LocalModelError("Tokenizer must define a pad token or EOS token")
            tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained(
            str(self.model_path),
            local_files_only=True,
            trust_remote_code=False,
            torch_dtype=dtype,
        )
        parameter_count = sum(parameter.numel() for parameter in model.parameters())
        if parameter_count > self.parameter_cap:
            raise LocalModelError(
                f"Model has {parameter_count:,} parameters, above cap "
                f"{self.parameter_cap:,}"
            )
        model_storage_bytes = sum(
            parameter.numel() * parameter.element_size()
            for parameter in model.parameters()
        ) + sum(
            buffer.numel() * buffer.element_size()
            for buffer in model.buffers()
        )
        try:
            free_bytes, total_bytes = torch.cuda.mem_get_info(0)
        except RuntimeError as exc:
            raise LocalModelError(
                f"Cannot inspect CUDA device memory before loading {self.model_name}"
            ) from exc
        if model_storage_bytes > free_bytes:
            raise LocalModelError(
                f"Model weights and buffers need at least {model_storage_bytes:,} CUDA "
                f"bytes, but only {free_bytes:,} are free. This estimate excludes "
                "generation cache and runtime workspace."
            )
        model.to(torch.device("cuda:0"))
        model.eval()
        self._model = model
        self._tokenizer = tokenizer
        self._torch = torch
        self.parameter_count = parameter_count
        self.model_storage_bytes = model_storage_bytes
        self.cuda_free_bytes_before_transfer = free_bytes
        self.cuda_total_bytes = total_bytes

    def load(self) -> None:
        """Load the selected checkpoint once; subsequent calls are idempotent."""
        with self._lock:
            self._load()

    def generate(self, prompt: str) -> str:
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError("prompt must be non-empty text")
        with self._lock:
            self._load()
            tokenizer = self._tokenizer
            torch = self._torch
            if getattr(tokenizer, "chat_template", None):
                encoded = tokenizer.apply_chat_template(
                    [{"role": "user", "content": prompt}],
                    tokenize=True,
                    add_generation_prompt=True,
                    return_tensors="pt",
                    return_dict=True,
                )
            else:
                encoded = tokenizer(
                    prompt,
                    return_tensors="pt",
                    add_special_tokens=True,
                    truncation=False,
                )
            input_ids = encoded["input_ids"]
            if input_ids.shape[-1] > self.max_prompt_tokens:
                raise LocalModelError(
                    f"Prompt has {input_ids.shape[-1]} tokens; limit is "
                    f"{self.max_prompt_tokens}"
                )
            inputs = {
                key: value.to(torch.device("cuda:0"))
                for key, value in encoded.items()
                if key in {"input_ids", "attention_mask"}
            }
            generation_options: dict[str, object] = {
                "max_new_tokens": self.max_new_tokens,
                "do_sample": False,
                "num_beams": 1,
                "use_cache": True,
                "pad_token_id": tokenizer.pad_token_id,
            }
            if tokenizer.eos_token_id is not None:
                generation_options["eos_token_id"] = tokenizer.eos_token_id
            with torch.inference_mode():
                generated = self._model.generate(**inputs, **generation_options)
            continuation = generated[0, input_ids.shape[-1] :]
            if continuation.numel() == 0:
                raise LocalModelError("Model returned an empty completion")
            return tokenizer.decode(
                continuation,
                skip_special_tokens=True,
            ).strip()
