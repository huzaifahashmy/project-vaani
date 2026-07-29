"""
translate_audio_local.py — direct Transformers path, no vLLM, no WSL.

Runs natively on Windows (or Mac/Linux). Loads Gemma 4 with 4-bit
quantization so it fits on smaller GPUs (tested target: 6GB VRAM), then
sends a .wav file straight to the model for transcription + translation.

This is the same approach as Google's official audio-understanding guide
(pipeline() + AutoProcessor), with 4-bit quantization added so the E4B
model's ~9-10GB of BF16 weights fit into much less VRAM.

Requirements:
  pip install torch accelerate "transformers>=5.10.1" bitsandbytes
  huggingface-cli login   (Gemma models are gated)

Usage:
  python translate_audio_local.py clean.wav
  python translate_audio_local.py patient.wav --source Hindi --target English
  python translate_audio_local.py note.wav --model google/gemma-4-E2B-it --no-quantize
"""

import argparse
import sys
from pathlib import Path

import torch
from transformers import (
    AutoProcessor,
    AutoModelForImageTextToText,
    BitsAndBytesConfig,
    GenerationConfig,
    pipeline,
)

DEFAULT_MODEL_ID = "google/gemma-4-E4B-it"


def load_pipeline(model_id: str, quantize: bool):
    """
    Higher-level path using pipeline() — the officially documented API from
    Google's audio-understanding guide. Handles audio device/dtype plumbing
    internally, which the manual AutoModel/AutoProcessor path in load_model()
    was apparently getting wrong on GPU.
    """
    print(f"Loading {model_id} via pipeline() (first run downloads the model)...")
    model_kwargs = {}
    if quantize:
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
        )
    return pipeline(
        task="any-to-any",
        model=model_id,
        device_map="auto",
        dtype="auto",
        model_kwargs=model_kwargs,
    )


def transcribe_and_translate_via_pipeline(pipe, model_id: str, wav_path: str, source_lang: str, target_lang: str, max_new_tokens: int = 128) -> str:
    config = GenerationConfig.from_pretrained(model_id)
    config.max_new_tokens = max_new_tokens

    prompt = (
        f"Transcribe the following speech segment in {source_lang}, then translate "
        f"it into {target_lang}. When formatting the answer, first output the "
        f"transcription in {source_lang}, then one newline, then output the string "
        f"'{target_lang}: ', then the translation in {target_lang}."
    )

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "audio", "audio": wav_path},
            ],
        }
    ]

    outputs = pipe(messages, return_full_text=False, generate_kwargs=dict(generation_config=config))
    return outputs[0]["generated_text"]


def load_model(model_id: str, quantize: bool, cpu_offload: bool, force_cpu: bool, gpu_only: bool):
    print(f"Loading {model_id} (first run downloads the model)...")

    if force_cpu:
        # Everything on one device (CPU) — sidesteps the meta-device/offload
        # bug entirely. Slow, but the most reliable way to confirm the
        # pipeline itself works before fighting GPU memory further.
        model = AutoModelForImageTextToText.from_pretrained(
            model_id,
            device_map="cpu",
            dtype=torch.float32,
        )
    elif gpu_only and quantize:
        # Force the ENTIRE quantized model onto cuda:0 as one block — no
        # auto-splitting, no CPU offload, no meta-device tensors. This either
        # fits cleanly (and then just works, no device-mismatch bugs
        # possible) or fails with a normal CUDA out-of-memory error instead
        # of the confusing meta-device crash that device_map="auto" +
        # offload produced.
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,  # quantizes the quant constants too, extra ~10-15% savings
        )
        model = AutoModelForImageTextToText.from_pretrained(
            model_id,
            device_map={"": 0},
            quantization_config=bnb_config,
        )
    elif gpu_only and not quantize:
        # Full precision (BF16), single block, GPU only. No quantization
        # noise at all — best shot at matching the quality you saw on CPU,
        # while still being much faster than CPU. Needs more VRAM than the
        # 4-bit path; will OOM cleanly if E2B's full BF16 weights + audio
        # tower + KV cache don't fit in 6GB.
        model = AutoModelForImageTextToText.from_pretrained(
            model_id,
            device_map={"": 0},
            dtype=torch.bfloat16,
        )
    elif quantize:
        # 4-bit quantization — shrinks weights roughly 4x vs BF16.
        # This is what makes E4B feasible on a 6GB GPU.
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            # Without this, bitsandbytes refuses to offload any leftover
            # modules to CPU/disk when the quantized model still doesn't
            # fully fit in VRAM — it just errors out instead. Setting this
            # lets those modules run in 32-bit on the CPU as a fallback.
            llm_int8_enable_fp32_cpu_offload=cpu_offload,
        )
        model_kwargs = dict(
            device_map="auto",
            quantization_config=bnb_config,
            attn_implementation="sdpa",
        )
        if cpu_offload:
            # Cap what's allowed on GPU so accelerate is forced to push the
            # remainder to CPU RAM instead of erroring. Adjust "5GiB" down
            # if you still hit OOM (leaves headroom for KV cache/activations).
            model_kwargs["max_memory"] = {0: "5GiB", "cpu": "24GiB"}
        model = AutoModelForImageTextToText.from_pretrained(model_id, **model_kwargs)
    else:
        model = AutoModelForImageTextToText.from_pretrained(
            model_id,
            device_map="auto",
            dtype="auto",
        )

    processor = AutoProcessor.from_pretrained(model_id)
    return model, processor


def transcribe_and_translate(model, processor, model_id: str, wav_path: str, source_lang: str, target_lang: str, max_new_tokens: int = 128) -> str:
    config = GenerationConfig.from_pretrained(model_id)
    config.max_new_tokens = max_new_tokens

    prompt = (
        f"Transcribe the following speech segment in {source_lang}, then translate "
        f"it into {target_lang}. When formatting the answer, first output the "
        f"transcription in {source_lang}, then one newline, then output the string "
        f"'{target_lang}: ', then the translation in {target_lang}."
    )

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "audio", "audio": wav_path},
            ],
        }
    ]

    inputs = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_tensors="pt",
        return_dict=True,
    ).to(model.device, dtype=model.dtype)

    output_ids = model.generate(**inputs, generation_config=config)

    # Strip the prompt portion, keep only what the model generated.
    new_tokens = output_ids[0][inputs["input_ids"].shape[-1]:]
    return processor.decode(new_tokens, skip_special_tokens=True).strip()


def main():
    parser = argparse.ArgumentParser(description="Translate a .wav file with Gemma 4 — no vLLM, runs natively")
    parser.add_argument("wav_path", help="Path to a WAV file (run make_wav.py first if needed)")
    parser.add_argument("--source", default="English", help="Spoken language in the audio")
    parser.add_argument("--target", default="Hindi", help="Language to translate into")
    parser.add_argument("--model", default=DEFAULT_MODEL_ID, help="HF model id (default: google/gemma-4-E4B-it)")
    parser.add_argument("--no-quantize", action="store_true", help="Load at full precision instead of 4-bit (needs much more VRAM)")
    parser.add_argument("--cpu-offload", action="store_true", help="Allow leftover modules that don't fit in VRAM to run on CPU instead of erroring")
    parser.add_argument("--force-cpu", action="store_true", help="Run entirely on CPU, no GPU at all (slow, but avoids GPU offload bugs)")
    parser.add_argument("--gpu-only", action="store_true", help="Force the whole model onto GPU with no CPU offload (fails cleanly with OOM if it doesn't fit, instead of the meta-device bug)")
    parser.add_argument("--use-pipeline", action="store_true", help="Use the high-level pipeline() API instead of manual AutoModel/AutoProcessor code (recommended if you're seeing wrong/unrelated output)")
    args = parser.parse_args()

    wav_path = Path(args.wav_path)
    if not wav_path.exists():
        sys.exit(f"File not found: {wav_path}")

    if not torch.cuda.is_available() and not args.force_cpu:
        print("Warning: no CUDA GPU detected — this will run on CPU and be significantly slower.")

    if args.use_pipeline:
        pipe = load_pipeline(args.model, quantize=not args.no_quantize)
        print(f"Translating {wav_path} ({args.source} -> {args.target})...")
        result = transcribe_and_translate_via_pipeline(pipe, args.model, str(wav_path), args.source, args.target)
        print("\n--- Gemma 4 output ---")
        print(result)
        return

    model, processor = load_model(args.model, quantize=not args.no_quantize, cpu_offload=args.cpu_offload, force_cpu=args.force_cpu, gpu_only=args.gpu_only)

    print(f"Translating {wav_path} ({args.source} -> {args.target})...")
    result = transcribe_and_translate(model, processor, args.model, str(wav_path), args.source, args.target)

    print("\n--- Gemma 4 output ---")
    print(result)


if __name__ == "__main__":
    main()