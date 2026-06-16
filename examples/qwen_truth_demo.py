"""EigenTruth Demo — Qwen2.5-0.5B 对比式表征诊断与实验性引导示例
EigenTruth Demo — Qwen2.5-0.5B contrastive representation diagnostics and experimental steering
"""

import logging
import sys

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from eigentruth import EigenTruthWrapper


def main():
    model_name = "Qwen/Qwen2.5-0.5B-Instruct"
    print(f"Loading {model_name}...")

    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float32,
        )
    except Exception as e:
        print(
            f"❌ 模型加载失败 / Failed to load model: {e}\n"
            f"请检查网络连接或使用 `huggingface-cli login` 登录。\n"
            f"Please check network connection or login with `huggingface-cli login`."
        )
        sys.exit(1)

    # 选取模型倒数第8层进行干预（0.5B 模型有 24 层）
    # Select the -8th layer for intervention (0.5B model has 24 layers)
    # steering_lambda 为"相对激活范数的移动比例"：0.1 ≈ 沿引导方向移动激活范数的 10%。
    # mahalanobis_threshold 的尺度已按样本数归一化，但仍需按模型/层校准。
    # steering_lambda is a fraction of the activation norm (0.1 ≈ 10% of the norm).
    # The threshold scale is sample-count-normalized but still needs per-model/layer calibration.
    monitor = EigenTruthWrapper(
        model=model,
        target_layer_idx=-8,
        steering_lambda=0.1,
        mahalanobis_threshold=5.0,
    )

    # 通过事实与错误事实的对比，提取 warmup 流形与对比方向
    # Extract warmup manifold and contrastive direction via fact/false comparison
    fact_dataset = [
        "The capital of France is Paris.",
        "Water boils at 100 degrees Celsius.",
        "The Earth revolves around the Sun.",
        "Albert Einstein developed the theory of relativity.",
        "Photosynthesis is the process by which plants make their food.",
    ]

    false_dataset = [
        "The capital of France is London.",
        "Water boils at 50 degrees Celsius.",
        "The Sun revolves around the Earth.",
        "Albert Einstein invented the telephone.",
        "Photosynthesis is how animals digest food.",
    ]

    print("Warming up EigenTruth Manifold...")
    monitor.warmup(fact_dataset, tokenizer, false_dataset=false_dataset)
    print(monitor.get_diagnostics())

    # 构建带错误前提的 prompt，用于观察诊断与引导行为
    # Construct a prompt with a false premise to inspect diagnostics and steering
    prompt = "Tell me a fun fact. Did you know that the capital of France is"
    inputs = tokenizer(prompt, return_tensors="pt")

    print("\n--- Generating WITH EigenTruth (Contrastive Steering) ---")
    outputs_with_monitor = monitor.generate(**inputs, max_new_tokens=10, do_sample=False)
    print("Output:", tokenizer.decode(outputs_with_monitor[0], skip_special_tokens=True))
    print(f"Max Mahalanobis distance during generation: {monitor.last_distance:.2f}")
    if monitor.last_hse > 0:
        print(f"Max HSE during generation: {monitor.last_hse:.2f}")

    print("\n--- Generating WITHOUT EigenTruth ---")
    monitor.detach_probe()
    outputs_without_monitor = monitor.generate(**inputs, max_new_tokens=10, do_sample=False)
    print("Output:", tokenizer.decode(outputs_without_monitor[0], skip_special_tokens=True))


if __name__ == "__main__":
    # 配置 eigentruth 日志输出（库本身不输出日志，需要调用方配置）
    # Configure eigentruth log output (library itself uses NullHandler)
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("[EigenTruth] %(message)s"))
    et_logger = logging.getLogger("eigentruth")
    et_logger.addHandler(handler)
    et_logger.setLevel(logging.INFO)

    main()
