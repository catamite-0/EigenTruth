# EigenTruth Examples

The scripts in this directory are qualitative research demonstrations. They are useful for learning the wrapper API and exploring experiment design. They are not benchmarks and do not establish factuality or production safety.

## Included Scripts

### `qwen_truth_demo.py`

A minimal end-to-end demonstration using `Qwen/Qwen2.5-0.5B-Instruct`. It loads a model, builds small factual and false warmup sets, generates with experimental steering enabled, prints diagnostics, detaches the probe, and generates again for comparison.

### `adversarial_test.py`

A larger qualitative comparison over several prompts. It prints generated text with and without steering, plus Mahalanobis-style distance and HSE diagnostics. Output differences should be treated as observations, not proof of factual correction.

### `calibrated_control_demo.py`

A dependency-light control-plane demonstration. It loads a `CalibrationArtifact`
or uses built-in toy thresholds, routes diagnostics through `RiskController`,
verifies simple exact-match claims, and prints a JSON `ProductTrace`.

## Running An Example

Install EigenTruth in editable mode and run a script from the repository root:

```bash
python -m pip install -e ".[examples]"
python examples/qwen_truth_demo.py
python examples/calibrated_control_demo.py
```

The examples may download model weights from Hugging Face. Review model licenses, download sizes, and any requirements for remote code before running a new model.

`calibrated_control_demo.py` does not load a model or download data. It is a
small product-flow check for artifact-driven diagnostics and trace output:

```bash
python examples/calibrated_control_demo.py \
  --diagnostics '{"maha_last": 4.2, "subspace_resid": 0.4}'
```

It can also use the dependency-free lexical groundedness verifier. Pass evidence
snippets as a JSON list and optional explicit refutations as a JSON object:

```bash
python examples/calibrated_control_demo.py \
  --text "Paris is the capital of France. The moon is made of cheese." \
  --evidence '[{"source": "atlas", "text": "Paris is the capital of France."}, {"source": "nasa", "text": "The moon is not made of cheese; lunar samples are rock."}]'
```

## Structure For New Example Scripts

New examples should be easy to inspect and reproduce. Keep this sequence explicit:

1. State the research question and limitations in the module docstring.
2. Define the model identifier and, when available, the exact model revision.
3. Set deterministic seeds when sampling or randomized data is involved.
4. Declare warmup dataset provenance and include or link the factual and false examples.
5. Record target layer, thresholds, steering strength, and generation arguments.
6. Separate monitor-only and steering-enabled runs clearly.
7. Print or save diagnostics alongside generated output.
8. Document hardware, dependency versions, and expected runtime for heavier experiments.

Prefer small scripts with a `main()` entry point. Reusable experiment utilities should move into a dedicated module when they become substantial.

## Interpreting Results

- A changed output is not proof of improved truthfulness.
- A lower distance is not proof that an output is correct.
- HSE is an experimental dispersion signal, not a calibrated risk score.
- Thresholds are specific to the model, layer, warmup set, and generation configuration.
- Research claims require external evaluation and human review.

## 示例说明

本目录中的脚本属于定性研究演示。它们适合用于学习 wrapper API 和探索实验设计，但不是基准测试，也不能证明事实性或生产安全性。

新增示例脚本时，请明确研究问题和局限性，记录模型标识与 revision、随机种子、warmup 数据来源、目标层、阈值、激活引导强度、生成参数、依赖版本、硬件环境和预期运行时间。请清晰区分纯监测运行和启用引导的运行，并将诊断指标与生成输出一起记录。

输出发生变化不能证明真实性有所提升。更低的距离不能证明输出正确。HSE 是实验性离散指标，不是经过校准的风险评分。任何研究结论都需要外部评估和人工审查。
