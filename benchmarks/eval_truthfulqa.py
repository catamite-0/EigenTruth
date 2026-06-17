"""EigenTruth 基准评测 — 隐状态几何信号能否分离真/假陈述 (AUROC)。
EigenTruth benchmark — can hidden-state geometry separate true vs. false statements (AUROC)?

研究问题 / Research questions:
    1. 流形马氏距离 (maha_last) 能否把"假陈述"排在"真陈述"之上？ 它打得过困惑度基线 (nll) 吗？
       Does manifold Mahalanobis distance rank false statements above true ones, and beat a
       perplexity baseline?
    2. 双曲离散度 (disp_hse) 是否优于欧氏离散度 (disp_euclid)？ —— 即"双曲几何有没有用"的消融。
       Does hyperbolic dispersion (disp_hse) beat Euclidean dispersion (disp_euclid)? — the
       "does the hyperbolic projection earn its keep?" ablation.
    3. 对比方向投影 (truth_proj，即工具自带的 contrastive_direction 用作 mass-mean 探针，
       参见 Marks & Tegmark) 是否是更强的检测器？最佳目标层在哪 (--sweep)？
       Is the contrastive-direction projection (the tool's own steering direction used as a
       mass-mean probe, cf. Marks & Tegmark) an even stronger detector, and which layer is
       best (--sweep)?

方法 / Method (SAPLMA 式、确定性、无需 LLM 裁判 / SAPLMA-style, deterministic, judge-free):
    - 从**留出**题目的*正确*答案构建真值流形（无标签泄漏）。
      Build the truth manifold from the *correct* answers of **held-out** questions (no leakage).
    - 对其余题目的每条候选答案（正确=负类, 错误=正类/幻觉）做单次前向，提取目标层隐状态。
      For each candidate answer of the remaining questions (correct=negative, incorrect=positive),
      run a single forward pass and read the target-layer hidden states.
    - 每条陈述计算 4 个信号的分数，分别报告 AUROC。
      Score each statement with 4 signals and report AUROC per signal.

局限 / Limitations:
    - 强制候选答案的"陈述级"打分是开放生成幻觉的*代理*，干净地检验表征假设但不等同于在线检测。
      Forced-answer statement scoring is a *proxy* for open-generation hallucination; it cleanly
      tests the representation hypothesis but is not the same as online detection.
    - 句内 token 离散度是基于多次采样的语义熵 (Farquhar et al.) 的廉价代理。
      Within-statement token dispersion is a cheap proxy for sample-based semantic entropy.
    - 小模型 + 几百条样本 → AUROC 置信区间较宽，结果仅供参考，非定论。
      Small model + a few hundred items → wide AUROC CIs; results are indicative, not conclusive.

用法 / Usage:
    # 真实基准 / real benchmark (downloads model + TruthfulQA):
    python benchmarks/eval_truthfulqa.py --model Qwen/Qwen2.5-0.5B-Instruct --layer -8 --limit 200
    # 快速管线自检 / fast pipeline smoke check (tiny model, bundled statements, no dataset):
    python benchmarks/eval_truthfulqa.py --model sshleifer/tiny-gpt2 --offline
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from typing import List, Mapping, Optional, Sequence

import torch

from eigentruth.calibration import DEFAULT_SCORE_DIRECTIONS
from eigentruth.core import TruthSubspace
from eigentruth.core.math_engine import (
    TruthManifold,
    hyperbolic_semantic_entropy,
    mahalanobis_distance,
    poincare_map,
)
from eigentruth.eval.conformal import directional_conformal_threshold
from eigentruth.eval.metrics import euclidean_dispersion, roc_auc, selective_classification_report

SIGNALS = ["maha_last", "truth_proj", "subspace_resid", "disp_euclid", "disp_hse", "nll_answer"]
REPORT_ALPHA = 0.10


@dataclass
class Statement:
    question: str
    answer: str
    is_false: int  # 1 = 错误答案(正类/幻觉) / incorrect (positive), 0 = 正确答案(负类)


def _selective_reports(
    scores: Mapping[str, Sequence[float]],
    labels: Sequence[int],
    *,
    alpha: float = REPORT_ALPHA,
    directions: Mapping[str, str] | None = None,
) -> dict[str, dict]:
    labels_t = torch.tensor(labels)
    reports = {}
    for signal, values in scores.items():
        direction = (directions or DEFAULT_SCORE_DIRECTIONS).get(signal, "higher")
        signal_scores = torch.tensor(values, dtype=torch.float64)
        true_scores = signal_scores[labels_t == 0]
        threshold = directional_conformal_threshold(true_scores, alpha, direction)
        reports[signal] = {
            "alpha": alpha,
            **selective_classification_report(signal_scores, labels_t, threshold, direction=direction),
        }
    return reports


# ---------------------------------------------------------------------------
# 离线烟雾集（仅用于验证管线，不构成基准）/ Offline smoke set (pipeline check only)
# ---------------------------------------------------------------------------

_TRUE_SMOKE = [
    "The capital of France is Paris.",
    "Water boils at 100 degrees Celsius at sea level.",
    "The Earth revolves around the Sun.",
    "Humans have two lungs.",
    "The Pacific is the largest ocean on Earth.",
    "Ice is frozen water.",
    "A triangle has three sides.",
    "The Sun rises in the east.",
    "Honey is made by bees.",
    "Mount Everest is the tallest mountain above sea level.",
]
_FALSE_SMOKE = [
    "The capital of France is Berlin.",
    "Water boils at 30 degrees Celsius at sea level.",
    "The Sun revolves around the Earth.",
    "Humans have five lungs.",
    "The Pacific is the smallest ocean on Earth.",
    "Ice is boiling water.",
    "A triangle has seven sides.",
    "The Sun rises in the west.",
    "Honey is made by spiders.",
    "Mount Everest is the shortest mountain on Earth.",
]


def load_offline() -> tuple[List[str], List[str], List[Statement]]:
    """返回 (流形构建用真陈述, 对比方向用假陈述, 评测陈述)。"""
    manifold_true = _TRUE_SMOKE[:6]
    manifold_false = _FALSE_SMOKE[:6]
    eval_stmts: List[Statement] = []
    for t in _TRUE_SMOKE[6:]:
        eval_stmts.append(Statement("", t, 0))
    for f in _FALSE_SMOKE[6:]:
        eval_stmts.append(Statement("", f, 1))
    return manifold_true, manifold_false, eval_stmts


def load_truthfulqa(
    manifold_questions: int, limit: int
) -> tuple[List[str], List[str], List[Statement]]:
    """加载 TruthfulQA multiple_choice，切分为流形集 / 评测集（题目层面不重叠）。

    流形集题目的正确答案用于构建真值流形，错误答案仅用于 mass-mean 对比方向。
    Correct answers of manifold-split questions build the truth manifold; their
    incorrect answers are used only for the mass-mean contrastive direction.
    """
    from datasets import load_dataset  # lazy

    # 新旧版 datasets 的数据集 id 不同，依次尝试 / dataset id differs across versions
    last_err: Optional[Exception] = None
    ds = None
    for dataset_id in ("truthfulqa/truthful_qa", "truthful_qa"):
        try:
            ds = load_dataset(dataset_id, "multiple_choice")["validation"]
            break
        except Exception as exc:  # noqa: BLE001
            last_err = exc
    if ds is None:
        raise RuntimeError(
            f"Could not load TruthfulQA. Last error: {last_err}. "
            f"Try `pip install -U datasets` or run with --offline for a pipeline check."
        )

    manifold_true: List[str] = []
    manifold_false: List[str] = []
    eval_stmts: List[Statement] = []
    for i, row in enumerate(ds):
        q = row["question"]
        targets = row["mc2_targets"]
        choices, labels = targets["choices"], targets["labels"]
        if i < manifold_questions:
            for c, lab in zip(choices, labels):
                if lab == 1:
                    manifold_true.append(f"{q} {c}")
                else:
                    manifold_false.append(f"{q} {c}")
        else:
            for c, lab in zip(choices, labels):
                eval_stmts.append(Statement(q, c, is_false=int(lab == 0)))
        if limit and (i - manifold_questions + 1) >= limit and i >= manifold_questions:
            break
    return manifold_true, manifold_false, eval_stmts


# ---------------------------------------------------------------------------
# 模型与表征提取 / Model and representation extraction
# ---------------------------------------------------------------------------

_DTYPES = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}


def load_model(model_name: str, device: torch.device, dtype: str = "float32"):
    from transformers import AutoModelForCausalLM, AutoTokenizer  # lazy

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    # dtype= 替代已弃用的 torch_dtype=；bfloat16 减半权重内存；
    # low_cpu_mem_usage 避免加载时 2× 峰值内存（低内存机器关键）
    model = AutoModelForCausalLM.from_pretrained(
        model_name, dtype=_DTYPES[dtype], low_cpu_mem_usage=True
    )
    model.to(device).eval()
    return model, tokenizer


def resolve_target_layer(layer: int, n_layers: int, *, offline: bool) -> int:
    """Return a hidden-state layer index valid for the loaded model."""
    if -(n_layers + 1) <= layer <= n_layers:
        return layer
    if offline and layer == -8:
        print(f"[!] Requested layer {layer} is unavailable for this tiny smoke model; using -1.\n")
        return -1
    raise ValueError(
        f"layer {layer} is out of range for a model with {n_layers} transformer layers "
        f"(valid hidden-state indexes: {-n_layers - 1}..{n_layers})."
    )


@torch.no_grad()
def statement_reps(model, tokenizer, stmt: Statement, layers: List[int],
                   device: torch.device, max_length: int) -> Optional[dict]:
    """单次前向：各目标层的末 token 隐状态、主层 (layers[0]) 答案 token 隐状态、答案 NLL。
    Single forward pass: last-token hidden state per requested layer, answer-token hidden
    states for the primary layer (layers[0]), and the answer NLL. output_hidden_states
    returns every layer at once, so a layer sweep costs no extra forward passes.
    """
    q_ids = tokenizer(stmt.question, add_special_tokens=True).input_ids if stmt.question \
        else tokenizer(tokenizer.bos_token or tokenizer.eos_token or " ").input_ids
    a_ids = tokenizer(" " + stmt.answer.strip(), add_special_tokens=False).input_ids
    if len(a_ids) == 0:
        return None
    ids = (q_ids + a_ids)[:max_length]
    n_ans = len(ids) - len(q_ids)
    if n_ans <= 0:
        return None

    input_ids = torch.tensor([ids], device=device)
    out = model(input_ids=input_ids, output_hidden_states=True)

    # hidden_states[-k] 对应 layers[-k] 的输出（负索引对齐，与 wrapper 约定一致）
    last_by_layer = {
        layer: out.hidden_states[layer][0][-1, :].float().cpu() for layer in layers
    }
    ans_hs = out.hidden_states[layers[0]][0][-n_ans:, :].float().cpu()

    # 答案 token 的平均负对数似然（困惑度基线）
    logits = out.logits[0].float()  # [T, V]
    logp = torch.log_softmax(logits[:-1], dim=-1)  # 由位置 t 预测 t+1
    targets = input_ids[0, 1:]
    tok_logp = logp[torch.arange(logp.shape[0]), targets]  # [T-1]
    ans_logp = tok_logp[-n_ans:] if n_ans <= tok_logp.shape[0] else tok_logp
    nll = float((-ans_logp.mean()).item())

    return {"last": last_by_layer, "ans_hs": ans_hs, "nll": nll}


def build_layer_stats(model, tokenizer, true_texts: List[str], false_texts: List[str],
                      layers: List[int], device: torch.device, max_length: int,
                      subspace_rank: int) -> tuple[dict, dict]:
    """逐层构建真值流形与 mass-mean 对比方向（与 EigenTruthWrapper.warmup 同构）。
    Per-layer truth manifolds plus the mass-mean contrastive direction
    (mirrors EigenTruthWrapper.warmup; cf. Marks & Tegmark mass-mean probing).
    """
    manifolds = {layer: TruthManifold() for layer in layers}
    true_state_lists = {layer: [] for layer in layers}
    false_state_lists = {layer: [] for layer in layers}
    false_sums: dict = {layer: None for layer in layers}
    n_false = 0

    for t in true_texts:
        reps = statement_reps(model, tokenizer, Statement("", t, 0), layers, device, max_length)
        if reps is None:
            continue
        for layer in layers:
            h = reps["last"][layer]
            manifolds[layer].update(h)
            true_state_lists[layer].append(h)

    for t in false_texts:
        reps = statement_reps(model, tokenizer, Statement("", t, 1), layers, device, max_length)
        if reps is None:
            continue
        n_false += 1
        for layer in layers:
            h = reps["last"][layer]
            false_state_lists[layer].append(h)
            false_sums[layer] = h if false_sums[layer] is None else false_sums[layer] + h

    subspaces = {}
    for layer in layers:
        m = manifolds[layer]
        if n_false > 0 and m.mean is not None:
            m.false_mean = (false_sums[layer] / n_false).to(torch.float32)
            raw = m.mean - m.false_mean
            m.contrastive_direction = raw / torch.norm(raw).clamp(min=1e-8)

        true_states = true_state_lists[layer]
        false_states = false_state_lists[layer]
        if len(true_states) >= 2 and false_states:
            subspaces[layer] = TruthSubspace.fit_contrastive(
                torch.stack(true_states), torch.stack(false_states), rank=subspace_rank
            )
        elif len(true_states) >= 2:
            subspaces[layer] = TruthSubspace.fit(torch.stack(true_states), rank=subspace_rank)
    return manifolds, subspaces


# ---------------------------------------------------------------------------
# 主流程 / Main
# ---------------------------------------------------------------------------

def run(args) -> dict:
    # Windows 控制台默认 cp1252，确保非 ASCII 不崩溃；行缓冲让进度实时可见（被杀也不丢日志）
    try:
        sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
    except Exception:
        pass

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)

    if args.offline:
        manifold_true, manifold_false, eval_stmts = load_offline()
        print("[!] OFFLINE SMOKE MODE - pipeline check only, NOT a benchmark.\n")
    else:
        manifold_true, manifold_false, eval_stmts = load_truthfulqa(
            args.manifold_questions, args.limit
        )

    print(f"Loading {args.model} on {device} (dtype={args.dtype}) ...")
    model, tokenizer = load_model(args.model, device, args.dtype)

    n_layers = int(model.config.num_hidden_layers)
    args.layer = resolve_target_layer(args.layer, n_layers, offline=args.offline)
    # --sweep: 主层在前，其余层按 hidden_states 负索引补全（同一次前向全部免费拿到）
    if args.sweep:
        layers = [args.layer] + [
            -(i + 1) for i in range(n_layers) if -(i + 1) != args.layer
        ]
    else:
        layers = [args.layer]

    print(f"Building per-layer truth stats from {len(manifold_true)} true / "
          f"{len(manifold_false)} false statements ({len(layers)} layer(s)) ...")
    manifolds, subspaces = build_layer_stats(
        model, tokenizer, manifold_true, manifold_false, layers, device, args.max_length, args.subspace_rank
    )
    primary = manifolds[args.layer]
    if not primary.is_ready():
        print("[X] Manifold not ready (need >=2 statements). Aborting.")
        sys.exit(1)
    print(f"   manifold: n={primary.n}, hidden_dim={primary.hidden_dim}, "
          f"contrastive_direction={'yes' if primary.contrastive_direction is not None else 'no'}  "
          f"subspace={'yes' if args.layer in subspaces else 'no'}\n")

    scores: dict[str, List[float]] = {s: [] for s in SIGNALS}
    sweep_scores: dict = {
        layer: {"maha_last": [], "truth_proj": [], "subspace_resid": []} for layer in layers
    }
    labels: List[int] = []

    print(f"Scoring {len(eval_stmts)} eval statements ...")
    for k, stmt in enumerate(eval_stmts):
        reps = statement_reps(model, tokenizer, stmt, layers, device, args.max_length)
        if reps is None:
            continue

        for layer in layers:
            m = manifolds[layer]
            h = reps["last"][layer]
            sweep_scores[layer]["maha_last"].append(
                float(mahalanobis_distance(h, m.mean, m.cov_inv).item())
            )
            # 沿真值方向的投影越小越可疑：score = -(h · direction)
            # Lower projection onto the truth direction = more suspect
            if m.contrastive_direction is not None:
                proj = -float(torch.dot(h, m.contrastive_direction).item())
            else:
                proj = 0.0
            sweep_scores[layer]["truth_proj"].append(proj)
            subspace = subspaces.get(layer)
            if subspace is not None and subspace.is_ready():
                resid = float(subspace.residual_distance(h).item())
            else:
                resid = 0.0
            sweep_scores[layer]["subspace_resid"].append(resid)

        ans = reps["ans_hs"]
        scores["maha_last"].append(sweep_scores[args.layer]["maha_last"][-1])
        scores["truth_proj"].append(sweep_scores[args.layer]["truth_proj"][-1])
        scores["subspace_resid"].append(sweep_scores[args.layer]["subspace_resid"][-1])
        scores["disp_euclid"].append(float(euclidean_dispersion(ans).item()))
        scores["disp_hse"].append(
            float(hyperbolic_semantic_entropy(poincare_map(ans)).item())
        )
        scores["nll_answer"].append(reps["nll"])
        labels.append(stmt.is_false)

        if (k + 1) % 50 == 0:
            print(f"   {k + 1}/{len(eval_stmts)}")

    n_pos = sum(labels)
    n_neg = len(labels) - n_pos
    results = {s: roc_auc(scores[s], labels) for s in SIGNALS}
    selective = _selective_reports(scores, labels, alpha=REPORT_ALPHA)

    # ---- 输出 ----
    print("\n" + "=" * 56)
    print("  AUROC  (positive = false/hallucinated statement)")
    print(f"  model={args.model}  layer={args.layer}  n_pos={n_pos}  n_neg={n_neg}")
    print("=" * 56)
    print(f"  {'signal':<14}{'AUROC':>10}   interpretation")
    print("  " + "-" * 52)
    for s in SIGNALS:
        print(f"  {s:<14}{results[s]:>10.3f}")
    print("  " + "-" * 52)
    # 关键对比 / key comparisons
    if not (results["disp_hse"] != results["disp_hse"]):  # not NaN
        delta = results["disp_hse"] - results["disp_euclid"]
        verdict = "hyperbolic HELPS" if delta > 0.01 else (
            "hyperbolic HURTS" if delta < -0.01 else "no meaningful difference")
        print(f"  disp_hse - disp_euclid = {delta:+.3f}  ->  {verdict}")
    geo = max(results["maha_last"], results["truth_proj"], results["subspace_resid"],
              results["disp_hse"], results["disp_euclid"])
    if not (results["nll_answer"] != results["nll_answer"]):
        print(f"  best geometry ({geo:.3f}) vs nll baseline ({results['nll_answer']:.3f})  ->  "
              f"{'geometry wins' if geo > results['nll_answer'] + 0.01 else 'baseline competitive'}")
    print("=" * 56)

    sweep_payload = None
    if args.sweep:
        sweep_payload = {}
        print("\n  Layer sweep (AUROC):")
        print(f"  {'layer':>6} {'maha_last':>11} {'truth_proj':>11} {'subspace':>11}")
        for layer in sorted(layers):
            r_m = roc_auc(sweep_scores[layer]["maha_last"], labels)
            r_p = roc_auc(sweep_scores[layer]["truth_proj"], labels)
            r_s = roc_auc(sweep_scores[layer]["subspace_resid"], labels)
            sweep_payload[str(layer)] = {"maha_last": r_m, "truth_proj": r_p, "subspace_resid": r_s}
            print(f"  {layer:>6} {r_m:>11.3f} {r_p:>11.3f} {r_s:>11.3f}")

    payload = {
        "config": {"model": args.model, "layer": args.layer, "offline": args.offline,
                   "manifold_n": primary.n, "n_manifold_false": len(manifold_false),
                   "hidden_dim": primary.hidden_dim, "subspace_rank": args.subspace_rank,
                   "n_pos": n_pos, "n_neg": n_neg, "seed": args.seed},
        "auroc": results,
        "selective": selective,
        "sweep": sweep_payload,
    }
    if args.json:
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        print(f"\nWrote structured results to {args.json}")
    if args.dump_scores:
        # 逐陈述原始分数：供共形校准等后处理复用，无需再跑模型
        # Raw per-statement scores: enables post-hoc analyses (e.g. conformal
        # calibration) without re-running the model
        dump = {"config": payload["config"], "labels": labels, "scores": scores}
        if args.sweep:
            dump["sweep_scores"] = {str(layer): sweep_scores[layer] for layer in layers}
        with open(args.dump_scores, "w", encoding="utf-8") as f:
            json.dump(dump, f)
        print(f"Dumped raw per-statement scores to {args.dump_scores}")
    print("\nJSON:", json.dumps(payload["auroc"]))
    return payload


def main():
    p = argparse.ArgumentParser(description="EigenTruth TruthfulQA AUROC benchmark")
    p.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    p.add_argument("--dtype", default="float32", choices=list(_DTYPES),
                   help="model weight dtype; bfloat16 halves memory on low-RAM machines")
    p.add_argument("--layer", type=int, default=-8, help="target layer index (negative ok)")
    p.add_argument("--sweep", action="store_true",
                   help="score geometry signals at every layer (free: one forward pass already "
                        "returns all hidden states)")
    p.add_argument("--limit", type=int, default=200, help="max eval questions (0 = all)")
    p.add_argument("--manifold-questions", type=int, default=80,
                   help="held-out questions whose correct answers build the manifold")
    p.add_argument("--max-length", type=int, default=64)
    p.add_argument("--subspace-rank", type=int, default=2,
                   help="rank for TruthSubspace residual scoring")
    p.add_argument("--offline", action="store_true",
                   help="use bundled smoke statements (pipeline check, not a benchmark)")
    p.add_argument("--json", default=None, help="optional path to write structured results")
    p.add_argument("--dump-scores", default=None,
                   help="optional path to dump raw per-statement scores+labels "
                        "(enables post-hoc analyses, e.g. conformal calibration)")
    p.add_argument("--seed", type=int, default=0)
    run(p.parse_args())


if __name__ == "__main__":
    main()
