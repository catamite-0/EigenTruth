# EigenTruth Examples

The scripts in this directory are qualitative research demonstrations. They are useful for learning the wrapper API and exploring experiment design. They are not benchmarks and do not establish factuality or production safety.

## Included Scripts

### `qwen_truth_demo.py`

A minimal end-to-end demonstration using `Qwen/Qwen2.5-0.5B-Instruct`. It loads a model, builds small factual and false warmup sets, generates with experimental steering enabled, prints diagnostics, detaches the probe, and generates again for comparison.

### `adversarial_test.py`

A larger qualitative comparison over several prompts. It prints generated text with and without steering, plus Mahalanobis-style distance and HSE diagnostics. Output differences should be treated as observations, not proof of factual correction.

### `calibrated_control_demo.py`

A dependency-light control-plane demonstration. It loads a `CalibrationArtifact`
or uses built-in toy thresholds when no repository artifact is present, verifies
simple claims, combines diagnostics and verification results through
`RiskController`, plans an `ActionRequest`, executes it through
`ActionExecutorRegistry`, feeds retrieval hits back into verification when
available, and prints a final JSON `ProductTrace`.
It accepts `--runtime-profile latency|balanced|audit` so the same release
profiles used by benchmark gates can also drive product-loop staging:
`latency` skips low-risk, non-sensitive verifier calls, `balanced` verifies
medium/high/unknown diagnostic risk and sensitive claims, and `audit` runs full
initial verification. It also accepts `--pre-generation-profile auto` to record
a dependency-free prompt/metadata risk assessment before the post-generation
diagnostic selector runs; when no explicit runtime profile is supplied, that
pre-generation assessment can choose the initial `latency`, `balanced`, or
`audit` profile for the trace.

### `sqlite_state_control_demo.py`

A dependency-free structured-state control demonstration. It creates a local
SQLite order/inventory/account fixture, maps read-only SQL query results into
`StructuredStateVerifier`, and emits a `ProductTrace` where database state
refutes one business claim even though internal diagnostics are below the toy
threshold.

### `state_transition_control_demo.py`

A dependency-free world-model control demonstration. It uses
`InMemoryWorldModelAdapter` plus `StateTransitionVerifier` to predict an action's
next state, verify structured postconditions, and emit a `ProductTrace` where a
false claim about the action consequence drives a dry-run `abstain` decision.

### `production_tool_loop_demo.py`

A dependency-free production-like local tool loop. It checks pre-tool business
state from SQLite, executes a deterministic reserve-inventory `execute_tool`
action through `PolicyGuardedActionExecutor`, maps the returned
`ActionResult.output` into `ToolOutputStateSource`, verifies post-tool claims,
optionally replays repeated executions through a JSON or SQLite idempotency ledger, and
stores selected/matched/skipped verifier route counts plus action audit metadata
in the final `ProductTrace`.

## Running An Example

Install EigenTruth in editable mode and run a script from the repository root:

```bash
python -m pip install -e ".[examples]"
python examples/qwen_truth_demo.py
python examples/calibrated_control_demo.py
python examples/sqlite_state_control_demo.py
python examples/state_transition_control_demo.py
python examples/production_tool_loop_demo.py
```

The examples may download model weights from Hugging Face. Review model licenses, download sizes, and any requirements for remote code before running a new model.

`calibrated_control_demo.py` does not load a model or download data. In this
repository it defaults to
`artifacts/smollm2_truthfulqa_l80_best_calibration.json` (`truth_proj`, layer
`-16`) when present, then falls back to
`artifacts/qwen05_truthfulqa_l80_best_calibration.json` and built-in toy
thresholds. It auto-generates diagnostics that cross the configured threshold
and provides a small product-flow check for artifact-driven diagnostics, claim
verification, action planning, dry-run execution, and trace output:

```bash
python examples/calibrated_control_demo.py
```

This default path produces a dry-run `abstain` trace for the built-in mixed
claim text because the artifact diagnostic threshold is exceeded and the second
claim is refuted.

Use runtime profiles to exercise the product control-plane defaults. The
latency profile enables staged verification and skips the verifier when
diagnostics are low and claims lack sensitive metadata:

```bash
python examples/calibrated_control_demo.py \
  --runtime-profile latency \
  --diagnostics '{"truth_proj": 0.0}'
python examples/calibrated_control_demo.py \
  --runtime-profile auto \
  --diagnostics '{"truth_proj": 0.0}'
python examples/calibrated_control_demo.py \
  --runtime-profile auto \
  --runtime-profile-selector-policy artifacts/smollm2_product_runtime_profile_sweep/runtime-profile-selector-policy.json \
  --diagnostics '{"truth_proj": 0.0}'
python examples/calibrated_control_demo.py \
  --pre-generation-profile auto \
  --pre-generation-metadata '{"requires_current_facts": true}' \
  --text "The latest BTC price today is 100 dollars." \
  --diagnostics '{"truth_proj": 0.0}'
```

The balanced profile enables staged verification but still verifies diagnostic
risk or sensitive claims; the audit profile disables staging and verifies all
initial claims. The `auto` selector runs only the cheap diagnostic decision and
claim metadata checks before choosing a profile: low-risk non-sensitive claims
use `latency`, medium diagnostic risk uses `balanced`, and high/unknown risk or
number/citation/time-sensitive/calculation claims use `audit`. Pass
`--runtime-profile-selector-policy` to make those routing thresholds and
sensitive claim metadata keys explicit in the trace. The
`--pre-generation-profile auto` path runs earlier, before diagnostics or claim
verification, using prompt features and optional `--pre-generation-metadata`;
its assessment is recorded in `metadata.pre_generation_risk_assessment`, and
`metadata.runtime_profile_source` shows whether the effective runtime profile
came from pre-generation routing, post-diagnostic auto routing, or an explicit
profile. When no explicit or pre-generation profile is selected, a
`ProductPromotionContract` with promoted release-efficiency evidence supplies
the default runtime profile; the trace records this as
`metadata.runtime_profile_source="promotion_contract_release_efficiency"`.
If the same contract carries feedback-policy evidence, the demo also passes its
validated `control_policy_config` into `RiskController` and records
`metadata.control_policy_source` plus `metadata.effective_control_policy_config`.

By default, `ProductTrace.runtime_trace` also records request phase timings for
diagnostics, verification, action planning/execution, retrieval evidence
collection, and re-verification. Pass `--no-runtime-trace` when comparing JSON
payloads that should omit timing noise. Pass `--compact-json` when trace output
is consumed by automated artifact workflows and compact size matters more than
manual diff readability. Pass `--bounded-trace` for online telemetry payloads
that keep routing summaries and artifact refs while truncating long claims,
verification results, events, and action outputs; use full traces for replay,
corpus, and runtime-baseline artifacts because those tools reject bounded
telemetry inputs. Optional runtime budget flags evaluate those timings
fail-closed and write the result into trace metadata. The demo also summarizes
verifier route cost metadata, can wrap verifier/retriever calls in request-local
caches, and can gate on route attempts, route-budget exhaustion, retrieval use,
and cache hit rates:

```bash
python examples/calibrated_control_demo.py --runtime-profile balanced
python examples/calibrated_control_demo.py --bounded-trace --compact-json
python examples/calibrated_control_demo.py --runtime-profile audit
python examples/calibrated_control_demo.py \
  --max-runtime-total-seconds 1.0 \
  --max-runtime-phase-seconds '{"initial_verification": 0.5}'
python examples/calibrated_control_demo.py \
  --max-runtime-phase-p95-seconds '{"initial_verification": 0.5}' \
  --max-runtime-phase-p99-seconds '{"initial_verification": 0.8}'
python examples/calibrated_control_demo.py \
  --max-mean-attempted-route-count 1.5 \
  --max-route-budget-exhaustion-rate 0.0 \
  --max-retrieval-use-rate 0.5
python examples/calibrated_control_demo.py \
  --promotion-contract artifacts/smollm2_product_promotion_contract_v1_5/product-promotion-contract.json
python examples/calibrated_control_demo.py \
  --promotion-contract artifacts/smollm2_product_promotion_contract_v1_5/product-promotion-contract.json \
  --promotion-contract-registry artifacts/local-release-registry.json \
  --verify-promotion-contract-manifest
python examples/calibrated_control_demo.py \
  --cache-verifier \
  --min-cache-hit-rate 0.5
python examples/calibrated_control_demo.py \
  --runtime-profile latency \
  --min-verification-skip-rate 0.9 \
  --max-verified-claim-count 0
```

When the SmolLM2 strict structured-retrieval-audit release candidate is present,
the demo loads its promotion contract by default as route, calibrated control
defaults, adapter-family, and required-audit metadata. The current default is
the compact v1.5 product promotion contract artifact, which also carries
selector replay and product-runtime-drift evidence. Contract control defaults
such as `max_verifier_route_attempts` fill unset demo controls, and any
feedback-derived `ControlPolicyConfig` in the contract becomes the effective
`RiskController` policy, while explicit CLI options still take precedence for
runtime-budget flags. Passing the same file explicitly with
`--promotion-contract` also enforces its runtime budget, including the
low-latency product-route gates `max_retrieval_use_rate=0.0` and
`max_mean_attempted_route_count=1.1`. The demo uses
`ProductRuntimeEvidenceBundle` to infer the sibling artifact manifest and can
optionally attach manifest verification plus local registry provenance with
`--verify-promotion-contract-manifest` and `--promotion-contract-registry`.
Staged-verification runs also emit
`verification_stage_summary`, so low-risk fast-path savings can be gated with
`--min-verification-skip-rate` and `--max-verified-claim-count`. The latency
runtime profile verifies only claims that triggered configured claim metadata or
feature flags, and records skipped claim ids in the trace instead of treating
them as supported.

The demo can also route unsupported claims to the dependency-free in-memory
retrieval executor, feed retrieval hits back into the groundedness verifier, and
register the saved trace in a local JSON registry:

```bash
python examples/calibrated_control_demo.py \
  --text "Paris is the capital of France." \
  --evidence '[]' \
  --diagnostics '{"maha_last":1.0,"subspace_resid":0.1}' \
  --retrieval-evidence '[{"source":"atlas","text":"Paris is the capital of France."}]' \
  --output /tmp/eigentruth_trace.json \
  --registry /tmp/eigentruth_registry.json
```

In this path the initial groundedness result is unsupported, the retrieve action
returns local evidence, and the final trace records the reverified supported
claim plus the final `accept` decision.

It can also use the dependency-free lexical groundedness verifier. Pass evidence
snippets as a JSON list and optional explicit refutations as a JSON object:

```bash
python examples/calibrated_control_demo.py \
  --text "Paris is the capital of France. The moon is made of cheese." \
  --evidence '[{"source": "atlas", "text": "Paris is the capital of France."}, {"source": "nasa", "text": "The moon is not made of cheese; lunar samples are rock."}]'
```

For deterministic arithmetic claims, enable the local calculator verifier. The
demo routes arithmetic-looking claims, extracted calculation metadata, or
structured calculation context to `CalculatorVerifier`, then falls back to the
lexical verifier for ordinary claims:

```bash
python examples/calibrated_control_demo.py \
  --enable-calculator \
  --text "2 plus 2 is 5." \
  --diagnostics '{"truth_proj": 0.0}'
```

This path produces a refuted claim and an `abstain` decision even when internal
diagnostics are low, because the calculator result contradicts the claim.

`sqlite_state_control_demo.py` also avoids model loading and network access. It
shows the structured-state product path using only stdlib SQLite:

```bash
python examples/sqlite_state_control_demo.py \
  --database /tmp/eigentruth_orders.db \
  --output /tmp/eigentruth_sqlite_trace.json
```

The demo seeds two orders. `ord_1` is supported as shippable; `ord_2` is refuted
because the account is suspended and inventory is insufficient. The final trace
therefore records low internal diagnostics, a `structured_state` refutation, and
a dry-run `abstain` action.

`state_transition_control_demo.py` also avoids model loading and network access.
It shows where world-model correction sits in the product loop: predict the
state after an action, then verify claims against postconditions on that
predicted state.

```bash
python examples/state_transition_control_demo.py \
  --output /tmp/eigentruth_state_transition_trace.json
```

The default action reserves three units of SKU 123. The trace supports the claim
that seven units remain, refutes the claim that ten units remain, and abstains
despite low internal diagnostics.

`production_tool_loop_demo.py` also avoids model loading and network access. It
shows the local integration shape for product tools: pre-check database state,
execute a reserve-inventory tool through `ActionExecutorRegistry` and
`PolicyGuardedActionExecutor`, map `ActionResult.output` into verifier state,
and verify post-tool claims with route-summary observability. Pass
`--execution-ledger` to persist successful action results and replay duplicate
requests without running the side-effecting SQLite mutation again; use
`--execution-ledger-backend sqlite` for a durable local SQLite ledger.

```bash
python examples/production_tool_loop_demo.py \
  --database /tmp/eigentruth_tool_loop.db \
  --execution-ledger /tmp/eigentruth_tool_loop_ledger.sqlite \
  --execution-ledger-backend sqlite \
  --output /tmp/eigentruth_tool_loop_trace.json
```

The default tool execution reserves five units and leaves seven available, but
does not capture payment. The trace therefore supports the pre-check and
inventory postcondition, refutes the payment claim, records
`database_state=1` / `tool_output_state=2` in `route_summary`, marks the action
execution summary as side-effecting, records idempotency and timeout audit
metadata, and abstains despite low internal diagnostics.

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
