"""Layer and score sweep calibration utilities."""

from __future__ import annotations

import json
import math
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, MutableMapping, Optional, Sequence

import torch

from eigentruth import __version__
from eigentruth.calibration.artifacts import CalibrationArtifact, CalibrationScore, SteeringPolicyConfig
from eigentruth.eval.metrics import roc_auc
from eigentruth.eval.score_dump import ScoreDump, ScoreDumpLayerScores, load_score_dump_layer_scores
from eigentruth.json_utils import strict_json_dumps

ArrayLike = torch.Tensor | Sequence[float]

DEFAULT_SCORE_DIRECTIONS: dict[str, str] = {
    "maha_last": "higher",
    "maha": "higher",
    "truth_proj": "higher",
    "subspace_resid": "higher",
    "resid_update_norm": "higher",
    "resid_update_profile_area": "higher",
    "resid_update_profile_peak": "higher",
    "resid_update_profile_late_mass": "higher",
    "resid_update_profile_concentration": "higher",
    "prompt_answer_distance": "higher",
    "prompt_answer_cosine_gap": "higher",
    "answer_anchor_distance": "higher",
    "answer_path_length": "higher",
    "pathway_disagreement": "higher",
    "attn_prompt_flow_loss": "higher",
    "attn_answer_self_flow": "higher",
    "attn_pathway_gap": "higher",
    "attn_pathway_concentration": "higher",
    "disp_euclid": "higher",
    "disp_hse": "higher",
    "eigenscore": "higher",
    "inside_eigenscore": "higher",
    "inside_semantic_entropy": "higher",
    "inside_embedding_entropy": "higher",
    "inside_semantic_energy": "higher",
    "first_token_entropy": "higher",
    "global_local_uncertainty": "higher",
    "glu": "higher",
    "nll_answer": "higher",
    "answer_char_length": "higher",
    "answer_token_count": "higher",
    "claim_char_length": "higher",
    "claim_token_count": "higher",
    "question_answer_token_overlap": "lower",
    "answer_negation_flag": "higher",
    "answer_number_count": "higher",
    "verifier_not_supported": "higher",
    "verifier_refuted": "higher",
    "verifier_insufficient": "higher",
    "verifier_refute_confidence": "higher",
    "verifier_uncertainty": "higher",
    "verifier_no_retrieval_hit": "higher",
    "selfcheck_support_rate": "lower",
    "selfcheck_refute_rate": "higher",
    "selfcheck_disagreement": "higher",
    "selfcheck_insufficient": "higher",
    "selfcheck_not_applicable": "higher",
    "selfcheck_sample_count": "lower",
    "selfcheck_best_overlap": "lower",
    "fact_selfcheck_support_rate": "lower",
    "fact_selfcheck_refute_rate": "higher",
    "fact_selfcheck_disagreement": "higher",
    "fact_selfcheck_insufficient": "higher",
    "fact_selfcheck_not_applicable": "higher",
    "fact_selfcheck_uncovered_rate": "higher",
    "evidence_alignment_failed": "higher",
    "evidence_alignment_insufficient": "higher",
    "evidence_alignment_keyword_gap": "higher",
    "evidence_alignment_number_gap": "higher",
    "evidence_alignment_entity_gap": "higher",
    "evidence_alignment_citation_gap": "higher",
    "evidence_alignment_issue_rate": "higher",
    "perturbation_conflict_rate": "higher",
    "perturbation_high_confidence_conflict_rate": "higher",
    "perturbation_missing_rate": "higher",
    "perturbation_failed": "higher",
    "perturbation_not_applicable": "higher",
    "world_model_disagreement": "higher",
    "world_model_agreement_gap": "higher",
    "world_model_low_agreement": "higher",
    "world_model_conflict": "higher",
    "world_model_conflict_delta": "higher",
    "world_model_trace_gap": "higher",
    "context_sensitivity_flagged_rate": "higher",
    "context_sensitivity_max_shift": "higher",
    "context_sensitivity_mean_shift": "higher",
    "context_sensitivity_max_ratio": "higher",
}


@dataclass(frozen=True)
class SweepScoreResult:
    """Calibration and ranking metrics for one layer/score pair."""

    layer: int
    score_name: str
    direction: str
    threshold: float
    conformal_alpha: float
    auroc: float
    false_alarm: float
    detection: float
    n_true: int
    n_false: int

    def __post_init__(self) -> None:
        if self.direction not in {"higher", "lower"}:
            raise ValueError("direction must be 'higher' or 'lower'.")
        if isinstance(self.threshold, bool):
            raise ValueError("threshold must be numeric and must not be bool.")
        object.__setattr__(self, "threshold", float(self.threshold))
        if isinstance(self.conformal_alpha, bool):
            raise ValueError("conformal_alpha must be in (0, 1).")
        conformal_alpha = float(self.conformal_alpha)
        if not (0.0 < conformal_alpha < 1.0):
            raise ValueError("conformal_alpha must be in (0, 1).")
        object.__setattr__(self, "conformal_alpha", conformal_alpha)

    def score_config(self) -> CalibrationScore:
        """Return this sweep result as a calibration score config."""
        return CalibrationScore(
            name=self.score_name,
            threshold=self.threshold,
            conformal_alpha=self.conformal_alpha,
            direction=self.direction,
        )

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation."""
        return {
            "layer": self.layer,
            "score_name": self.score_name,
            "direction": self.direction,
            "threshold": self.threshold,
            "conformal_alpha": self.conformal_alpha,
            "auroc": self.auroc,
            "false_alarm": self.false_alarm,
            "detection": self.detection,
            "n_true": self.n_true,
            "n_false": self.n_false,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "SweepScoreResult":
        """Build a sweep score result from JSON-like data."""
        return cls(
            layer=int(data["layer"]),
            score_name=str(data["score_name"]),
            direction=str(data.get("direction", "higher")),
            threshold=data["threshold"],
            conformal_alpha=data["conformal_alpha"],
            auroc=float(data["auroc"]),
            false_alarm=float(data["false_alarm"]),
            detection=float(data["detection"]),
            n_true=int(data["n_true"]),
            n_false=int(data["n_false"]),
        )


@dataclass(frozen=True)
class LayerScoreSweepResult:
    """Sweep results for one model layer."""

    layer: int
    scores: tuple[SweepScoreResult, ...]

    def best_score(self, *, best_by: str = "auroc") -> SweepScoreResult:
        """Return the best score result for this layer."""
        if not self.scores:
            raise ValueError("layer sweep result contains no scores.")
        return _best_result(self.scores, best_by=best_by)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation."""
        return {"layer": self.layer, "scores": [score.to_dict() for score in self.scores]}

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "LayerScoreSweepResult":
        """Build a layer sweep result from JSON-like data."""
        return cls(
            layer=int(data["layer"]),
            scores=tuple(SweepScoreResult.from_dict(score) for score in data.get("scores", ())),
        )


@dataclass(frozen=True)
class LayerScoreSweepReport:
    """Versioned report for layer/score calibration sweeps."""

    model_id: str
    conformal_alpha: float
    layers: tuple[LayerScoreSweepResult, ...]
    best_by: str = "auroc"
    eigentruth_version: str = __version__
    model_revision: Optional[str] = None
    scores_path: Optional[str] = None
    created_at: Optional[str] = None
    commit_sha: Optional[str] = None
    metadata: Mapping[str, Any] = field(default_factory=dict)
    schema_version: int = 1

    def __post_init__(self) -> None:
        if not (0.0 < self.conformal_alpha < 1.0):
            raise ValueError("conformal_alpha must be in (0, 1).")
        if self.best_by not in {"auroc", "detection"}:
            raise ValueError("best_by must be 'auroc' or 'detection'.")

    def score_results(self) -> tuple[SweepScoreResult, ...]:
        """Return all layer/score results in report order."""
        return tuple(score for layer in self.layers for score in layer.scores)

    def best_score(self) -> SweepScoreResult:
        """Return the best layer/score result according to ``best_by``."""
        results = self.score_results()
        if not results:
            raise ValueError("sweep report contains no score results.")
        return _best_result(results, best_by=self.best_by)

    def best_artifact(
        self,
        *,
        steering_policy: Optional[SteeringPolicyConfig] = None,
        warmup_dataset_metadata: Optional[Mapping[str, Any]] = None,
        calibration_dataset_metadata: Optional[Mapping[str, Any]] = None,
    ) -> CalibrationArtifact:
        """Build a single-score calibration artifact from the best sweep result."""
        best = self.best_score()
        metadata = {
            "source": "LayerScoreSweepReport",
            "best_by": self.best_by,
            "scores_path": self.scores_path,
            "n_true": best.n_true,
            "n_false": best.n_false,
        }
        if calibration_dataset_metadata:
            metadata.update(calibration_dataset_metadata)
        return CalibrationArtifact(
            model_id=self.model_id,
            model_revision=self.model_revision,
            target_layer=best.layer,
            scores=(best.score_config(),),
            eigentruth_version=self.eigentruth_version,
            steering_policy=steering_policy or SteeringPolicyConfig(),
            warmup_dataset_metadata=warmup_dataset_metadata or {},
            calibration_dataset_metadata=metadata,
            created_at=self.created_at,
            commit_sha=self.commit_sha,
        )

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation."""
        best = self.best_score() if self.score_results() else None
        return {
            "schema_version": self.schema_version,
            "model_id": self.model_id,
            "model_revision": self.model_revision,
            "conformal_alpha": self.conformal_alpha,
            "best_by": self.best_by,
            "best": None if best is None else best.to_dict(),
            "layers": [layer.to_dict() for layer in self.layers],
            "eigentruth_version": self.eigentruth_version,
            "scores_path": self.scores_path,
            "created_at": self.created_at,
            "commit_sha": self.commit_sha,
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "LayerScoreSweepReport":
        """Build a sweep report from JSON-like data."""
        return cls(
            model_id=str(data["model_id"]),
            model_revision=None if data.get("model_revision") is None else str(data["model_revision"]),
            conformal_alpha=float(data["conformal_alpha"]),
            layers=tuple(LayerScoreSweepResult.from_dict(layer) for layer in data.get("layers", ())),
            best_by=str(data.get("best_by", "auroc")),
            eigentruth_version=str(data.get("eigentruth_version", __version__)),
            scores_path=None if data.get("scores_path") is None else str(data["scores_path"]),
            created_at=None if data.get("created_at") is None else str(data["created_at"]),
            commit_sha=None if data.get("commit_sha") is None else str(data["commit_sha"]),
            metadata=dict(data.get("metadata", {})),
            schema_version=int(data.get("schema_version", 1)),
        )

    def save_json(self, path: str | Path) -> None:
        """Save the sweep report as UTF-8 JSON."""
        Path(path).write_text(strict_json_dumps(self.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8")

    @classmethod
    def load_json(cls, path: str | Path) -> "LayerScoreSweepReport":
        """Load a sweep report from UTF-8 JSON."""
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


@dataclass(frozen=True)
class LayerScoreSweepCalibrator:
    """Build calibration reports across layers and diagnostic scores."""

    alpha: float = 0.1
    best_by: str = "auroc"
    max_workers: int = 1

    def __post_init__(self) -> None:
        if not (0.0 < self.alpha < 1.0):
            raise ValueError("alpha must be in (0, 1).")
        if self.best_by not in {"auroc", "detection"}:
            raise ValueError("best_by must be 'auroc' or 'detection'.")
        if (
            isinstance(self.max_workers, bool)
            or not isinstance(self.max_workers, int)
            or self.max_workers < 1
        ):
            raise ValueError("max_workers must be a positive integer.")

    def calibrate_from_file(
        self,
        path: str | Path,
        *,
        signals: Optional[Sequence[str]] = None,
        directions: Optional[Mapping[str, str]] = None,
        model_id: Optional[str] = None,
        model_revision: Optional[str] = None,
        created_at: Optional[str] = None,
        commit_sha: Optional[str] = None,
        eigentruth_version: str = __version__,
        metadata: Optional[Mapping[str, Any]] = None,
        cache: MutableMapping[str, Any] | None = None,
    ) -> LayerScoreSweepReport:
        """Load a score dump and build a layer/score sweep report."""
        dump_path = Path(path)
        layer_dump = load_score_dump_layer_scores(dump_path, signals=signals, cache=cache)
        return self.calibrate_from_layer_scores(
            layer_dump,
            signals=signals,
            directions=directions,
            model_id=model_id,
            model_revision=model_revision,
            scores_path=str(dump_path),
            created_at=created_at,
            commit_sha=commit_sha,
            eigentruth_version=eigentruth_version,
            metadata=metadata,
        )

    def calibrate_from_layer_scores(
        self,
        layer_scores: ScoreDumpLayerScores,
        *,
        signals: Optional[Sequence[str]] = None,
        directions: Optional[Mapping[str, str]] = None,
        model_id: Optional[str] = None,
        model_revision: Optional[str] = None,
        scores_path: Optional[str] = None,
        created_at: Optional[str] = None,
        commit_sha: Optional[str] = None,
        eigentruth_version: str = __version__,
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> LayerScoreSweepReport:
        """Build a layer/score sweep report from preloaded layer-score columns."""
        return self._calibrate_layer_scores(
            labels=torch.as_tensor(layer_scores.labels, dtype=torch.float64),
            config=dict(layer_scores.config),
            layer_scores=layer_scores.layer_scores,
            signals=signals,
            directions=directions,
            model_id=model_id,
            model_revision=model_revision,
            scores_path=scores_path,
            created_at=created_at,
            commit_sha=commit_sha,
            eigentruth_version=eigentruth_version,
            metadata=metadata,
        )

    def calibrate_from_score_dump(
        self,
        score_dump: ScoreDump,
        *,
        signals: Optional[Sequence[str]] = None,
        directions: Optional[Mapping[str, str]] = None,
        model_id: Optional[str] = None,
        model_revision: Optional[str] = None,
        scores_path: Optional[str] = None,
        created_at: Optional[str] = None,
        commit_sha: Optional[str] = None,
        eigentruth_version: str = __version__,
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> LayerScoreSweepReport:
        """Build a layer/score sweep report from a validated ``ScoreDump``."""
        labels = torch.as_tensor(score_dump.labels, dtype=torch.float64)
        config = dict(score_dump.config)
        layer_scores = _collect_layer_scores_from_score_dump(score_dump)
        return self._calibrate_layer_scores(
            labels=labels,
            config=config,
            layer_scores=layer_scores,
            signals=signals,
            directions=directions,
            model_id=model_id,
            model_revision=model_revision,
            scores_path=scores_path,
            created_at=created_at,
            commit_sha=commit_sha,
            eigentruth_version=eigentruth_version,
            metadata=metadata,
        )

    def calibrate_from_dump(
        self,
        dump: Mapping[str, Any],
        *,
        signals: Optional[Sequence[str]] = None,
        directions: Optional[Mapping[str, str]] = None,
        model_id: Optional[str] = None,
        model_revision: Optional[str] = None,
        scores_path: Optional[str] = None,
        created_at: Optional[str] = None,
        commit_sha: Optional[str] = None,
        eigentruth_version: str = __version__,
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> LayerScoreSweepReport:
        """Build a layer/score sweep report from an ``eval_truthfulqa`` score dump."""
        score_dump = ScoreDump.from_mapping(dump)
        labels = torch.as_tensor(score_dump.labels, dtype=torch.float64)
        config = dict(score_dump.config)
        layer_scores = _collect_layer_scores_from_score_dump(score_dump)
        return self._calibrate_layer_scores(
            labels=labels,
            config=config,
            layer_scores=layer_scores,
            signals=signals,
            directions=directions,
            model_id=model_id,
            model_revision=model_revision,
            scores_path=scores_path,
            created_at=created_at,
            commit_sha=commit_sha,
            eigentruth_version=eigentruth_version,
            metadata=metadata,
        )

    def _calibrate_layer_scores(
        self,
        *,
        labels: torch.Tensor,
        config: Mapping[str, Any],
        layer_scores: Mapping[int, Mapping[str, Sequence[float]]],
        signals: Optional[Sequence[str]],
        directions: Optional[Mapping[str, str]],
        model_id: Optional[str],
        model_revision: Optional[str],
        scores_path: Optional[str],
        created_at: Optional[str],
        commit_sha: Optional[str],
        eigentruth_version: str,
        metadata: Optional[Mapping[str, Any]],
    ) -> LayerScoreSweepReport:
        """Build a layer/score sweep report from validated score families."""
        selected = set(signals) if signals is not None else _all_score_names(layer_scores)
        labels_t, true_mask, false_mask = _prepare_labels(labels)
        jobs: list[_SweepScoreJob] = []
        for layer in sorted(layer_scores):
            for score_name in sorted(layer_scores[layer]):
                if score_name not in selected:
                    continue
                direction = _score_direction(score_name, directions)
                scores_t = _prepare_sweep_score_tensor(layer_scores[layer][score_name], score_name=score_name)
                jobs.append(
                    _SweepScoreJob(
                        layer=layer,
                        score_name=score_name,
                        scores=scores_t,
                        labels=labels_t,
                        true_mask=true_mask,
                        false_mask=false_mask,
                        alpha=self.alpha,
                        direction=direction,
                    )
                )

        score_results = _calibrate_score_jobs(jobs, max_workers=int(self.max_workers))
        if not score_results:
            raise ValueError("no matching layer/score results were found in the score dump.")

        results_by_layer: dict[int, list[SweepScoreResult]] = {}
        for result in score_results:
            results_by_layer.setdefault(result.layer, []).append(result)
        results = [
            LayerScoreSweepResult(layer=layer, scores=tuple(results_by_layer[layer]))
            for layer in sorted(results_by_layer)
        ]
        report_metadata = (
            dict(metadata)
            if metadata is not None
            else {"source": "eval_truthfulqa.py", "config": config}
        )
        report_metadata["sweep_max_workers"] = int(self.max_workers)
        return LayerScoreSweepReport(
            model_id=model_id or str(config.get("model", "unknown")),
            model_revision=model_revision,
            conformal_alpha=self.alpha,
            layers=tuple(results),
            best_by=self.best_by,
            eigentruth_version=eigentruth_version,
            scores_path=scores_path,
            created_at=created_at or datetime.now(timezone.utc).isoformat(),
            commit_sha=commit_sha,
            metadata=report_metadata,
        )


@dataclass(frozen=True)
class _SweepScoreJob:
    layer: int
    score_name: str
    scores: torch.Tensor
    labels: torch.Tensor
    true_mask: torch.Tensor
    false_mask: torch.Tensor
    alpha: float
    direction: str


def _collect_layer_scores_from_score_dump(score_dump: ScoreDump) -> dict[int, dict[str, Sequence[float]]]:
    primary_layer = int(score_dump.config.get("layer", 0))
    layer_scores: dict[int, dict[str, Sequence[float]]] = {primary_layer: dict(score_dump.scores)}
    for layer_key, scores in score_dump.sweep_scores.items():
        layer = int(layer_key)
        layer_scores.setdefault(layer, {}).update(dict(scores))
    return layer_scores


def _collect_layer_scores(dump: Mapping[str, Any]) -> dict[int, dict[str, Sequence[float]]]:
    config = dict(dump.get("config", {}))
    primary_layer = int(config.get("layer", 0))
    layer_scores: dict[int, dict[str, Sequence[float]]] = {primary_layer: dict(dump.get("scores", {}))}
    for layer_key, scores in dict(dump.get("sweep_scores", {})).items():
        layer = int(layer_key)
        layer_scores.setdefault(layer, {}).update(dict(scores))
    return layer_scores


def _all_score_names(layer_scores: Mapping[int, Mapping[str, Sequence[float]]]) -> set[str]:
    return {score_name for scores in layer_scores.values() for score_name in scores}


def _score_direction(score_name: str, directions: Optional[Mapping[str, str]]) -> str:
    direction = (directions or {}).get(score_name, DEFAULT_SCORE_DIRECTIONS.get(score_name, "higher"))
    if direction not in {"higher", "lower"}:
        raise ValueError("directions values must be 'higher' or 'lower'.")
    return direction


def _calibrate_score(
    *,
    layer: int,
    score_name: str,
    scores: ArrayLike,
    labels: torch.Tensor,
    alpha: float,
    direction: str,
) -> SweepScoreResult:
    labels_t, true_mask, false_mask = _prepare_labels(labels)
    scores_t = _prepare_sweep_score_tensor(scores, score_name=score_name)
    return _calibrate_prepared_score(
        layer=layer,
        score_name=score_name,
        scores=scores_t,
        labels=labels_t,
        true_mask=true_mask,
        false_mask=false_mask,
        alpha=alpha,
        direction=direction,
    )


def _calibrate_prepared_score(
    *,
    layer: int,
    score_name: str,
    scores: torch.Tensor,
    labels: torch.Tensor,
    true_mask: torch.Tensor,
    false_mask: torch.Tensor,
    alpha: float,
    direction: str,
) -> SweepScoreResult:
    if scores.numel() != labels.numel():
        raise ValueError(f"score '{score_name}' has {scores.numel()} values but labels has {labels.numel()}.")
    true_scores = scores[true_mask]
    false_scores = scores[false_mask]
    if true_scores.numel() == 0 or false_scores.numel() == 0:
        raise ValueError("sweep calibration requires at least one true and one false labeled score.")

    threshold = _directional_conformal_threshold_from_tensor(true_scores, alpha, direction)
    anomaly_scores = _anomaly_scores(scores, direction)
    false_alarm = _directional_trigger_rate_from_tensor(true_scores, threshold, direction)
    detection = _directional_trigger_rate_from_tensor(false_scores, threshold, direction)
    return SweepScoreResult(
        layer=layer,
        score_name=score_name,
        direction=direction,
        threshold=threshold,
        conformal_alpha=alpha,
        auroc=roc_auc(anomaly_scores, labels),
        false_alarm=false_alarm,
        detection=detection,
        n_true=int(true_scores.numel()),
        n_false=int(false_scores.numel()),
    )


def _calibrate_score_jobs(
    jobs: Sequence[_SweepScoreJob],
    *,
    max_workers: int,
) -> tuple[SweepScoreResult, ...]:
    if not jobs:
        return ()
    if max_workers <= 1 or len(jobs) == 1:
        return tuple(_calibrate_score_job(job) for job in jobs)
    with ThreadPoolExecutor(max_workers=min(max_workers, len(jobs))) as executor:
        return tuple(executor.map(_calibrate_score_job, jobs))


def _calibrate_score_job(job: _SweepScoreJob) -> SweepScoreResult:
    return _calibrate_prepared_score(
        layer=job.layer,
        score_name=job.score_name,
        scores=job.scores,
        labels=job.labels,
        true_mask=job.true_mask,
        false_mask=job.false_mask,
        alpha=job.alpha,
        direction=job.direction,
    )


def _prepare_labels(labels: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    raw_labels = torch.as_tensor(labels, dtype=torch.float64).flatten()
    if not torch.isfinite(raw_labels).all():
        raise ValueError("labels must contain only finite values.")
    if not torch.logical_or(raw_labels == 0, raw_labels == 1).all():
        raise ValueError("labels must be binary values in {0, 1}.")
    labels_t = raw_labels.to(dtype=torch.int64)
    true_mask = labels_t == 0
    false_mask = labels_t == 1
    return labels_t, true_mask, false_mask


def _prepare_sweep_score_tensor(scores: ArrayLike, *, score_name: str) -> torch.Tensor:
    scores_t = torch.as_tensor(scores, dtype=torch.float64).flatten()
    if not torch.isfinite(scores_t).all():
        raise ValueError(f"score '{score_name}' must contain only finite values.")
    return scores_t


def _directional_conformal_threshold_from_tensor(
    true_scores: torch.Tensor,
    alpha: float,
    direction: str,
) -> float:
    if direction not in {"higher", "lower"}:
        raise ValueError("direction must be 'higher' or 'lower'.")
    if not (0.0 < alpha < 1.0):
        raise ValueError(f"alpha must be in (0, 1), got {alpha}.")
    n = true_scores.numel()
    if n == 0:
        raise ValueError("calibration scores must be non-empty.")
    rank = math.ceil((n + 1) * (1.0 - alpha))
    if rank > n:
        threshold = float("inf")
    else:
        oriented = true_scores if direction == "higher" else -true_scores
        threshold = float(torch.kthvalue(oriented, rank).values.item())
    return threshold if direction == "higher" else -threshold


def _directional_trigger_rate_from_tensor(
    scores: torch.Tensor,
    threshold: float,
    direction: str,
) -> float:
    if direction not in {"higher", "lower"}:
        raise ValueError("direction must be 'higher' or 'lower'.")
    if math.isnan(float(threshold)):
        raise ValueError("threshold must not be NaN.")
    if scores.numel() == 0:
        return 0.0
    if direction == "higher":
        return float((scores > threshold).double().mean().item())
    return float((scores < threshold).double().mean().item())


def _anomaly_scores(scores: torch.Tensor, direction: str) -> torch.Tensor:
    return scores if direction == "higher" else -scores



def _best_result(results: Sequence[SweepScoreResult], *, best_by: str) -> SweepScoreResult:
    if best_by == "auroc":
        return max(results, key=lambda result: _rankable(result.auroc))
    if best_by == "detection":
        return max(results, key=lambda result: (_rankable(result.detection), _rankable(result.auroc)))
    raise ValueError("best_by must be 'auroc' or 'detection'.")


def _rankable(value: float) -> float:
    return float("-inf") if math.isnan(value) else value
