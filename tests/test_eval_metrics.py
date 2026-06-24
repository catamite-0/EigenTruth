"""单元测试 — eigentruth.eval.metrics

纯函数测试，CPU 可运行，无需模型或网络。
"""

import json
import math

import pytest
import torch

from eigentruth.eval.metrics import (
    binomial_confidence_interval,
    euclidean_dispersion,
    roc_auc,
    selective_classification_report,
)
from eigentruth.eval.score_dump import (
    ScoreDump,
    iter_score_dump_jsonl_records,
    load_score_dump,
    load_score_dump_columns,
    load_score_dump_layer_scores,
    load_score_dump_statement_scores,
    score_dump_cache_summary,
    score_dump_file_metadata,
    write_score_dump_jsonl,
)


class TestRocAuc:
    """AUROC 计算测试（含已知值、平局、缺类）。"""

    def test_perfect_separation(self):
        """正类分数全高于负类 → AUROC = 1.0。"""
        scores = [0.1, 0.2, 0.3, 0.9, 1.0, 1.1]
        labels = [0, 0, 0, 1, 1, 1]
        assert roc_auc(scores, labels) == 1.0

    def test_perfect_inversion(self):
        """正类分数全低于负类 → AUROC = 0.0。"""
        scores = [0.9, 1.0, 1.1, 0.1, 0.2, 0.3]
        labels = [0, 0, 0, 1, 1, 1]
        assert roc_auc(scores, labels) == 0.0

    def test_known_partial_value(self):
        """已知部分分离值：neg={0.0,0.1}, pos={0.2,0.05} → 3/4 = 0.75。"""
        scores = [0.0, 0.1, 0.2, 0.05]
        labels = [0, 0, 1, 1]
        assert roc_auc(scores, labels) == pytest.approx(0.75)

    def test_chance_value(self):
        """交错分数 → AUROC = 0.5。neg={0.0,0.3}, pos={0.1,0.2}: 2/4。"""
        scores = [0.0, 0.3, 0.1, 0.2]
        labels = [0, 0, 1, 1]
        assert roc_auc(scores, labels) == pytest.approx(0.5)

    def test_all_ties_is_half(self):
        """全部分数相同 → 平均排名 → AUROC = 0.5。"""
        scores = [0.5, 0.5, 0.5, 0.5]
        labels = [0, 1, 0, 1]
        assert roc_auc(scores, labels) == pytest.approx(0.5)

    def test_absent_class_returns_nan(self):
        """缺少某一类时返回 NaN。"""
        assert math.isnan(roc_auc([1.0, 2.0, 3.0], [0, 0, 0]))
        assert math.isnan(roc_auc([1.0, 2.0, 3.0], [1, 1, 1]))

    def test_length_mismatch_raises(self):
        with pytest.raises(ValueError, match="same length"):
            roc_auc([1.0, 2.0], [1])

    def test_accepts_tensors(self):
        """接受张量输入。"""
        scores = torch.tensor([0.1, 0.9, 0.2, 0.8])
        labels = torch.tensor([0, 1, 0, 1])
        assert roc_auc(scores, labels) == pytest.approx(1.0)


class TestEuclideanDispersion:
    """欧氏离散度测试。"""

    def test_identical_points_zero(self):
        pts = torch.ones(8, 16) * 2.0
        assert euclidean_dispersion(pts).item() == pytest.approx(0.0, abs=1e-6)

    def test_single_point_zero(self):
        assert euclidean_dispersion(torch.randn(1, 16)).item() == 0.0

    def test_spread_larger_than_tight(self):
        torch.manual_seed(0)
        tight = torch.randn(20, 16) * 0.01
        spread = torch.randn(20, 16) * 5.0
        assert euclidean_dispersion(spread) > euclidean_dispersion(tight)

    def test_non_negative(self):
        torch.manual_seed(1)
        assert euclidean_dispersion(torch.randn(10, 32)) >= 0.0


class TestSelectiveClassificationReport:
    """Selective routing report tests."""

    def test_reports_coverage_accuracy_and_confidence_intervals(self):
        scores = [0.1, 0.2, 0.9, 1.1]
        labels = [0, 0, 1, 1]

        report = selective_classification_report(scores, labels, threshold=0.5)

        assert report["n_flagged"] == 2
        assert report["n_accepted"] == 2
        assert report["coverage"] == pytest.approx(0.5)
        assert report["selective_accuracy"] == pytest.approx(1.0)
        assert report["detection"] == pytest.approx(1.0)
        assert report["false_alarm"] == pytest.approx(0.0)
        assert report["coverage_ci"]["total"] == 4
        assert report["selective_accuracy_ci"]["successes"] == 2

    def test_lower_direction_flags_low_scores(self):
        report = selective_classification_report([0.1, 0.9], [1, 0], threshold=0.5, direction="lower")

        assert report["n_flagged"] == 1
        assert report["detection"] == pytest.approx(1.0)

    def test_empty_denominator_ci_serializes_as_none(self):
        ci = binomial_confidence_interval(0, 0)

        assert ci == {"estimate": None, "lower": None, "upper": None, "successes": 0, "total": 0}


class TestScoreDump:
    """Validated score dump loader tests."""

    def test_summary_roundtrip_and_file_metadata(self, tmp_path):
        path = tmp_path / "scores.json"
        path.write_text(
            json.dumps({
                "config": {"model": "unit-model", "layer": -2},
                "labels": [0, 1],
                "scores": {"maha_last": [0.1, 0.9]},
                "sweep_scores": {"-2": {"truth_proj": [0.3, 0.8]}},
                "statements": [{"text": "true"}, {"text": "false"}],
                "inside_sampling": {"mode": "off"},
            }),
            encoding="utf-8",
        )

        dump = load_score_dump(path, required_scores=("maha_last",))
        metadata = score_dump_file_metadata(path, dump)

        assert isinstance(dump, ScoreDump)
        assert dump.summary()["n_total"] == 2
        assert dump.summary()["n_true"] == 1
        assert dump.summary()["n_false"] == 1
        assert dump.summary()["model"] == "unit-model"
        assert dump.summary()["sweep_score_count"] == 1
        assert dump.summary()["sweep_score_names"] == ("truth_proj",)
        assert dump.signal_names() == ("maha_last", "truth_proj")
        assert dump.to_mapping()["inside_sampling"] == {"mode": "off"}
        assert metadata["exists"] is True
        assert metadata["sha256"]
        assert metadata["summary"]["all_signal_names"] == ("maha_last", "truth_proj")

    def test_validates_lengths_and_required_scores(self, tmp_path):
        path = tmp_path / "bad-scores.json"
        path.write_text(
            json.dumps({
                "labels": [0, 1],
                "scores": {"maha_last": [0.1]},
            }),
            encoding="utf-8",
        )

        with pytest.raises(ValueError, match="length does not match labels"):
            load_score_dump(path)

        missing_path = tmp_path / "missing-score.json"
        missing_path.write_text(
            json.dumps({
                "labels": [0, 1],
                "scores": {"maha_last": [0.1, 0.9]},
            }),
            encoding="utf-8",
        )
        with pytest.raises(ValueError, match="missing requested score"):
            load_score_dump(missing_path, required_scores=("truth_proj",))

    def test_can_validate_statement_only_dump_when_explicitly_allowed(self, tmp_path):
        path = tmp_path / "statement-dump.json"
        path.write_text(
            json.dumps({
                "labels": [0, 1],
                "statements": [{"text": "true"}, {"text": "false"}],
            }),
            encoding="utf-8",
        )

        with pytest.raises(ValueError, match="scores"):
            load_score_dump(path, require_statements=True)

        dump = load_score_dump(path, allow_missing_scores=True, require_statements=True)

        assert dump.summary()["score_count"] == 0
        assert dump.summary()["statement_count"] == 2

    def test_file_metadata_cache_reuses_fingerprint(self, tmp_path, monkeypatch):
        path = tmp_path / "scores.json"
        path.write_text(
            json.dumps({
                "labels": [0, 1],
                "scores": {"maha_last": [0.1, 0.9]},
            }),
            encoding="utf-8",
        )
        calls = []

        def fake_sha256_file(sha_path):
            calls.append(sha_path)
            return "cached-sha"

        monkeypatch.setattr("eigentruth.eval.score_dump._sha256_file", fake_sha256_file)
        dump = load_score_dump(path)
        cache = {}

        first = score_dump_file_metadata(path, dump, cache=cache)
        second = score_dump_file_metadata(path, dump, cache=cache)

        assert first["sha256"] == "cached-sha"
        assert second["sha256"] == "cached-sha"
        assert first["summary"] == second["summary"]
        assert len(calls) == 1

    def test_jsonl_manifest_roundtrip_and_streaming_records(self, tmp_path):
        dump = ScoreDump.from_mapping({
            "config": {"model": "unit-model", "layer": -2},
            "labels": [0, 1],
            "scores": {"maha_last": [0.1, 0.9]},
            "sweep_scores": {"-2": {"truth_proj": [0.3, 0.8]}},
            "statements": [{"text": "true"}, {"text": "false"}],
            "inside_sampling": {"mode": "off"},
            "batch_indexes": [0, 1],
            "inside_sample_texts": [["true sample"], ["false sample"]],
        })
        manifest_path = tmp_path / "scores.manifest.json"

        manifest = write_score_dump_jsonl(
            dump,
            manifest_path,
            record_extra_names=("batch_indexes", "inside_sample_texts"),
        )
        records = tuple(iter_score_dump_jsonl_records(manifest_path))
        loaded = load_score_dump(manifest_path, required_scores=("maha_last",))
        metadata = score_dump_file_metadata(manifest_path)

        assert manifest.records_path == "scores.manifest.records.jsonl"
        assert manifest.extras == {"inside_sampling": {"mode": "off"}}
        assert records[0].label == 0
        assert records[0].scores["maha_last"] == pytest.approx(0.1)
        assert records[0].extras["batch_indexes"] == 0
        assert records[0].extras["inside_sample_texts"] == ["true sample"]
        assert records[1].sweep_scores["-2"]["truth_proj"] == pytest.approx(0.8)
        assert loaded.summary() == dump.summary()
        assert loaded.to_mapping()["inside_sampling"] == {"mode": "off"}
        assert loaded.to_mapping()["batch_indexes"] == [0, 1]
        assert loaded.to_mapping()["inside_sample_texts"] == [["true sample"], ["false sample"]]
        assert loaded.statements == dump.statements
        assert metadata["source_format"] == "eigentruth.score_dump.jsonl"
        assert metadata["records"]["path"].endswith("scores.manifest.records.jsonl")
        assert metadata["records"]["sha256"]

    def test_jsonl_writer_rejects_bad_record_extra_length(self, tmp_path):
        dump = ScoreDump.from_mapping({
            "labels": [0, 1],
            "scores": {"maha_last": [0.1, 0.9]},
            "batch_indexes": [0],
        })
        manifest_path = tmp_path / "scores.manifest.json"

        with pytest.raises(ValueError, match="record extra 'batch_indexes' length"):
            write_score_dump_jsonl(dump, manifest_path, record_extra_names=("batch_indexes",))

    def test_jsonl_manifest_validates_record_count_and_schema(self, tmp_path):
        manifest_path = tmp_path / "scores.manifest.json"
        records_path = tmp_path / "scores.records.jsonl"
        manifest_path.write_text(
            json.dumps({
                "schema_version": 1,
                "format": "eigentruth.score_dump.jsonl",
                "records_path": records_path.name,
                "config": {"model": "unit"},
                "score_names": ["maha_last"],
                "sweep_scores": {},
                "n_total": 2,
            }),
            encoding="utf-8",
        )
        records_path.write_text(
            json.dumps({"label": 0, "scores": {"maha_last": 0.1}}) + "\n",
            encoding="utf-8",
        )

        with pytest.raises(ValueError, match="record count"):
            tuple(iter_score_dump_jsonl_records(manifest_path))

        manifest_path.write_text(
            json.dumps({
                "schema_version": 1,
                "format": "eigentruth.score_dump.jsonl",
                "records_path": records_path.name,
                "config": {"model": "unit"},
                "score_names": ["maha_last"],
                "sweep_scores": {},
                "n_total": 1,
            }),
            encoding="utf-8",
        )
        records_path.write_text(
            json.dumps({"label": 0, "scores": {"other": 0.1}}) + "\n",
            encoding="utf-8",
        )

        with pytest.raises(ValueError, match="missing score"):
            load_score_dump(manifest_path)

    def test_load_score_dump_columns_reads_selected_jsonl_scores(self, tmp_path, monkeypatch):
        dump = ScoreDump.from_mapping({
            "labels": [0, 1],
            "scores": {
                "maha_last": [0.1, 0.9],
                "unused": [99.0, 100.0],
            },
        })
        manifest_path = tmp_path / "scores.manifest.json"
        write_score_dump_jsonl(dump, manifest_path)

        def fail_from_mapping(*args, **kwargs):
            raise AssertionError("JSONL column loading should not materialize ScoreDump")

        monkeypatch.setattr(ScoreDump, "from_mapping", fail_from_mapping)
        columns = load_score_dump_columns(manifest_path, ("maha_last",))

        assert columns.labels == (0, 1)
        assert columns.scores == {"maha_last": (0.1, 0.9)}
        assert columns.summary["score_names"] == ("maha_last", "unused")
        assert columns.source_format == "eigentruth.score_dump.jsonl"

    def test_load_score_dump_columns_ignores_unselected_jsonl_scores(self, tmp_path):
        manifest_path = tmp_path / "scores.manifest.json"
        records_path = tmp_path / "scores.records.jsonl"
        manifest_path.write_text(
            json.dumps({
                "schema_version": 1,
                "format": "eigentruth.score_dump.jsonl",
                "records_path": records_path.name,
                "score_names": ["maha_last", "unused"],
                "sweep_scores": {},
                "n_total": 2,
            }),
            encoding="utf-8",
        )
        records_path.write_text(
            "\n".join([
                json.dumps({"label": 0, "scores": {"maha_last": 0.1, "unused": "bad"}}),
                json.dumps({"label": 1, "scores": {"maha_last": 0.9, "unused": "bad"}}),
            ]) + "\n",
            encoding="utf-8",
        )

        columns = load_score_dump_columns(manifest_path, ("maha_last",))

        assert columns.scores == {"maha_last": (0.1, 0.9)}
        with pytest.raises(ValueError):
            load_score_dump(manifest_path)

    def test_score_dump_file_metadata_summarizes_jsonl_without_score_materialization(self, tmp_path):
        manifest_path = tmp_path / "scores.manifest.json"
        records_path = tmp_path / "scores.records.jsonl"
        manifest_path.write_text(
            json.dumps({
                "schema_version": 1,
                "format": "eigentruth.score_dump.jsonl",
                "records_path": records_path.name,
                "config": {"model": "unit-model", "layer": -1},
                "score_names": ["maha_last", "unused"],
                "sweep_scores": {"-2": ["truth_proj", "unused"]},
                "n_total": 3,
                "has_statements": True,
            }),
            encoding="utf-8",
        )
        records_path.write_text(
            "\n".join([
                json.dumps({
                    "label": 0,
                    "scores": {"maha_last": 0.1, "unused": "bad"},
                    "sweep_scores": {"-2": {"truth_proj": 0.2, "unused": "bad"}},
                }),
                json.dumps({
                    "label": 1,
                    "scores": {"maha_last": 0.9, "unused": "bad"},
                    "sweep_scores": {"-2": {"truth_proj": 0.8, "unused": "bad"}},
                }),
                json.dumps({
                    "label": 0,
                    "scores": {"maha_last": 0.3, "unused": "bad"},
                    "sweep_scores": {"-2": {"truth_proj": 0.4, "unused": "bad"}},
                }),
            ]) + "\n",
            encoding="utf-8",
        )

        metadata = score_dump_file_metadata(manifest_path)

        assert metadata["source_format"] == "eigentruth.score_dump.jsonl"
        assert metadata["records"]["exists"] is True
        assert metadata["summary"]["n_total"] == 3
        assert metadata["summary"]["n_true"] == 2
        assert metadata["summary"]["n_false"] == 1
        assert metadata["summary"]["score_names"] == ("maha_last", "unused")
        assert metadata["summary"]["sweep_layers"] == ("-2",)
        assert metadata["summary"]["all_signal_names"] == ("maha_last", "truth_proj", "unused")
        with pytest.raises(ValueError):
            load_score_dump(manifest_path)

    def test_score_dump_file_metadata_cache_reuses_jsonl_summary_scan(self, tmp_path, monkeypatch):
        manifest_path = tmp_path / "scores.manifest.json"
        records_path = tmp_path / "scores.records.jsonl"
        manifest_path.write_text(
            json.dumps({
                "schema_version": 1,
                "format": "eigentruth.score_dump.jsonl",
                "records_path": records_path.name,
                "score_names": ["maha_last"],
                "sweep_scores": {},
                "n_total": 2,
            }),
            encoding="utf-8",
        )
        records_path.write_text(
            "\n".join([
                json.dumps({"label": 0, "scores": {"maha_last": 0.1}}),
                json.dumps({"label": 1, "scores": {"maha_last": 0.9}}),
            ]) + "\n",
            encoding="utf-8",
        )

        from eigentruth.eval import score_dump as score_dump_module

        calls = []
        original_loader = score_dump_module._load_score_dump_jsonl_labels

        def counting_loader(*args, **kwargs):
            calls.append(args[0])
            return original_loader(*args, **kwargs)

        monkeypatch.setattr(score_dump_module, "_load_score_dump_jsonl_labels", counting_loader)
        cache = {}

        first = score_dump_file_metadata(manifest_path, cache=cache)
        second = score_dump_file_metadata(manifest_path, cache=cache)

        assert first["summary"] == second["summary"]
        assert first["summary"]["n_true"] == 1
        assert second["summary"]["n_false"] == 1
        assert len(calls) == 1

    def test_jsonl_selected_view_cache_reuses_record_scans(self, tmp_path, monkeypatch):
        dump = ScoreDump.from_mapping({
            "config": {"model": "unit-model", "layer": -1},
            "labels": [0, 1],
            "scores": {
                "maha_last": [0.1, 0.9],
                "truth_proj": [0.2, 0.8],
            },
            "sweep_scores": {
                "-2": {"truth_proj": [0.3, 0.7]},
            },
            "statements": [
                {"text": "Supported claim."},
                {"text": "Refuted claim."},
            ],
        })
        manifest_path = tmp_path / "scores.manifest.json"
        write_score_dump_jsonl(dump, manifest_path)

        from eigentruth.eval import score_dump as score_dump_module

        selected_calls = []
        statement_calls = []
        original_selected = score_dump_module._iter_score_dump_jsonl_selected_records
        original_statement = score_dump_module._iter_score_dump_jsonl_selected_statement_records

        def counting_selected(*args, **kwargs):
            selected_calls.append(kwargs)
            yield from original_selected(*args, **kwargs)

        def counting_statement(*args, **kwargs):
            statement_calls.append(kwargs)
            yield from original_statement(*args, **kwargs)

        monkeypatch.setattr(
            score_dump_module,
            "_iter_score_dump_jsonl_selected_records",
            counting_selected,
        )
        monkeypatch.setattr(
            score_dump_module,
            "_iter_score_dump_jsonl_selected_statement_records",
            counting_statement,
        )
        cache = {}

        first_columns = load_score_dump_columns(manifest_path, ("maha_last",), cache=cache)
        second_columns = load_score_dump_columns(manifest_path, ("maha_last",), cache=cache)
        first_layers = load_score_dump_layer_scores(manifest_path, signals=("truth_proj",), cache=cache)
        second_layers = load_score_dump_layer_scores(manifest_path, signals=("truth_proj",), cache=cache)
        first_statements = load_score_dump_statement_scores(manifest_path, ("truth_proj",), cache=cache)
        second_statements = load_score_dump_statement_scores(manifest_path, ("truth_proj",), cache=cache)

        def fail_label_loader(*args, **kwargs):
            raise AssertionError("metadata should reuse the selected-view JSONL summary cache")

        monkeypatch.setattr(score_dump_module, "_load_score_dump_jsonl_labels", fail_label_loader)
        metadata = score_dump_file_metadata(manifest_path, cache=cache)
        summary = score_dump_cache_summary(cache)

        assert first_columns is second_columns
        assert first_layers is second_layers
        assert first_statements is second_statements
        assert metadata["summary"] == first_columns.summary
        assert summary["enabled"] is True
        assert summary["jsonl_view"]["hits"] == 3
        assert summary["jsonl_view"]["misses"] == 3
        assert summary["jsonl_view"]["writes"] == 3
        assert summary["jsonl_summary"]["hits"] == 1
        assert summary["jsonl_summary"]["writes"] == 3
        assert summary["fingerprint"]["misses"] == 2
        assert summary["fingerprint"]["writes"] == 2
        assert summary["cache_entries"] >= 5
        assert first_columns.scores == {"maha_last": (0.1, 0.9)}
        assert first_layers.layer_scores == {
            -2: {"truth_proj": (0.3, 0.7)},
            -1: {"truth_proj": (0.2, 0.8)},
        }
        assert first_statements.statements == (
            {"text": "Supported claim."},
            {"text": "Refuted claim."},
        )
        assert len(selected_calls) == 2
        assert len(statement_calls) == 1

    def test_jsonl_selected_view_cache_invalidates_changed_records(self, tmp_path, monkeypatch):
        manifest_path = tmp_path / "scores.manifest.json"
        write_score_dump_jsonl(
            ScoreDump.from_mapping({
                "labels": [0, 1],
                "scores": {"maha_last": [0.1, 0.9]},
            }),
            manifest_path,
        )

        from eigentruth.eval import score_dump as score_dump_module

        calls = []
        original_loader = score_dump_module._iter_score_dump_jsonl_selected_records

        def counting_loader(*args, **kwargs):
            calls.append(kwargs)
            yield from original_loader(*args, **kwargs)

        monkeypatch.setattr(score_dump_module, "_iter_score_dump_jsonl_selected_records", counting_loader)
        cache = {}

        first = load_score_dump_columns(manifest_path, ("maha_last",), cache=cache)
        second = load_score_dump_columns(manifest_path, ("maha_last",), cache=cache)
        write_score_dump_jsonl(
            ScoreDump.from_mapping({
                "labels": [0, 1, 0],
                "scores": {"maha_last": [0.4, 0.8, 0.2]},
            }),
            manifest_path,
        )
        third = load_score_dump_columns(manifest_path, ("maha_last",), cache=cache)
        summary = score_dump_cache_summary(cache)

        assert first is second
        assert first.labels == (0, 1)
        assert third.labels == (0, 1, 0)
        assert third.scores == {"maha_last": (0.4, 0.8, 0.2)}
        assert len(calls) == 2
        assert summary["jsonl_view"]["hits"] == 1
        assert summary["jsonl_view"]["misses"] == 2
        assert summary["jsonl_view"]["writes"] == 2

    def test_score_dump_cache_summary_handles_empty_cache(self):
        assert score_dump_cache_summary(None) == {
            "enabled": False,
            "cache_entries": 0,
            "fingerprint": {"hits": 0, "misses": 0, "writes": 0, "attempts": 0, "hit_rate": None},
            "jsonl_summary": {"hits": 0, "misses": 0, "writes": 0, "attempts": 0, "hit_rate": None},
            "jsonl_view": {"hits": 0, "misses": 0, "writes": 0, "attempts": 0, "hit_rate": None},
        }

    def test_load_score_dump_layer_scores_reads_selected_jsonl_sweep_scores(self, tmp_path, monkeypatch):
        dump = ScoreDump.from_mapping({
            "config": {"model": "unit-model", "layer": -1},
            "labels": [0, 1],
            "scores": {
                "maha_last": [0.1, 0.9],
                "unused": [99.0, 100.0],
            },
            "sweep_scores": {
                "-2": {
                    "truth_proj": [0.2, 0.8],
                    "unused": [11.0, 12.0],
                },
            },
        })
        manifest_path = tmp_path / "scores.manifest.json"
        write_score_dump_jsonl(dump, manifest_path)

        def fail_from_mapping(*args, **kwargs):
            raise AssertionError("JSONL layer-score loading should not materialize ScoreDump")

        monkeypatch.setattr(ScoreDump, "from_mapping", fail_from_mapping)
        layer_dump = load_score_dump_layer_scores(manifest_path, signals=("truth_proj",))

        assert layer_dump.labels == (0, 1)
        assert layer_dump.layer_scores == {-2: {"truth_proj": (0.2, 0.8)}}
        assert layer_dump.score_sources == {-2: {"truth_proj": "sweep_scores"}}
        assert layer_dump.summary["all_signal_names"] == ("maha_last", "truth_proj", "unused")
        assert layer_dump.source_format == "eigentruth.score_dump.jsonl"

    def test_load_score_dump_statement_scores_reads_selected_jsonl_fields(self, tmp_path):
        manifest_path = tmp_path / "scores.manifest.json"
        records_path = tmp_path / "scores.records.jsonl"
        manifest_path.write_text(
            json.dumps({
                "schema_version": 1,
                "format": "eigentruth.score_dump.jsonl",
                "records_path": records_path.name,
                "score_names": ["truth_proj", "unused"],
                "sweep_scores": {"-2": ["unused"]},
                "n_total": 2,
                "has_statements": True,
            }),
            encoding="utf-8",
        )
        records_path.write_text(
            "\n".join([
                json.dumps({
                    "label": 0,
                    "scores": {"truth_proj": 0.1, "unused": "bad"},
                    "sweep_scores": {"-2": {"unused": "bad"}},
                    "statement": {"text": "Supported claim."},
                }),
                json.dumps({
                    "label": 1,
                    "scores": {"truth_proj": 0.9, "unused": "bad"},
                    "sweep_scores": {"-2": {"unused": "bad"}},
                    "statement": {"text": "Refuted claim."},
                }),
            ]) + "\n",
            encoding="utf-8",
        )

        columns = load_score_dump_statement_scores(manifest_path, ("truth_proj",))

        assert columns.labels == (0, 1)
        assert columns.scores == {"truth_proj": (0.1, 0.9)}
        assert columns.statements == ({"text": "Supported claim."}, {"text": "Refuted claim."})
        assert columns.summary["statement_count"] == 2
        with pytest.raises(ValueError):
            load_score_dump(manifest_path)
