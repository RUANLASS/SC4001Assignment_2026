"""SUMMARY.md must be built only from CSVs and must say plainly what was not run."""
import pandas as pd

import make_summary as ms
import probe_transfer_eval as pe
from test_probe_eval import write_caches


def _run(tmp_path, monkeypatch, name="qwen_logreg", amber=False, **opt):
    write_caches(tmp_path, monkeypatch, amber=amber)
    final = tmp_path / "final"
    pe.run(["synth"], tmp_path, final / name, "logreg", ("B", "A"), 2, 30, True, 0, log=lambda *_: None,
           opts=pe.Options(n_boot_paired=30, recalibrate=True, **opt))
    (final / name / "command.txt").write_text("python src/probe_transfer_eval.py --models synth\n")
    return final


def test_summary_numbers_come_from_the_csv_and_missing_runs_are_reported(tmp_path, monkeypatch):
    final = _run(tmp_path, monkeypatch)
    # a metadata file whose features .pt is absent (as for SmolVLM / AMBER in the real repo) must be listed as blocked
    pd.DataFrame({"row_idx": [0]}).to_parquet(tmp_path / "amber_synth_metadata.parquet")
    text = ms.build_summary(final, tmp_path)
    d = pd.read_csv(final / "qwen_logreg" / "pope_test_summary.csv")
    r = d[(d.task == "B") & (d.method == ms.HID)].iloc[0]
    assert f"{r.auroc:.3f} [{r.ci_lo:.3f}, {r.ci_hi:.3f}]" in text                 # the exact CSV value, formatted
    assert "amber_synth_features.pt" in text and "Not run (blocked by missing cache files)" in text
    assert "**Not run.**" in text                                                     # transfer + cross-model sections
    assert "Paired comparison" in text and "Calibration" in text and "Selective prediction" in text and "By POPE setting" in text
    assert text.index("Task B (primary)") < text.index("Task A (secondary)")           # primary task first


def test_summary_includes_transfer_table_when_amber_ran(tmp_path, monkeypatch):
    final = _run(tmp_path, monkeypatch, amber=True)
    text = ms.build_summary(final, tmp_path)
    assert "probe POPE->AMBER" in text and "Not run (blocked" in text and "All caches with committed metadata" in text


def test_summary_handles_an_empty_results_folder(tmp_path):
    text = ms.build_summary(tmp_path / "nothing", tmp_path)
    assert "Not run" in text and text.startswith("# Probe vs output-level baselines")


def test_error_mix_counts_hallucinated_and_missed_yes_exactly():
    m = pd.DataFrame(dict(setting=["a"] * 6,
                          gt_answer=["no", "no", "yes", "yes", "yes", "no"],
                          pred_answer=["yes", "yes", "no", "yes", "yes", "no"]))        # 2 halluc-yes, 1 missed-yes, 3 correct
    r = ms.error_mix(m, "toy")[0]
    assert r["errors"] == 3 and r["hallucinated_yes_share"] == "0.667" and r["missed_yes_share"] == "0.333"
    assert r["model_yes_rate"] == "0.667" and r["true_yes_rate"] == "0.500"       # 4 of 6 answers are yes; 3 of 6 truths are yes


def test_summary_has_transfer_headline_error_mix_and_per_layer_sections_when_amber_ran(tmp_path, monkeypatch):
    final = _run(tmp_path, monkeypatch, amber=True)
    text = ms.build_summary(final, tmp_path)
    assert "POPE -> AMBER transfer, Qwen2-VL-2B" in text and "Error composition of each cache" in text
    assert "Per-layer POPE -> AMBER transfer" in text and "ranks AMBER errors the wrong way round" in text
    t = pd.read_csv(final / "qwen_logreg" / "transfer_amber.csv")
    r = t[(t.task == "B") & (t.eval_set == "test") & (t.subset == "attribute") & (t.method == "probe AMBER->AMBER")].iloc[0]
    assert f"{r.auroc:.3f} [{r.ci_lo:.3f}, {r.ci_hi:.3f}]" in text                   # ceiling number comes straight from the CSV
