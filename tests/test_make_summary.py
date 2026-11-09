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


# ---------------------------------------------------------------- follow-up (error-mix) section and verdict rule
def _fu_df(rows):
    cols = ["experiment", "variant", "eval_set", "auroc", "ci_lo", "ci_hi", "diff_vs_orig", "diff_vs_orig_lo", "diff_vs_orig_hi"]
    return pd.DataFrame(rows, columns=cols)


def _case(exp, sub, auroc, lo, hi, d, dlo, dhi):
    return (exp, f"{exp} ({sub})", f"AMBER {sub} test", auroc, lo, hi, d, dlo, dhi)


def test_h5_verdict_follows_the_preregistered_rule():
    good = [_case(e, s, 0.8, 0.7, 0.9, 0.3, 0.1, 0.5) for e in ("E1a", "E2") for s in ("existence", "attribute")]
    assert ms.h5_verdict(_fu_df(good))[0] == "SUPPORTED"
    # improves on one subset only (significantly) -> partly; above 0.5 on only one subset is not enough for 'supported'
    part = [_case("E1a", "existence", 0.7, 0.6, 0.8, 0.2, 0.05, 0.3), _case("E1a", "attribute", 0.45, 0.3, 0.6, 0.1, -0.05, 0.25)]
    assert ms.h5_verdict(_fu_df(part))[0] == "PARTLY SUPPORTED"
    # point estimate above 0.5 but CI includes 0.5 -> not 'supported'
    wide = [_case("E1a", s, 0.62, 0.45, 0.78, 0.2, 0.1, 0.3) for s in ("existence", "attribute")]
    assert ms.h5_verdict(_fu_df(wide))[0] == "PARTLY SUPPORTED"
    none = [_case(e, s, 0.4, 0.3, 0.5, 0.01, -0.1, 0.1) for e in ("E1a", "E2", "E1b") for s in ("existence", "attribute")]
    assert ms.h5_verdict(_fu_df(none))[0] == "NOT SUPPORTED"
    # E1b uses target-domain statistics, so even a perfect E1b can never give 'supported'
    only_e1b = [_case("E1b", s, 0.9, 0.85, 0.95, 0.5, 0.3, 0.7) for s in ("existence", "attribute")]
    assert ms.h5_verdict(_fu_df(only_e1b))[0] == "PARTLY SUPPORTED"


def test_summary_contains_followup_section_built_from_the_csvs(tmp_path, monkeypatch):
    import followup_errormix as fu
    final = _run(tmp_path, monkeypatch, amber=True)
    fu.main(["--models", "synth", "--cache-dir", str(tmp_path), "--out-dir", str(final / "followup_errormix"), "--n-boot", "20",
             "--n-boot-paired", "20", "--reference-dirs", str(final / "qwen_logreg")])
    text = ms.build_summary(final, tmp_path)
    assert "## Follow-up: error-mix test (H5)" in text and fu.PREDICTION in text and "Verdict: H5" in text
    d = pd.read_csv(final / "followup_errormix" / "synth" / "followup_summary.csv")
    r = d[(d.experiment == "E2") & (d.eval_set == "AMBER attribute test")].iloc[0]
    assert f"{r.auroc:.3f} [{r.ci_lo:.3f}, {r.ci_hi:.3f}]" in text                    # exact CSV value
    assert "E0 sign diagnostic" in text and "cost of each variant on the training domain" in text and "Per-layer AUROC (descriptive only" in text
    assert text.index(fu.PREDICTION) < text.index("Verdict: H5")                   # prediction stated before the verdict


def test_summary_says_followup_not_run_when_absent(tmp_path, monkeypatch):
    final = _run(tmp_path, monkeypatch)
    assert "## Follow-up: error-mix test (H5)\n\n_not run_" in ms.build_summary(final, tmp_path)
