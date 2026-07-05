# Calibration Workflows

Home for score dump, conformal calibration, layer sweep, calibration artifact,
and correction-training export CLIs.

Current 0.3 entry point:

- `correction_training_export.py`: exports verified correction-buffer records
  into SFT or DPO JSONL without trusting unverified self-generated corrections.
