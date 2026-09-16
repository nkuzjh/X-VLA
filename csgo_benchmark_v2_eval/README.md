# CSGO Benchmark v2 shared evaluator

This folder is a standalone evaluator for the Seen-10 localization and image
generation tasks. Runtime code reads `minimal_dataset_report.json`,
`benchmark_manifest.json`, the published split files and
`calibration/z_calibration.json`; it imports neither UniLIP nor ControlAR model
code.

Use the existing UniLIP metric environment by default:

```bash
UNILIP_PYTHON=${UNILIP_PYTHON:-/home/jiahao/miniconda3/envs/UniLIP/bin/python}
DATA_ROOT=/home/jiahao/task/UniLIP/data/csgo_benchmark_v2
```

The evaluator never installs packages. `requirements.txt` lists metric-only
dependencies and deliberately leaves PyTorch/torchvision to the caller's
compatible environment.

Generation predictions are JPEGs at
`<pred-root>/<map>/<file_frame>.jpg`; `<pred-root>` may also be the containing
`gen_imgs` directory or its parent task output directory. Localization takes
`predictions.jsonl` (or a directory containing it). Each row uses
`sample_id="<map>/<file_frame>"`, `map_name`, and normalized
`pred_x`, `pred_y`, `pred_z`, `pred_pitch`, `pred_yaw`. The normalized pose
order is `[x, y, z, pitch, yaw]`; X/Y divide by 1024, Z uses the published
per-map calibration without clipping, and both angles divide degrees by 360.
An explicit `--pose-space physical` option is available for external files
whose same fields are already physical coordinates and degrees.

Official commands require every Seen-10 sample, reject missing or extra image
predictions, and write no result until all maps and metrics have completed.
The output directory contains `per_map/<map>.json` and
`summary_equal_map.json`:

```bash
$UNILIP_PYTHON csgo_benchmark_v2_eval/run_eval.py localization \
  --pred-root outputs/csgo_benchmark_v2_seen10/ControlAR/seed_0/localization \
  --data-root "$DATA_ROOT" \
  --output outputs/csgo_benchmark_v2_seen10/ControlAR/seed_0/eval/localization

$UNILIP_PYTHON csgo_benchmark_v2_eval/run_eval.py discrete \
  --pred-root outputs/csgo_benchmark_v2_seen10/ControlAR/seed_0/discrete/gen_imgs \
  --data-root "$DATA_ROOT" \
  --output outputs/csgo_benchmark_v2_seen10/ControlAR/seed_0/eval/discrete

$UNILIP_PYTHON csgo_benchmark_v2_eval/run_eval.py continuous \
  --pred-root outputs/csgo_benchmark_v2_seen10/ControlAR/seed_0/continuous/gen_imgs \
  --data-root "$DATA_ROOT" \
  --output outputs/csgo_benchmark_v2_seen10/ControlAR/seed_0/eval/continuous
```

An explicit smoke command reads only the first selected sample (or complete
manifest clips for continuous), prints `smoke_only: true`, and never writes a
per-map file or formal summary. It is suitable for the single-image model
smoke and intentionally skips FID/FVD distribution scores:

```bash
$UNILIP_PYTHON csgo_benchmark_v2_eval/run_eval.py smoke discrete \
  --pred-root outputs/csgo_benchmark_v2_seen10/ControlAR/seed_0/discrete/gen_imgs \
  --data-root "$DATA_ROOT" --limit 1
```

`BenchmarkData(data_root).rows(split)` provides the manifest-driven model data
contract. Supported names include `seen_train`, `seen_validation`,
`seen_discrete_test` and `seen_continuous`; continuous rows retain `clip_id` and
zero-based `frame_index`, and `BenchmarkData(data_root).clips()` returns whole
clips in their published order. Each row contains `sample_id`, `map_name`,
`file_frame`, absolute `image_path` and `radar_path`, normalized `pose`,
`pose_raw`, `z_calibration`, and continuous identity fields.

Metric settings and calculation details are kept in `benchmark_v2.yaml`.
Continuous evaluation is fixed to 16-frame FVD windows, stride 16, 224-pixel
FVD inputs, frame gap threshold 2 and minimum track length 4. It retains
UniLIP's Farneback flow parameters and `BORDER_REFLECT` warp. TWE/TDE operate on
resized uint8 RGB (0–255), average adjacent-frame errors within each track,
then average tracks within a map. PSNR/SSIM preserve UniLIP's batch-metric
outputs weighted by batch size (batch size 8 for discrete, 1 for continuous).
Boundary F1 accumulates edge hit/count totals over the map before computing
precision, recall and F1. The equal-map macro averages the ten per-map results
with equal weight.

For continuous evaluation, FVD reuses the I3D cache at
`/home/jiahao/task/UniLIP/loaded_models` by default. Override it with
`--fvd-cache-dir` or `UNILIP_FVD_CACHE_DIR`. FID uses the installed
torchmetrics/torch-fidelity weights and follows the configured metric
environment.

See `THIRD_PARTY_NOTICES.md` for metric-code attribution and `LICENSE` for the
evaluator project license.
