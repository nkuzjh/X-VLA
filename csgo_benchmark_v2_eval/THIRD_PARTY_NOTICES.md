# Third-party metric code

The FVD helper in `fvd_metric/fvd.py` is copied from the UniLIP checkout's
`third_party/PyTorch-Frechet-Video-Distance/fvd_metric/fvd.py` at commit
`431844ef10417f661dbe47832831ab0558acb340`. Its upstream README and source
disclaimer are retained in `fvd_metric/README.upstream.md` and the source
header. The only local adaptation selects the model cache directory from
`UNILIP_FVD_CACHE_DIR`; the FVD feature extraction and Fréchet calculation are
unchanged.

The evaluator's project files are covered by the adjacent Apache-2.0
`LICENSE`. The upstream FVD checkout did not include a separate license file
in the inspected snapshot; its source attribution and disclaimer are retained
here without adding a new license claim.
