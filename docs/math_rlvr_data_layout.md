# Math RLVR data layout

The canonical shared location is:

```text
/mnt/shared-storage-user/puyuan/data/math_rlvr/
```

It contains the normalized JSONL files and their manifests/checksums:

| Dataset | File | Rows | Unique questions |
|---|---|---:|---:|
| AIME2025 | `aime-2025.jsonl` | 30 | 30 |
| AIME2024 | `aime-2024.jsonl` | 30 | 30 |
| AMC23 | `amc23.jsonl` | 40 | 40 |
| MATH-500 | `math-500.jsonl` | 500 | 500 |
| DAPO-Math-17k | `dapo-math-17k.jsonl` | 17,255 | 17,255 |

`MATH_DATA_ROOT` is the explicit override for both training and evaluation;
`LIGHTRL_DATA_ROOT` is accepted as a generic alias. If neither is set, the
shared canonical root is preferred, followed by `data/math_rlvr` in the
repository and the repository benchmark directory.

The copy into the canonical root was verified against the original
`SHA256SUMS` before updating launchers. The former staging directory was
removed after verification, so new runs should use the canonical root or an
explicit `MATH_DATA_ROOT` override.
