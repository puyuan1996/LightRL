# Math RLVR RJob entry points

These scripts are the only scheduler-facing entry points for the Math RLVR
evaluation, DAPO training, and checkpoint conversion flows.  They intentionally
contain no site namespace, registry, mount, or persistent-storage address.

Configure the scheduler through environment variables or a private
`rjob.env` file (see `rjob.env.example`):

```bash
cp local/rjob/rjob.env.example local/rjob/rjob.env
# edit the private file, then:
source local/rjob/rjob.env
export LIGHTRL_ROOT="$PWD"
```

The submitters fail early when the required scheduler values are absent.  All
model, checkpoint, dataset, output, reward, cap, and evaluation settings are
also explicit environment variables; no checkpoint is silently selected.

Examples:

```bash
MODEL_PATH=/path/to/checkpoint MODEL=my-model RJOB_NAME=math-eval \
  bash local/rjob/submit_math_rlvr_eval.sh

HF_CKPT=/path/to/checkpoint REF_LOAD=/path/to/reference \
  RJOB_NAME=math-dapo-seed1 NUM_EPOCHS=10 \
  bash local/rjob/submit_math_rlvr_train.sh
```

Set `DRY_RUN=1` (CLI submitter) or `RJOB_DRY_RUN=1` (Python client submitter)
to inspect the generated command/spec without submitting a job.
