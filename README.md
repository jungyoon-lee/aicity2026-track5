# Cosmos3-Super LoRA, iteration 1,680 — AI City Challenge 2026 Track 5

Traffic-video future prediction using a generation-path LoRA adapter for
NVIDIA Cosmos3-Super. The submitted archive scored 76.0385 and ranked second
on the post-deadline Public leaderboard (checked July 23, 2026).

## Layout

```
cosmos/       NVIDIA Cosmos at the pinned revisions below, with our 5-file change applied
cosmos.patch  that change on its own, for reading
model/        the trained LoRA adapter
prepare/      WTS / BDD_PC_5K -> training clip builders
scripts/      training, inference, verification
config/       training config
```

Pinned revisions:
[NVIDIA/cosmos@335392c](https://github.com/NVIDIA/cosmos/tree/335392ca873a374ef1ceef6477e2c2c983eebee8) ·
[NVIDIA/cosmos-framework@411d25b](https://github.com/NVIDIA/cosmos-framework/tree/411d25b2e35bc441126f48c44a4b93e1c0564274)

We added exactly two things to Cosmos: model-only checkpoint saving during
training (`checkpoint/dcp.py`, `trainer/__init__.py`), and LoRA adapter loading
at inference (`inference/*.py`).

## Requirements

Inference peaked at 124,260 MiB of GPU memory, so it needs an H200-class card.
Training needs 8 of them.

Tested on: 8x NVIDIA H200, driver 580.126.16, Python 3.13, PyTorch 2.10.0+cu130,
transformers 4.57.6, safetensors 0.7.0, FFmpeg 7.0.2.

`ffmpeg` and `ffprobe` must be on `PATH` — the data builders shell out to them.

## Setup

```bash
sudo apt-get install -y --no-install-recommends curl ffmpeg git-lfs libx11-dev wget

cd cosmos/packages/cosmos3
uv sync --all-extras --group=cu130-train
source .venv/bin/activate
export LD_LIBRARY_PATH=
```

Then fetch the exact model revisions used by the submission and convert the
base checkpoint to DCP format. Both training and inference read DCP, not the
raw Hugging Face release:

```bash
# Run from cosmos/packages/cosmos3.
mkdir -p examples/checkpoints/hf/Cosmos3-Super examples/checkpoints/wan22_vae

hf download nvidia/Cosmos3-Super \
  --revision 0b900a087112c48d82f001804da8ba25e6969397 \
  --local-dir examples/checkpoints/hf/Cosmos3-Super

hf download Wan-AI/Wan2.2-TI2V-5B Wan2.2_VAE.pth \
  --revision 921dbaf3f1674a56f47e83fb80a34bac8a8f203e \
  --local-dir examples/checkpoints/wan22_vae

python -m cosmos_framework.scripts.convert_model_to_dcp \
  -o examples/checkpoints/Cosmos3-Super \
  --checkpoint-path examples/checkpoints/hf/Cosmos3-Super

# Return to the repository root for all remaining commands.
cd ../../..

sha256sum -c CHECKSUMS.sha256
```

Run the remaining commands from the repository root.

`train.sh` refuses to start if `$BASE_CHECKPOINT/model/.metadata` is missing,
which is the signal that the DCP conversion has not been run. The DCP files
are derived from the pinned Hugging Face snapshot and are not redistributed.

## Weights

`model/c3super1680_lora.safetensors` — 79.8 MB, LoRA rank 16 / alpha 32, 512
tensors, included in this repository. Its SHA-256 digest is recorded in
`CHECKSUMS.sha256`.

## Inference

The test root is the official Track 5 test set, one directory per case:

```
$WTS_TEST_ROOT/<case>/caption.json       # needs "frame length" and "event_phase"
$WTS_TEST_ROOT/<case>/input/0.png
$WTS_TEST_ROOT/<case>/input/1.png
...
$WTS_TEST_ROOT/<case>/input/<K-1>.png    # contiguous observed-history frames
```

```bash
export AICITY_ROOT="$(pwd)"

python scripts/run_infer.py \
  --test-root $WTS_TEST_ROOT \
  --checkpoint cosmos/packages/cosmos3/examples/checkpoints/Cosmos3-Super \
  --config-file config/train.yaml \
  --adapter-path model/c3super1680_lora.safetensors \
  --output-dir outputs/pred \
  --gpus 0,1,2,3,4,5,6,7 \
  --batch-size 1 \
  --num-steps 8 --guidance 3.0 --seed 0

python scripts/verify.py --test-root $WTS_TEST_ROOT --zip outputs/pred/submission_prediction.zip
```

Writes exactly the requested `N` frames per case as
`prediction/<case>/0.png ... <N-1>.png`. The submitted run took 27 minutes for
all 71 cases across 8 GPUs. `verify.py` checks the case set, contiguous names,
frame counts, and the resolution of every PNG.

## Data

WTS and BDD_PC_5K are obtained from the challenge organizers under their own
terms; see the official [Track 5 page](https://www.aicitychallenge.org/2026-track5/)
and [dataset-access page](https://www.aicitychallenge.org/ai-city-challenge-dataset-access/).
This repository contains no challenge data.

Set the three variables below to the dataset directories themselves (not their
parent directory):

```bash
export WTS_ROOT=/path/to/WTS
export BDD_CAPTION_ROOT=/path/to/BDD_PC_5K/caption
export BDD_VIDEO_ROOT=/path/to/bdd_pc_5k/videos
```

The builders expect the official layouts unchanged. WTS train and validation
are both used; only BDD_PC_5K train is used by the submitted model:

```
$WTS_ROOT/caption/{train,val}/**/{overhead_view,vehicle_view}/*_caption.json
$WTS_ROOT/video/{train,val}/**/{overhead_view,vehicle_view}/*.mp4

$BDD_CAPTION_ROOT/train/*_caption.json
$BDD_VIDEO_ROOT/train/<video file named in the caption>
```

A WTS caption at `caption/train/<scenario>/overhead_view/x_caption.json`
resolves its videos from `video/train/<scenario>/overhead_view/`, so the two
trees must stay parallel.

## Training

```bash
python -m prepare.wts --legacy-v1 --wts-root $WTS_ROOT --output-root data/wts
python -m prepare.bdd --legacy-v1 --caption-root $BDD_CAPTION_ROOT \
                      --video-root $BDD_VIDEO_ROOT --output-root data/bdd --splits train

bash scripts/train.sh
```

Preparation transcodes every phase to a 165-frame clip, so it is I/O bound and
slow. Both builders skip clips that already exist, so an interrupted run can be
restarted with the same command; each writes a manifest under
`<output-root>/manifests/`. A complete build used by the submission must produce
exactly these training manifests:

```text
data/wts/train/video_dataset_file.jsonl   3,522 rows
data/bdd/train/video_dataset_file.jsonl  12,151 rows
```

Check them before training:

```bash
wc -l data/wts/train/video_dataset_file.jsonl \
      data/bdd/train/video_dataset_file.jsonl
```

`train.sh` reads those two files and then runs 1,680 iterations on 8 GPUs. The
BDD_PC_5K validation split is not required and is not read by `train.sh`.

All hyperparameters sit at the top of `scripts/train.sh`, with defaults matching
the submitted run. Extract the adapter from a checkpoint with
`python scripts/export_lora.py --checkpoint <iter_000001680> --output lora.safetensors`.

## Method

Each phase becomes a 165-frame, 16 fps, 1280x720 clip whose first 45 frames are
context. Every clip yields 19 windows of length 93 to 165, which varies the
prediction horizon seen during training. WTS train+val contributes 3,522 clips
and BDD_PC_5K train contributes 12,151.

The submitted `--legacy-v1` preprocessing starts every training window at clip
frame zero. Consequently, a short visual condition comes from the beginning of
the 45-frame pre-target history, whereas inference retains the most recent
compatible observations. This alignment limitation is preserved here so the
released recipe matches iteration 1,680; it should be corrected in future
training runs.

LoRA is injected only into the generation-path attention projections
(`q_proj_moe_gen`, `k_proj_moe_gen`, `v_proj_moe_gen`, `o_proj_moe_gen`).
Context length is sampled during training over `{0, 1, 5, 8, 10, 12}` latent
frames, exposing the model to variable context lengths also encountered at test
time. Prompts are generated deterministically from the official pedestrian and
vehicle captions, with no manual editing.

AdamW, lr 5e-4, betas (0.9, 0.95), weight decay 0, 50 warmup steps, CFG dropout
0.05, FSDP with context-parallel degree 2, max packed sequence length 45,056,
1,680 iterations.

This submission does not use PhysicalAI AV, a separate Reasoner/Grounder, manual
prompt rewriting, test-set annotations, seed ensembling, or post-processing
selected on the test set.

## License

Our code is MIT (`LICENSE`). `cosmos/` comes from NVIDIA under OpenMDW-1.1
(`NOTICE`, `cosmos/packages/cosmos3/LICENSE`).
