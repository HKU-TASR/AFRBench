# Running the benchmark

`python -m afrbench.benchmark` runs the [AFRBench-basic protocol](protocol.md) on one
method: a built-in method or your own. This page goes from a fresh machine to results.

## Requirements

- Linux, conda, and one NVIDIA GPU with at least 24 GB (the largest jobs need about
  22 GB to train, 19 GB to protect, and 16 GB to evaluate). A CUDA toolkit and a C++
  compiler are needed for the methods with custom CUDA ops (CLIP2Protect, DivTrackee,
  FALCO, Adv-Inversion), Protego (pytorch3d), and AT3D (nvdiffrast).
- Disk:

  | Item | Size |
  |---|---|
  | MillionCelebs download (can be deleted after packing) | 445 GB |
  | MillionCelebs pack | 179 GiB |
  | Distractor embedding cache, per evaluation FR model | 47 GB |
  | Getty photos | 3.2 GB |
  | Training corpora (methods with a train stage) | up to 70 GB |
  | Method weights | tens of GB |
  | Protected photos, per method | 25-40 GB |

- Time: protecting the 2,500 query and gallery photos takes from minutes to about 70
  GPU-hours depending on the method (about 260 GPU-hours for all 24 methods). Methods
  with a train stage train first (five times for Adv-Makeup, AMT-GAN, and DiffAM). The
  first evaluation with an FR model embeds all 22.8M distractors once; later methods
  reuse the cache.

## 1. Install

```bash
git clone https://github.com/HKU-TASR/AFRBench && cd AFRBench
bash setup.sh
conda activate afrbench
```

Two methods use files whose licenses require you to obtain them yourself:

- **Protego** needs the FLAME 2020 head model. Register at
  <https://flame.is.tue.mpg.de>, download `FLAME2020.zip`, extract `generic_model.pkl`,
  and register it:

  ```bash
  python -m afrlib add protego/generic_model.pkl /path/to/FLAME2020/generic_model.pkl
  ```

  The command checks the file's sha256
  (`efcd14cc4a69f3a3d9af8ded80146b5b6b50df3bd74cf69108213b144eba725b`).
- **AT3D** needs `BFM_model_front_patch.mat`. It contains Basel Face Model 2009 data,
  under the [BFM license](https://faces.dmi.unibas.ch/bfm/main.php?nav=1-2&id=downloads).
  It is downloaded automatically from the link the AT3D authors publish (step 3 of the
  [AT3D README](https://github.com/thu-ml/AT3D)). If that fails, download it from there
  and run `python -m afrlib add at3d/BFM_model_front_patch.mat <file>`. Nothing has to
  be generated: AFRLib's AT3D does not use `01_MorphableModel.mat`.

## 2. Obtain the data

Download the 5,000 Getty photos listed by `getty_asset_id` in
`data/afrbench-basic/annotations.json` (see [data.md](data.md)); keep the asset id in
each file name. Then download and pack MillionCelebs (both steps resume if interrupted):

```bash
python -m afrbench.millioncelebs download --out /big/mc
python -m afrbench.millioncelebs pack --src /big/mc --out /big/mc/noise.lmdb
```

## 3. Assemble the dataset

```bash
python -m afrbench.prepare --getty ~/getty --noise /big/mc/noise.lmdb --out /big/afrbench/basic
```

`prepare` matches each photo by its asset id, checks that it has the recorded size and
no EXIF rotation, links it into place, and verifies the pack. Problems are listed in
`prepare_report.txt`; fix them and run the command again. The result:

```
/big/afrbench/basic/
  annotations.json  identities.json  dataset.json  protectee_groups.json
  purge/  targets/pool.json  targets/hq/
  protectees/<identity>/<image>.jpg   -> your photos
  noise.lmdb                          -> /big/mc/noise.lmdb
```

Distractor embeddings are cached next to it, in `/big/afrbench/_shared/`.

## 4. Run a method

```bash
python -m afrbench.benchmark --method fawkes --dataset /big/afrbench/basic --out results/fawkes
```

The command writes the method's job configs to `results/fawkes/configs/`, then runs
them in order: train (if the method trains), fit (if it fits), protect for each of the
50 protectees, and evaluation. Finished jobs are skipped, so running the same command
again resumes after an interruption. At the end it prints the scenario table and writes
`results/fawkes/summary.csv`.

Options:

| Option | Meaning |
|---|---|
| `--evals minimal,jpeg,median` | evaluation jobs from the suite in [protocol.md](protocol.md), or `all` (default `minimal`) |
| `--set epsilon=8` | override a constructor argument (repeatable; results are then not the protocol's defaults) |
| `--eval-fr backbone/loss/dataset` or `--eval-fr package.module:Class` | also evaluate with another FR model |
| `--corpus-root ms1m=/data/ms1m` | use a local copy of a training corpus |
| `--configs-only` | write the configs without running them |
| `--device cuda:1` | device for all jobs |

## 5. Your own method or FR model

```bash
export PYTHONPATH=$PWD/examples/custom_method:$PYTHONPATH
python -m afrbench.benchmark --method pgd_cloak:PGDCloak --dataset /big/afrbench/basic --out results/pgd
python -m afrbench.benchmark --method fawkes --eval-fr my_pkg.fr:MyRecognizer --dataset /big/afrbench/basic --out results/fawkes
```

See [extending.md](extending.md) for the method and FR-model interfaces.

## Outputs

```
results/fawkes/
  configs/{train,fit,protect,eval}/...yaml   one file per job
  train/fawkes/                              trained weights, train/<method>/<target>/ per target
  fit/fawkes/<protectee>/                    weights fitted to one protectee
  protect/fawkes/<protectee>/                protect/<method>/<target>/<protectee> for methods given a target image
    outputs/*.npy                            protected frames, float32 in [0, 255]
    preview/*.jpg
    manifest.json
  eval/minimal/
    metrics.json                             scenario means per metric
    per_image.jsonl                          one row per query and scenario
    units/, vq_partial/                      per-scenario and per-run results, reused on resume
  summary.csv                                one row per run, FR system, seen flag and
                                             scenario, with each metric's mean and variance
```

Every job is a plain config; `python -m afrbench results/fawkes/configs/protect/fawkes/<name>.yaml`
runs or re-runs one of them. Each job records its inputs and code in
`config.lock.yaml`; a job whose inputs or code changed runs again.

When a code or metadata change cannot have altered the outputs, `python -m afrbench.relock
<results_root> --kinds train,fit,protect` re-stamps the finished locks instead of re-running
them: it refreshes each lock's code, upstream-lock and dataset digests, leaves the config,
the outputs and the image hashes alone, and appends the time and the replaced digests to the
lock, so a re-stamped artifact stays visible as one. `--dry-run` lists what it would touch.
