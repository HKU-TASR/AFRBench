# Using AFRLib as a library

AFRLib gives every method the same interface: construct it with a device and a list of
face-recognition (FR) models to attack, then call `train`, `fit`, and `protect` as the
method requires (see [methods.md](methods.md)).

## Protect a photo

```python
from PIL import Image
from torchvision.transforms.functional import pil_to_tensor

from afrlib import build_fr_ensemble
from afrlib.afr import Fawkes
from afrlib.contracts import EvalData

photo = pil_to_tensor(Image.open("me.jpg").convert("RGB")).float()   # [3, H, W] in [0, 255]
frs = build_fr_ensemble(
    [{"backend": "frbench", "backbone": "irse-100", "loss": "arcface", "dataset": "ms1m"}],
    device="cuda",
)
det = frs[0].detect([photo])                                          # RetinaFace (FRBench)
geometry = {"face_box": det.boxes[0][0].tolist(), "landmarks": det.landmarks[0][0].tolist()}

method = Fawkes("cuda", frs)                                          # hyper-parameters are kwargs
protected = method.protect(EvalData(eval_imgs=[photo], eval_geometry=[geometry]))[0]
```

`protect` takes original photos and returns protected photos of the same size, as
float32 tensors in `[0, 255]`. Each photo comes with its geometry: five landmarks
(left eye, right eye, nose, left and right mouth corners) and a face box
`[x1, y1, x2, y2]`, in that photo's pixel coordinates. Any detector can provide them;
`detect` of an FRBench model is the simplest. The method crops, attacks, and pastes the
result back itself. Photos of different sizes can be passed together as a list.

## Inputs by role

Besides `eval_imgs`, a method may read the roles below. Prepare each one to the format
the method declares, with `afrlib.utils.face_crop.prepare_input(photo, spec, landmarks, face_box)`,
which returns `(crop, plan)`.

| Role | Stage | What it is | Format |
|---|---|---|---|
| `target_img` | train or protect | one photo of the identity to impersonate | `method.spec_for("target_img")` (train: `method.input_specs("train")["target_img"]`) |
| `target_pool` | protect | photos of several other identities | `method.spec_for("target_pool")`, stacked |
| `protectee_album` | fit | photos of the protected person, other than the ones to protect | `method.input_specs("fit")["protectee_album"]`, stacked |
| `reference_imgs` | protect | one photo of the same person per protected photo | `method.spec_for("reference_imgs")` |
| `corpus` | train | a training corpus | `afrlib.data.get_corpus(name)` |
| `prompt`, `message`, `makeup_reference_name` | protect or train | method-specific scalars | as documented by the method |

A method raises an error when a required role is missing and warns about roles it does
not read.

## Other lifecycles

**Impersonation** (for example DiffAIM):

```python
import torch
from afrlib.afr import DiffAIM
from afrlib.utils.face_crop import prepare_input

method = DiffAIM("cuda", frs)
target, _ = prepare_input(target_photo, method.spec_for("target_img"), t_landmarks, t_box)
out = method.protect(EvalData(eval_imgs=[photo], eval_geometry=[geometry], target_img=target))
```

**Fit to one person** (for example OPOM):

```python
from afrlib.afr import OPOM
from afrlib.contracts import FitData

method = OPOM("cuda", frs)
spec = method.input_specs("fit")["protectee_album"]
album = torch.stack([prepare_input(p, spec, lm, box)[0] for p, lm, box in my_album])
method.fit(FitData(protectee_album=album))
method.save_run("runs/opom_me")                 # reload with OPOM.from_run("runs/opom_me", "cuda")
out = method.protect(EvalData(eval_imgs=[photo], eval_geometry=[geometry]))
```

**Train** (for example AMT-GAN, which bakes the target identity into its generator):

```python
from afrlib.afr import AMT_GAN
from afrlib.contracts import TrainData
from afrlib.data import get_corpus

method = AMT_GAN("cuda", frs)
target, _ = prepare_input(target_photo, method.input_specs("train")["target_img"], t_landmarks, t_box)
method.train(TrainData(corpus=get_corpus("mt"), target_img=target))
method.save_run("runs/amt_gan")
method = AMT_GAN.from_run("runs/amt_gan", device="cuda")
```

A run folder holds the constructor configuration, the weights, and the baked-in inputs,
so `from_run` restores exactly the trained (or fitted) state.

## FR models

`build_fr_ensemble` takes one spec per model. Any FRBench model is
`{"backend": "frbench", "backbone": ..., "loss": ..., "dataset": ...}`
(see the [FRBench model zoo](https://github.com/HKU-TASR/FRBench)). `facenet_pytorch`
is also built in, and your own model can be used directly or registered by import path;
see [extending.md](extending.md).

## Weights, corpora, and the asset CLI

Weights are downloaded on first use to `~/.afrlib` (set `AFRLIB_HOME` to move it;
`AFRLIB_OFFLINE=1` forbids downloads). Training corpora are downloaded and packed on
first use by `afrlib.data.get_corpus(name)`; `get_corpus(name, root=...)` uses a local
copy instead. A corpus is packed once into an LMDB of raw bytes plus the annotations the
dataset ships, then filtered per run, so a method sees only what its filters allow:

```python
from afrlib.data import ExcludeIdentities, get_corpus

corpus = get_corpus("casia_webface", filters=[ExcludeIdentities(protectees)])
```

`DATASET_REGISTRY` lists the corpora; AFRBench filters the protectees out of every
training corpus (see [protocol.md](protocol.md)). FRBench models are cached in `~/.frbench`. Protego's FLAME model and,
if its download fails, AT3D's BFM file are registered by hand; see
[benchmark.md](benchmark.md#1-install).

```bash
python -m afrlib status                  # registered assets and what is cached
python -m afrlib download clip2protect gift   # pre-fetch the assets of some methods (or --all)
python -m afrlib add <key> <path>        # register a file you downloaded yourself
python -m afrlib hash <path>             # the sha256 a local file would need
python -m afrlib purge [method]          # delete cached assets
```

## Logs and progress bars

AFRLib logs under the standard-library logger `afrlib`, so `logging` configures it as
usual and `logging.getLogger("afrlib.afr.dadp").setLevel(...)` filters one module.
`afrlib.set_verbosity(...)` and `AFRLIB_VERBOSITY` set the level, `afrlib.add_file_handler(path)`
also writes to a file, and warnings are `AFRLibWarning`, which `warnings.filterwarnings`
matches. Progress bars are switched off with `afrlib.disable_progress_bars()` or
`AFRLIB_DISABLE_PROGRESS_BARS=1`.
