# Adding a method or an FR model

Both registries accept out-of-tree classes by import path, `"package.module:Class"`,
so you do not need to edit AFRLib. The module must be importable (installed, or on
`PYTHONPATH`).

```python
from afrlib import AFR_REGISTRY, FR_REGISTRY

Method = AFR_REGISTRY["my_pkg.cloak:MyCloak"]
Recognizer = FR_REGISTRY["my_pkg.fr:MyRecognizer"]
```

The same strings work on the command line:

```bash
python -m afrbench.benchmark --method my_pkg.cloak:MyCloak ...
python -m afrbench.benchmark --method fawkes --eval-fr my_pkg.fr:MyRecognizer ...
```

## A new AFR method

Subclass `afrlib.afr.base.AFRMethod`. A complete, runnable example is
[`examples/custom_method/pgd_cloak.py`](../examples/custom_method/pgd_cloak.py).

1. Set `name`, and declare the roles each stage reads with `StageInputs`:
   `eval_inputs = StageInputs(required=("eval_imgs",), optional=(...))`, and likewise
   `train_inputs` / `fit_inputs` if the method has those stages. The role names are the
   fields of `TrainData`, `FitData`, and `EvalData` in `afrlib/contracts.py`.
2. Call `super().__init__(device, fr_models, assets_path=assets_path)` and keep
   hyper-parameters as constructor arguments. The injected FR models are frozen and
   available as `self.fr_models` (`self._require_fr()` checks there are at least
   `min_surrogates`).
3. Implement `attack_spec()`, the `ImageSpec` (size, channels, crop, alignment, value
   range) of the face crop the method works on.
4. Implement `protect(data)`: take `data.eval_imgs` (original photos, float in
   `[0, 255]`) and `data.eval_geometry`, and return one protected photo per input, same
   shape and range. `afrlib.utils.face_crop` does the geometry:
   `prepare_inputs(photos, geometry, spec)` crops, `paste_deltas` puts a perturbation
   (`adv - clean`) back, and `paste_faces` puts a regenerated face back. Each crop is an
   affine warp to the crop's natural size (antialiased when it downsamples), then a
   bicubic resize to the spec's size, so the paste helpers use its exact geometric inverse;
   `paste_deltas` leaves pixels outside the crop unchanged. The supported framings:

   | `crop_type`, `align_type` | Crop |
   |---|---|
   | `Original`, any | the photo itself |
   | `FaceBox`, `NoAlign` | the detector box, squared |
   | `FaceBox`, `ArcFace5` | 5-point similarity warp onto the ArcFace template |
   | `FFHQ`, `FFHQQuad` and `LADN600`, `LADN600` | the FFHQ quad from the eye and mouth landmarks (StyleGAN aligner) |
   | `FFHQ`, `NoAlign` | the FFHQ quad's axis-aligned bounding square |
5. Implement `get_config()` by extending `super().get_config()` with the constructor
   arguments, so `from_config` and `from_run` can rebuild the method.
6. Optional stages: implement `train(data)` (and set `training_corpus` to one of the
   corpora in `afrlib.data.DATASET_REGISTRY`) or `fit(data)`. A method with weights also
   implements `save_run(folder)` / `load_run(folder)` and calls `self._maybe_load_run()`
   at the end of `__init__`.
7. Assets: declare downloadable weights with `afrlib.register_assets(Asset(...))` and
   load them with `afrlib.get_asset(key)`. Frozen pretrained priors stay assets; they are
   not written to or read from a run folder.

Each stage runs at most once per instance, and the helpers on the base class enforce the
order: `train` and `fit` call `_check_inputs(...)` and then `_begin_training()` or
`_begin_fitting()`; the last stage the method implements calls `_mark_ready()`, as does a
`load_run` that finds a complete run; `protect` calls `_check_inputs(...)`,
`_require_ready()`, and `_require_fr()` when it uses the surrogates.

`save_run(folder)` writes the method's weights under names it chooses, the baked training
inputs under `inputs/`, and `config.json` last, so a folder holding `config.json` is a
complete run. The `get_config()` keys that name an `__init__` parameter are what
`from_config` and `from_run` rebuild the method from; any other keys are provenance.

AFRBench decides how to call your method from these declarations: a method with a
`target_img` role is run once per protectee against that protectee's assigned target,
and one with `target_pool` receives a pool of other identities (see
[protocol.md](protocol.md)). A method that trains must use one of the registered
training corpora, because the protocol removes the protectees from it.

To add a method to AFRLib itself, put the package in `afrlib/afr/<name>/`, add it to
`_AFR_TARGETS` in `afrlib/afr/__init__.py`, and declare its assets in
`afrlib/afr/<name>/assets.py`.

## A new FR model

Subclass `afrlib.fr.base.FaceRecognizer`; `afrlib/fr/facenet/` is a complete example.

1. Class attributes: `backend_name`, `channel`, `embedding_dim`, `input_size`. For an
   out-of-tree model, `backend_name` must equal its import path (for example
   `"my_pkg.fr:MyRecognizer"`), so that recorded specs rebuild it.
2. Properties: `name`, and `mean` / `std` on the `[0, 1]` scale in the backbone's
   channel order; the `backbone` module.
3. `forward(...)`: detect, align, and embed; `embed_crops`, `normalize`, and `detect`
   have default implementations in the base class.
4. `get_spec()`: the constructor arguments that identify the model (JSON values only).
   `build_fr(model.get_spec())` must rebuild it, on another machine too, so leave out
   per-process objects (a shared detector or backbone, a local `model_dir`) and the default
   checkpoint, which the asset cache resolves.

Pixel conventions: `forward` and `detect` take RGB float images in `[0, 255]`, while
`embed_crops` takes pre-cropped faces in `[0, 1]` or `[-1, 1]` (its `input_space`) and
applies no detection, range validation or clamping — some methods feed the model unclamped
tensors on purpose.

In AFRBench, evaluation uses the benchmark's own detection and ArcFace-112 alignment and
feeds the crops to `embed_crops`, resized to the model's `input_size`. Embeddings are
cached by the FR spec, not by the model's code: put a version or checkpoint name in the
spec when you change the weights.
