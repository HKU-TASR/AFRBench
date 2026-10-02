# Methods

AFRLib implements 26 anti-facial-recognition methods behind one interface,
`afrlib.afr.base.AFRMethod`. AFRBench-basic runs 24 of them; EfficientDecision and DPA
are implemented but not part of the protocol.

Each method goes through up to three stages:

- **train** on a training corpus, once per method (or once per target when the target
  is baked into the weights);
- **fit** to one protectee, using that person's album photos;
- **protect** original photos at run time.

The roles each stage reads are declared on the class (`train_inputs`, `fit_inputs`,
`eval_inputs`) and listed below. `eval_geometry` (landmarks and face box of each photo)
always accompanies `eval_imgs`.

| Method | Stages | Goal | Train reads | Fit reads | Protect reads | Training corpus |
|---|---|---|---|---|---|---|
| `accessorize` | fit, protect | dodging | | `protectee_album` | `eval_imgs` | |
| `adv_diffusion` | protect | impersonation | | | `eval_imgs`, `target_img` | |
| `adv_inversion` | protect | impersonation | | | `eval_imgs`, `target_img` | |
| `adv_makeup` | train, protect | impersonation (target baked in) | `corpus`, `target_img` | | `eval_imgs` | `ladn` |
| `advcloak` | train, fit, protect | dodging | `corpus` | `protectee_album` | `eval_imgs` | `casia_webface` |
| `advhat` | fit, protect | dodging | | `protectee_album` | `eval_imgs` | |
| `aidpro` | train, protect | dodging | `corpus` | | `eval_imgs`; optional `message` | `vggface2` |
| `amt_gan` | train, protect | impersonation (target baked in) | `corpus`, `target_img` | | `eval_imgs`; optional `makeup_reference_name` | `mt` |
| `apf` | train, fit, protect | dodging | `corpus` | `protectee_album` | `eval_imgs` | `ms1m` |
| `at3d` | protect | impersonation | | | `eval_imgs`, `target_img` | |
| `chameleon` | fit, protect | dodging | | `protectee_album` | `eval_imgs` | |
| `clip2protect` | protect | impersonation | | | `eval_imgs`, `target_img`; optional `prompt` | |
| `dadp` | protect | impersonation | | | `eval_imgs`, `target_img` | |
| `diffaim` | protect | impersonation | | | `eval_imgs`, `target_img` | |
| `diffam` | train, protect | impersonation (target baked in) | `corpus`, `target_img`; optional `makeup_reference_name` | | `eval_imgs` | `celebamask_hq` |
| `divtrackee` | protect | dodging, guided by a target pool | | | `eval_imgs`, `target_pool`; optional `protected_imgs` | |
| `dpa` *(not in AFRBench-basic)* | train, protect | impersonation | `corpus` | | `eval_imgs`, `target_img` | `ms1m` |
| `efficient_decision` *(not in AFRBench-basic)* | protect | dodging (black-box) | | | `eval_imgs`, `reference_imgs` | |
| `falco` | protect | dodging | | | `eval_imgs` | |
| `fawkes` | protect | dodging | | | `eval_imgs` | |
| `fpp_wdp` | protect | impersonation | | | `eval_imgs`, `target_img` | |
| `gift` | protect | impersonation | | | `eval_imgs`, `target_img` | |
| `lowkey` | protect | dodging | | | `eval_imgs` | |
| `opom` | fit, protect | dodging | | `protectee_album` | `eval_imgs` | |
| `protego` | fit, protect | dodging | | `protectee_album` | `eval_imgs` | |
| `tip_im` | protect | impersonation, target chosen from a pool | | | `eval_imgs`, `target_pool` | |

## Surrogates

Methods attack a list of face-recognition models passed as `fr_models`. Most need at
least one; `adv_makeup` needs two, and `aidpro`, `apf`, and `dpa` need as many as their
index arguments refer to. AFRBench-basic uses three FRBench models:
SwinV2-B/ArcFace/Glint360K, SwinV2-B/CenterLoss/MS1M, and ConvNeXtV2-B/ArcFace/MS1M.

## Weights

Method priors (pretrained generators, parsers, encoders) are downloaded on first use to
`~/.afrlib` and are listed in each `afrlib/afr/<method>/assets.py`
(`python -m afrlib status` shows them). Methods with a train stage start from their
training corpus; `save_run(folder)` and `Method.from_run(folder, device)` store and
restore trained or fitted state.

## Papers and code

See the method table in the [README](../README.md) and [`THIRD_PARTY.md`](../THIRD_PARTY.md)
for papers, upstream code, and licenses.
