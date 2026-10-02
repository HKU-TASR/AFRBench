# Third-party code, models, and data

The MIT license in [`LICENSE`](LICENSE) covers the code written for AFRLib and AFRBench.
It does not relicense third-party code. Each method directory and each vendored
component keeps the license of its upstream; where the upstream ships no license file,
the directory has a `NOTICE` and no license is granted for that code. Several
components are restricted to non-commercial or evaluation use (marked below). Check
the terms of every component you use. If you are the author of any included method or
component and want it removed or relicensed, open an issue at
<https://github.com/HKU-TASR/AFRBench/issues>.

## AFR methods (`afrlib/afr/`)

| Method | Paper | Code origin | License |
|---|---|---|---|
| `accessorize` | Accessorize to a Crime (CCS 2016) | [accessorize-to-a-crime](https://github.com/mahmoods01/accessorize-to-a-crime) | no license file (`NOTICE`) |
| `adv_diffusion` | Adv-Diffusion (AAAI 2024) | [Adv-Diffusion](https://github.com/kopperx/Adv-Diffusion) | MIT |
| `adv_inversion` | Adv-Inversion (TIFS 2025) | [Adv-Inversion](https://github.com/BiiiGerrr/Adv-Inversion) | no license file (`NOTICE`) |
| `adv_makeup` | Adv-Makeup (IJCAI 2021) | [Adv-Makeup](https://github.com/TencentYoutuResearch/Adv-Makeup) | Tencent Adv-Makeup license, academic research only (`LICENSE`) |
| `advcloak` | AdvCloak (Pattern Recognition 2025) | [AdvCloak](https://github.com/liuxuannan/AdvCloak) | CC BY-SA 4.0 |
| `advhat` | AdvHat (ICPR 2020) | [advhat](https://github.com/papermsucode/advhat) | MIT |
| `aidpro` | AIDPro (TIFS 2025) | [AIDPro](https://github.com/daizigege/AIDPro) | no license file (`NOTICE`) |
| `amt_gan` | AMT-GAN (CVPR 2022) | [AMT-GAN](https://github.com/CGCL-codes/AMT-GAN) | MIT |
| `apf` | APF (ACM MM 2020) | [APF](https://github.com/adversarial-for-goodness/APF) | GPL-3.0 |
| `at3d` | AT3D (CVPR 2023) | [AT3D](https://github.com/thu-ml/AT3D), renderer from [Deep3DFaceRecon_pytorch](https://github.com/sicxu/Deep3DFaceRecon_pytorch) | MIT, MIT |
| `chameleon` | Chameleon (ECCV 2024) | reimplementation; authors' code: [Chameleon](https://github.com/git-disl/Chameleon) | `NOTICE` |
| `clip2protect` | CLIP2Protect (CVPR 2023) | [Clip2Protect](https://github.com/fahadshamshad/Clip2Protect) | no license file (`NOTICE`) |
| `dadp` | DADP (TIFS 2025) | [DADP](https://github.com/LiYuanBoJNU/DADP) | no license file (`NOTICE`) |
| `diffaim` | DiffAIM (ACM MM 2025) | [DiffAIM](https://github.com/Liqin-Wang/DiffAIM) | MIT |
| `diffam` | DiffAM (CVPR 2024) | [DiffAM](https://github.com/HansSunY/DiffAM) | no license file (`NOTICE`) |
| `divtrackee` | DivTrackee (CCS 2025) | [divtrackee](https://github.com/fiora6/divtrackee) | no license file (`NOTICE`) |
| `dpa` | DPA (CVPR 2025) | [DPA](https://github.com/fengfanzhou/DPA) | MIT |
| `efficient_decision` | Efficient Decision-based Attack (CVPR 2019) | [efficient-decision-based-black-box-adversarial-attacks-on-face-recognition](https://github.com/sgmath12/efficient-decision-based-black-box-adversarial-attacks-on-face-recognition) | no license file (`NOTICE`) |
| `falco` | FALCO (CVPR 2023) | [FALCO](https://github.com/chi0tzp/FALCO) | Apache-2.0 |
| `fawkes` | Fawkes (USENIX Security 2020) | [fawkes](https://github.com/Shawn-Shan/fawkes) | BSD-3-Clause |
| `fpp_wdp` | FPP-WDP (CVPR 2025) | [Facial-Privacy-Protection](https://github.com/parham1998/Facial-Privacy-Protection) | no license file (`NOTICE`) |
| `gift` | GIFT (ACM MM 2024) | reimplementation from the paper, with BDInvert code | **AGPL-3.0, whole directory** (`LICENSE`, `NOTICE`) |
| `lowkey` | LowKey (ICLR 2021) | code in the paper's [supplementary material](https://openreview.net/forum?id=hJmtwocEqzc) | no license file (`NOTICE`) |
| `opom` | OPOM (TPAMI 2022) | [OPOM](https://github.com/zhongyy/OPOM) | MIT |
| `protego` | Protego (CVPR 2026) | [Protego](https://github.com/HKU-TASR/Protego) | MIT (this repo's `LICENSE`; see `NOTICE`) |
| `tip_im` | TIP-IM (ICCV 2021) | [TIP-IM](https://github.com/ShawnXYang/TIP-IM) | Apache-2.0 |

## Vendored components

| Path | Upstream | License |
|---|---|---|
| `afrlib/afr/adv_diffusion/ldm/` | [Latent Diffusion](https://github.com/CompVis/latent-diffusion), with MiDaS and BSRGAN helpers | MIT (`ldm/LICENSE`) |
| `afrlib/afr/adv_diffusion/face_parser.py` | FaceParseNet50 from [FaceParsing.PyTorch](https://github.com/TracelessLe/FaceParsing.PyTorch) | MIT (`adv_diffusion/LICENSE_FaceParsing`) |
| `afrlib/afr/adv_inversion/models/stylegan2/` | [stylegan2-pytorch](https://github.com/rosinality/stylegan2-pytorch); `dnnlib/`, `torch_utils/` and CUDA ops from [NVIDIA StyleGAN2-ADA](https://github.com/NVlabs/stylegan2-ada-pytorch) | MIT (`stylegan2/LICENSE`); NVIDIA Source Code License, non-commercial (`stylegan2/LICENSE_NVIDIA`) |
| `afrlib/afr/adv_inversion/models/` (encoder) | FeatureStyleEncoder (InterDigital) | Limited Software Evaluation License (`LICENSE_FeatureStyleEncoder.md`) |
| `afrlib/afr/{clip2protect,divtrackee}/clip/` | [OpenAI CLIP](https://github.com/openai/CLIP) | MIT (`clip/LICENSE`) |
| `afrlib/afr/{clip2protect,divtrackee,falco}/models/` | [encoder4editing](https://github.com/omertov/encoder4editing), [stylegan2-pytorch](https://github.com/rosinality/stylegan2-pytorch); the ArcFace backbone follows [InsightFace_Pytorch](https://github.com/TreB1eN/InsightFace_Pytorch) | MIT (`encoders/LICENSE`, `encoders/LICENSE_InsightFace_Pytorch`, `stylegan2/LICENSE`); CUDA ops under the NVIDIA Source Code License, non-commercial (`stylegan2/LICENSE_NVIDIA`) |
| `afrlib/afr/falco/models/arcface.py` | ArcFace IR-SE50 from [InsightFace_Pytorch](https://github.com/TreB1eN/InsightFace_Pytorch) | MIT (`falco/models/LICENSE_InsightFace_Pytorch`) |
| `afrlib/afr/dadp/diffae/` | [Diffusion Autoencoders](https://github.com/phizaz/diffae) | MIT (`diffae/LICENSE`) |
| `afrlib/afr/{falco,gift}/models/genforce/` | [GenForce](https://github.com/genforce/genforce) | MIT (`genforce/LICENSE`) |
| `afrlib/afr/gift/models/bisenet/`, `afrlib/afr/amt_gan/face_parser.py` | [face-parsing.PyTorch](https://github.com/zllrunning/face-parsing.PyTorch) | MIT (`bisenet/LICENSE`, `amt_gan/LICENSE_face_parsing`) |
| `afrlib/afr/gift/models/basecode_encoder.py`, `afrlib/afr/gift/models/genforce/stylegan2_generator.py` | [BDInvert](https://github.com/kkang831/BDInvert_Release) | AGPL-3.0 (`LICENSE_BDInvert`) |
| `afrlib/afr/protego/smirk/` | [SMIRK](https://github.com/georgeretsi/smirk), including its FLAME masks, template and landmark files | MIT (`smirk/LICENSE`) |
| `afrbench/fr_system/diffjpeg/` | [DiffJPEG](https://github.com/mlomnitz/DiffJPEG) | MIT |
| `afrbench/fr_system/restore/registry.py` | stubs for the vendored restorers; one init function from [BasicSR](https://github.com/xinntao/BasicSR) | Apache-2.0 (see upstream) |
| `afrbench/fr_system/restore/codeformer/` | [CodeFormer](https://github.com/sczhou/CodeFormer) | S-Lab License 1.0 (non-commercial) |
| `afrbench/fr_system/restore/gfpgan/` | [GFPGAN](https://github.com/TencentARC/GFPGAN) | see `LICENSE` |
| `afrbench/fr_system/restore/diffpure/` | [DiffPure](https://github.com/NVlabs/DiffPure), DDPM UNet | NVIDIA Source Code License (non-commercial); MIT |

## Models and data downloaded at run time

Pretrained weights are downloaded from their original sources, listed in each
`afrlib/afr/<method>/assets.py` and `afrbench/assets.py`, and remain under their
authors' terms. Several are restricted to research or evaluation use: the FFHQ
StyleGAN2 generator checkpoints (NVIDIA), the InsightFace models (the buffalo_l
landmark regressor, the SCRFD detector, and the ArcFace backbones), CodeFormer's
restoration net (S-Lab License 1.0), and the metric weights `pyiqa` downloads.
FaRL's face-representation weights ship without a license file upstream. FR models
come from [FRBench](https://github.com/HKU-TASR/FRBench).
Protego's FLAME 2020 model is not distributed; obtain it under the
[FLAME license](https://flame.is.tue.mpg.de). AT3D's `BFM_model_front_patch.mat`,
downloaded from the AT3D authors' link, contains Basel Face Model 2009 data under the
[BFM license](https://faces.dmi.unibas.ch/bfm/main.php?nav=1-2&id=downloads).
Training corpora (MS1M, CASIA-WebFace, VGGFace2, MT, LADN, CelebAMask-HQ) and the
[MillionCelebs](https://buptzyb.github.io/MillionCelebs/) distractors are subject to
their own terms. The five HQ target images in `data/afrbench-basic/targets/hq/` are
from CelebA-HQ, which is provided for non-commercial research use. AFRBench-basic
protectee images are not distributed; see [`docs/data.md`](docs/data.md).

Build dependencies installed by `setup.sh` keep their own licenses; note that
[nvdiffrast](https://github.com/NVlabs/nvdiffrast) (needed by AT3D and Protego) is
under the NVIDIA Source Code License for research or evaluation use.
