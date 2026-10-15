# The AFRBench-basic protocol

AFRBench measures how well a protection method keeps a person's photos from being
retrieved by a face-recognition search engine, and how much it changes the photos.
This page describes the protocol on AFRBench-basic; [AFRBench-DF](#afrbench-df) applies
it to video frames.

## Data

- **Protectees.** 50 people (25 actors, 25 actresses), 100 photos each, split into
  roles that are fixed in the dataset: 50 **album**, 25 **query**, and 25 **gallery**
  photos. Photos taken on the same day never appear in two different roles.
- **Album** photos are what a method may learn from (its fit stage). **Query** and
  **gallery** photos are the ones it protects and the ones that are scored.
- **Targets.** Impersonation methods need a target identity. There are five HQ targets
  (one CelebA-HQ image each) and a pool of 2,000 further identities from MillionCelebs.
  Each protectee is assigned one HQ target (five groups of ten, each with five actors
  and five actresses).
- **Distractors.** The search gallery also contains all of MillionCelebs: 22.8M face
  images of 719,563 identities, after removing the 50 classes that are the protectees.
  Target identities stay in it as ordinary distractors.
- **Training corpora.** Methods that train (AdvCloak, AIDPro, APF, Adv-Makeup, AMT-GAN,
  DiffAM) use their papers' corpora with the protectees removed.

See [data.md](data.md) for what the repository ships and how to obtain the rest.

## Protection

Every method runs with its default hyper-parameters, seed 0, and the same attack
surrogates: FRBench SwinV2-B/ArcFace/Glint360K, SwinV2-B/CenterLoss/MS1M, and
ConvNeXtV2-B/ArcFace/MS1M.

- A method with a **train** stage is trained once, or once per HQ target when the
  target is baked into its weights (Adv-Makeup, AMT-GAN, DiffAM).
- A method with a **fit** stage is fitted per protectee on that person's 50 album photos.
- **protect** runs per protectee on the 50 query and gallery photos. Each photo is
  processed as the square frame of side `ceil(2 x face size)` around the face, and the
  output is that frame, protected, as float32 in `[0, 255]`.
- Impersonation methods protect each protectee against its assigned target;
  pool-based methods (TIP-IM, DivTrackee) receive a seeded draw of pool identities.

The 24 methods of the protocol are all built-in methods except EfficientDecision and
DPA.

## Evaluation

An **FR system** is an FR model plus the intruder's image pipeline: optional
quantization, JPEG, and resizing; RetinaFace-ResNet50 detection; ArcFace alignment to
112x112; an optional purifier; optional test-time flip augmentation. The default system
is FRBench IRSE-100/ArcFace/MS1M, a model that none of the methods attacks, with no
pipeline operations (`minimal`).

The pipeline runs in a fixed order: quantize, JPEG, resize, then one detection pass,
then alignment and embedding. A purifier runs after detection: the model restorers
(GFPGAN, CodeFormer, DiffPure) work on a 5-point warp to FFHQ-512 (256 for DiffPure) whose
canvas is then composed to ArcFace-112, while the median filter runs on the aligned 112
crop. Every optional stage can be switched off, and the config lists them all, so the YAML
is the system's full provenance.

Protectee photos, protected outputs and the HQ targets go through that whole pipeline. The
MillionCelebs distractors are already aligned 112-px crops, so only the pre-detection ops
apply to them, and the gallery they form is embedded with purification off (and quantization
off where JPEG and flip augmentation are also off). So for one FR model the purification and
distractor-set jobs all share a single distractor cache, while `jpeg`, `tta` and
`tta_jpeg_quantize` keep their own, and each white-box model has its own.

Each query photo is searched against the gallery (the protectees' gallery photos plus
the distractors) in four scenarios:

| Scenario | Query | Gallery | Question |
|---|---|---|---|
| 1A | clean | protected | Is a person still found when their uploaded photos are protected? |
| 1B | clean | clean | Baseline without protection |
| 2A | protected | protected | Can a protected photo find other protected photos? |
| 2B | protected | clean | Can a protected photo find older, unprotected photos? |

Two sweeps interpolate between them over `r` in 0.0, 0.1, ..., 1.0: `mix-gallery-r`
protects a fraction `r` of each person's gallery photos (protected query), and
`mix-query-r` protects a fraction `r` of the queries (protected gallery). Which images are
protected is chosen per image from a seed on its id, so the sweeps are nested; `r = 0` and
`r = 1` of the two sweeps coincide with 2B/2A and 1A/2A.

**Metrics** (averaged over the 1,250 queries):

- `recall@25` (primary): the fraction of a person's 25 gallery photos found in the top
  25 results, also reported as `recall@K(user)`. Lower means better protection.
- `recall@100`, `AP`, `AP@100`, `NDCG@10`.
- Identification: `rank{1,5,10}_{appearance,plurality,majority}`, whether the person
  appears in, is the most frequent, or is the majority identity of the top K results.
  Plurality ties go to the first of the tied labels in the top K, and majority means a
  strict majority of the K results.
- A query whose face is not detected counts as not retrieved; an undetected gallery
  photo is removed from the gallery but still counts in the denominator.
- A gallery photo's rank is one plus the number of photos strictly more similar to the
  query, so a tie does not outrank the person's own photo.
- Visual quality of protected against clean photos: PSNR, SSIM, MS-SSIM, LPIPS,
  BRISQUE, NIQE, and per-method FID, computed on the whole frame (without padding
  beyond the photo's border) after rounding to uint8. FID compares the frames resized
  to 299x299 (bicubic) and needs at least 50 images.

Impersonation success is not scored; every method is evaluated on how well it prevents
the protectee from being found.

Results are filed by run. The baseline 1B does not depend on the method and is filed
once, as `no_protection`. A method that runs once per HQ target protects each target's
ten protectees in a separate run; those runs are scored as one ranking over all 50
protectees. A result is tagged `seen` when the evaluation FR model is one of the
method's attack surrogates.

## Evaluation suite

`afrbench.benchmark --evals` selects jobs from the 17-job suite:

| Job | FR system |
|---|---|
| `minimal` | IRSE-100, no pipeline operations (default; also computes visual quality) |
| `whitebox_swinv2b_arcface_glint360k`, `whitebox_swinv2b_centerloss_ms1m`, `whitebox_convnextv2b_arcface_ms1m` | each attack surrogate as the FR model |
| `quantize`, `jpeg`, `tta`, `tta_jpeg_quantize` | uint8 quantization, DiffJPEG quality 75, flip test-time augmentation, and all three |
| `median`, `gfpgan`, `codeformer` | purification of query and protectee photos before recognition |
| `mc_no_targets`, `mc_drop_2005`, `mc_ids_{500000,100000,20000,4000}` | variants of the distractor set |

## AFRBench-DF

AFRBench-DF runs this protocol unchanged on frames of the DeeperForensics-1.0 source
videos ([data.md](data.md#afrbench-df)); only the data and the reporting differ.

- **Protectees and photos.** A protectee is an actor and a photo is a decoded video frame
  of 1920x1080 pixels, processed as the same square frame around the face. There are 50
  actors, 25 men and 25 women, with 100 frames each: 50 album, 25 query, and 25 gallery
  frames, every one from a different clip.
- **Roles are disjoint by clip.** No clip feeds two roles of one actor, and within a role the
  clips are balanced over camera, emotion, and light. Each actor was recorded in one session,
  so, unlike in AFRBench-basic, the roles share clothing, hairstyle, and background; they
  differ in viewpoint, expression, and lighting.
- **Targets, distractors, protection, evaluation.** As above: the same HQ targets and
  pool, MillionCelebs distractors, attack surrogates, FR systems, evaluation suite,
  scenarios, and metrics.
- **Labels.** Every frame carries the camera and emotion of its clip.
  `python -m afrbench.groups` averages the per-query metrics of any finished eval within
  each value of these labels, so a method can be compared across viewpoints and
  expressions. Poses that are hard for the detector or the FR model count against the
  method like any other failure: a query whose face is not detected counts as not retrieved.
