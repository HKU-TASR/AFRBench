"""OPOM: a person-specific universal privacy mask (convex-hull variant).

Reference Paper: OPOM: Customized Invisible Cloak towards Face Privacy Protection (IEEE TPAMI 2022)
Reference Code: https://github.com/zhongyy/OPOM
"""
import os
from pathlib import Path
from typing import Any, Dict, List, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
import cvxpy as cp

from afrlib.contracts import (
    AlignType,
    ChannelType,
    CropType,
    EvalData,
    FitData,
    ImageSpec,
    StageInputs,
)
from afrlib.afr.base import AFRMethod
from afrlib.fr import FaceRecognizer
from afrlib.utils.face_crop import paste_deltas, prepare_inputs
from afrlib.utils.log import tqdm, get_logger

logger = get_logger(__name__)


class ConvexHullLoss(nn.Module):
    """MSE between L2-normalized protected features and their projection onto the convex hull of
    the clean features, whose coefficients are solved per image (paper Eq. 12) and held fixed.
    """

    def __init__(self):
        super(ConvexHullLoss, self).__init__()
        self.mse = nn.MSELoss()

    def forward(self, prot_feats: torch.Tensor, orig_feats: torch.Tensor, upper: float, lower: float) -> torch.Tensor:
        normalized_prot = F.normalize(prot_feats, p=2, dim=1)
        normalized_orig = F.normalize(orig_feats, p=2, dim=1)
        orig_feats_np = normalized_orig.clone().detach().cpu().numpy()
        coeffs_lst = []
        for img_idx in range(normalized_prot.shape[0]):
            prot_feat_np = normalized_prot[img_idx].clone().detach().cpu().numpy()
            coeff = cp.Variable(orig_feats_np.shape[0])
            objective = cp.Minimize(cp.sum_squares(coeff @ orig_feats_np - prot_feat_np))
            constraints = [sum(coeff) == 1, coeff >= lower, coeff <= upper]
            problem = cp.Problem(objective, constraints)
            problem.solve()
            coeffs_lst.append(torch.tensor(coeff.value, dtype=torch.float32, device=prot_feats.device))
        coeffs = torch.stack(coeffs_lst, dim=0)
        return self.mse(torch.mm(coeffs, normalized_orig), normalized_prot)


class OPOM(AFRMethod):
    """One universal mask per protectee, fitted to push their features outside the convex hull of
    their clean features, and added to every protected photo.

    Args:
        epochs: Fitting epochs over the album.
        batch_size: Album images per update.
        learning_rate: Step size of the signed-gradient update, on the ``[0, 1]`` scale.
        epsilon: L-inf budget of the mask, on the ``[0, 1]`` scale.
        init_epochs: Leading epochs that use the class center instead of the hull.
        hull_upper_bound: Upper bound of the hull coefficients.
        hull_lower_bound: Lower bound of the hull coefficients.
    """

    name = "OPOM"
    fit_inputs = StageInputs(required=("protectee_album",))
    def __init__(
            self,
            device: torch.device,
            fr_models: Sequence[FaceRecognizer] = (),
            *,
            assets_path: "str | os.PathLike[str] | None" = None,
            epochs: int = 100,
            batch_size: int = 4,
            learning_rate: float = 1 / 255.,
            epsilon: float = 16 / 255.,
            init_epochs: int = 10,
            hull_upper_bound: float = 1.0,
            hull_lower_bound: float = 0.0,
            ):
        super().__init__(device, fr_models, assets_path=assets_path)

        self.epochs = epochs
        self.batch_size = batch_size
        # OPOM operates directly in [0, 255] pixel space.
        self.learning_rate = learning_rate * 255.
        self.epsilon = epsilon * 255.
        self.init_epochs = init_epochs
        self.hull_upper_bound = hull_upper_bound
        self.hull_lower_bound = hull_lower_bound

        # Fixed seed for the random mask initialization.
        self.generator = torch.Generator(device=self.device)
        self.generator.manual_seed(114)

        self.hull_loss = ConvexHullLoss().to(self.device)

        # The fitted mask, (1, 3, 224, 224) in [-epsilon, epsilon] on the [0, 255] scale.
        self.univ_mask = None
        self._maybe_load_run()

    def attack_spec(self) -> ImageSpec:
        return ImageSpec(size=224, channel=ChannelType.RGB, crop_type=CropType.FaceBox, align_type=AlignType.NoAlign)

    def save_run(self, save_path: "str | os.PathLike[str]") -> Path:
        """Save the trained privacy-cloak mask ``univ_mask``; no priors are owned."""
        self._require_savable()
        root = Path(save_path).expanduser().resolve()
        root.mkdir(parents=True, exist_ok=True)
        torch.save(self.univ_mask.detach().cpu(), root / "univ_mask.pt")
        return super().save_run(root, files=["univ_mask.pt"])

    def load_run(self, load_path: "str | os.PathLike[str]") -> Path:
        """Load the privacy-cloak mask ``univ_mask`` written by :meth:`save_run`."""
        root = super().load_run(load_path)
        path = root / "univ_mask.pt"
        if not path.is_file():
            raise FileNotFoundError(f"No OPOM mask under {root}")
        self.univ_mask = torch.load(
            path, map_location=self.device, weights_only=True
        ).to(self.device, torch.float32)
        self._fitted = True
        self._mark_ready()
        return root

    def get_config(self) -> Dict[str, Any]:
        """Return the configuration used to initialize this instance."""
        config = super().get_config()
        config.update({
            "epochs": self.epochs,
            "batch_size": self.batch_size,
            "learning_rate": self.learning_rate / 255.,
            "epsilon": self.epsilon / 255.,
            "init_epochs": self.init_epochs,
            "hull_upper_bound": self.hull_upper_bound,
            "hull_lower_bound": self.hull_lower_bound,
        })
        return config

    def fit(self, data: FitData) -> None:
        """Fit the protectee's privacy cloak from their facial images.

        Args:
            data.protectee_album: Images of the same person following ``spec_for``: RGB in
                ``[0, 255]``, shape ``(N, 3, 224, 224)``, ``float32``.
        """
        self._check_inputs("fit", data)
        self._require_fr()
        self._begin_fitting()
        imgs = data.protectee_album
        orig_imgs = imgs.to(self.device)

        self.generator.manual_seed(114)
        univ_mask = (torch.rand(size=(1, 3, 224, 224), device=self.device, generator=self.generator,
                                dtype=torch.float32) - 0.5) * self.epsilon / 0.5

        pbar = tqdm(range(self.epochs), desc="OPOM training")
        for epoch in pbar:
            batch_losses, batch_cos_sims = [], []
            for start in range(0, orig_imgs.shape[0], self.batch_size):
                orig_faces = orig_imgs[start:start + self.batch_size]
                img_num = orig_faces.shape[0]

                protected_faces = torch.clamp(orig_faces + univ_mask.repeat(img_num, 1, 1, 1), 0, 255.)
                protected_faces.requires_grad_(True)

                ensemble_losses, cos_sims = [], []
                for fr in self.fr_models:
                    protected_features = self._embed_one(fr, protected_faces)
                    orig_features = self._embed_one(fr, orig_faces)
                    cos_sims.append(torch.mean(F.cosine_similarity(protected_features, orig_features, dim=1)))
                    if epoch < self.init_epochs:
                        # Class-center warm-up: push features away from the mean first.
                        loss = self.hull_loss(protected_features, orig_features, upper=1 / img_num, lower=1 / img_num)
                    else:
                        loss = self.hull_loss(protected_features, orig_features,
                                              upper=self.hull_upper_bound, lower=self.hull_lower_bound)
                    ensemble_losses.append(loss)
                overall_loss = torch.mean(torch.stack(ensemble_losses, dim=0))
                cos_sim = torch.mean(torch.stack(cos_sims, dim=0), dim=0)
                overall_loss.backward()

                # Gradient ascent on the hull distance, then fold the image deltas back
                # into the universal mask via the batch mean and the L-inf projection.
                grad_avg = torch.mean(protected_faces.grad.detach(), dim=0, keepdim=True)
                protected_faces = protected_faces.detach() + self.learning_rate * torch.sign(grad_avg)
                univ_mask = torch.clamp(torch.mean(protected_faces - orig_faces, dim=0, keepdim=True),
                                        -self.epsilon, self.epsilon)

                batch_losses.append(overall_loss.item())
                batch_cos_sims.append(cos_sim.item())

            pbar.set_description(
                f"OPOM training | Loss: {sum(batch_losses)/len(batch_losses):.4f} | "
                f"Cos Sim: {sum(batch_cos_sims)/len(batch_cos_sims):.4f}")

        self.univ_mask = univ_mask.detach()
        self._mark_ready()
        logger.info("Trained the OPOM mask for %d epochs on %d images.", self.epochs, orig_imgs.shape[0])

    def protect(self, data: EvalData) -> List[torch.Tensor]:
        """Add the fitted mask to each 224 px crop and paste the change back."""
        self._check_inputs("protect", data)
        self._require_ready()
        photos = list(data.eval_imgs)
        crops, plans = prepare_inputs(
            photos, data.eval_geometry, self.attack_spec()
        )
        clean = torch.stack(crops).to(self.device)
        with torch.no_grad():
            masked = torch.clamp(clean + self.univ_mask, 0, 255.)
        return paste_deltas(photos, masked - clean, plans)

    def _embed_one(self, fr: FaceRecognizer, imgs255: torch.Tensor) -> torch.Tensor:
        """Embed ``[0, 255]`` RGB faces with one FR model (pre-cropped, no TTA, raw features)."""
        return fr.forward(imgs255, need_crop=False, tta=(), l2_normalize=False).embeddings
