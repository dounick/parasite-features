"""
featurizers.py
==============
Utility classes for defining *invertible* feature spaces on top of a model’s
hidden-state tensors, together with intervention helpers that operate inside
those spaces.

Key ideas
---------

* **Featurizer** – a lightweight wrapper holding:
    • a forward `featurizer` module that maps a tensor **x → (f, error)**
      where *error* is the reconstruction residual (useful for lossy
      featurizers such as sparse auto-encoders);
    • an `inverse_featurizer` that re-assembles the original space
      **(f, error) → x̂**.

* **Interventions** – three higher-order factory functions build PyVENE
  interventions that work in the featurized space:
    - *interchange*
    - *collect*
    - *mask* (differential binary masking)

All public classes / functions below carry PEP-257-style doc-strings.
"""

from typing import Optional, Tuple

import torch
import torch.nn as nn
import pyvene as pv
import json
import random
import numpy as np
import os

# --------------------------------------------------------------------------- #
#  Basic identity featurizers                                                 #
# --------------------------------------------------------------------------- #
class IdentityFeaturizerModule(torch.nn.Module):
    """A no-op featurizer: *x → (x, None)*."""

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, None]:
        return x, None


class IdentityInverseFeaturizerModule(torch.nn.Module):
    """Inverse of :class:`IdentityFeaturizerModule`."""

    def forward(self, x: torch.Tensor, error: None) -> torch.Tensor:  # noqa: D401
        return x


# --------------------------------------------------------------------------- #
#  High-level Featurizer wrapper                                              #
# --------------------------------------------------------------------------- #
class Featurizer:
    """Container object holding paired featurizer and inverse modules.

    Parameters
    ----------
    featurizer :
        A `torch.nn.Module` mapping **x → (features, error)**.
    inverse_featurizer :
        A `torch.nn.Module` mapping **(features, error) → x̂**.
    n_features :
        Dimensionality of the feature space.  **Required** when you intend to
        build a *mask* intervention; optional otherwise.
    id :
        Human-readable identifier used by `__str__` methods of the generated
        interventions.
    """

    # --------------------------------------------------------------------- #
    #  Construction / public accessors                                      #
    # --------------------------------------------------------------------- #
    def __init__(
        self,
        featurizer: torch.nn.Module = IdentityFeaturizerModule(),
        inverse_featurizer: torch.nn.Module = IdentityInverseFeaturizerModule(),
        *,
        n_features: Optional[int] = None,
        id: str = "null",
    ):
        self.featurizer = featurizer
        self.inverse_featurizer = inverse_featurizer
        self.n_features = n_features
        self.id = id

    # -------------------- Intervention builders -------------------------- #
    def get_interchange_intervention(self):
        if not hasattr(self, "_interchange_intervention"):
            self._interchange_intervention = build_feature_interchange_intervention(
                self.featurizer, self.inverse_featurizer, self.id
            )
        return self._interchange_intervention

    def get_collect_intervention(self):
        if not hasattr(self, "_collect_intervention"):
            self._collect_intervention = build_feature_collect_intervention(
                self.featurizer, self.id
            )
        return self._collect_intervention

    def get_mask_intervention(self):
        if self.n_features is None:
            raise ValueError(
                "`n_features` must be provided on the Featurizer "
                "to construct a mask intervention."
            )
        if not hasattr(self, "_mask_intervention"):
            self._mask_intervention = build_feature_mask_intervention(
                self.featurizer,
                self.inverse_featurizer,
                self.n_features,
                self.id,
            )
        return self._mask_intervention

    # ------------------------- Convenience I/O --------------------------- #
    def featurize(self, x: torch.Tensor):
        return self.featurizer(x)

    def inverse_featurize(self, x: torch.Tensor, error):
        return self.inverse_featurizer(x, error)

    # --------------------------------------------------------------------- #
    #  (De)serialisation helpers                                            #
    # --------------------------------------------------------------------- #
    def save_modules(self, path: str) -> Tuple[str, str]:
        """Serialise featurizer & inverse to `<path>_{featurizer, inverse}`.

        Notes
        -----
        * **SAE featurizers** are *not* serialisable: a
          :class:`NotImplementedError` is raised.
        * Existing files will be *silently overwritten*.
        """
        featurizer_class = self.featurizer.__class__.__name__

        if featurizer_class == "SAEFeaturizerModule":
            #SAE featurizers are to be loaded from sae_lens
            return None, None

        inverse_featurizer_class = self.inverse_featurizer.__class__.__name__

        # Extra config needed for Subspace featurizers
        additional_config = {}
        if featurizer_class == "SubspaceFeaturizerModule":
            additional_config["rotation_matrix"] = (
                self.featurizer.rotate.weight.detach().clone()
            )
            additional_config["requires_grad"] = (
                self.featurizer.rotate.weight.requires_grad
            )
        #Denis:changed
        ###############################################################
        elif featurizer_class == "SubspaceFeaturizerModule_RevNet":
            additional_config["architectureshape"] = self.featurizer.shape
            additional_config["state_dict"] = self.featurizer.RevNet.state_dict()
            additional_config["requires_grad"] = self.featurizer.trainable


        elif featurizer_class == "SubspaceFeaturizerModule_RevNet_SufficientExperiment":
            additional_config["architectureshape"] = self.featurizer.shape
            additional_config["state_dict"] = self.featurizer.RevNet.state_dict()
            additional_config["requires_grad"] = self.featurizer.trainable
            additional_config["Suff_RS"] = self.featurizer.Suff_RS
            additional_config["rotation_dims"] = self.featurizer.Suff_dims


        elif featurizer_class == "SubspaceFeaturizerModule_RotationRevNet":
            additional_config["architectureshape"] = self.featurizer.shape
            additional_config["state_dict_RN"] = self.featurizer.RevNet.state_dict()
            additional_config["state_dict_Ro"] = self.featurizer.rotate.state_dict()
            additional_config["requires_grad"] = self.featurizer.trainable


        elif featurizer_class == "SubspaceFeaturizerModule_RotationRevNet_SufficientExperiment":
            additional_config["architectureshape"] = self.featurizer.shape
            additional_config["state_dict_RN"] = self.featurizer.RevNet.state_dict()
            additional_config["state_dict_Ro"] = self.featurizer.rotate.state_dict()
            additional_config["requires_grad"] = self.featurizer.trainable
            additional_config["Suff_RS"] = self.featurizer.Suff_RS
            additional_config["rotation_dims"] = self.featurizer.Suff_dims


        elif featurizer_class == "SubspaceFeaturizerModule_Linear":
            additional_config["architectureshape"] = self.featurizer.shape
            additional_config["state_dict_Li"] = self.featurizer.Li.state_dict()
            additional_config["requires_grad"] = self.featurizer.trainable


        elif featurizer_class == "SubspaceFeaturizerModule_RotationLinear":
            additional_config["architectureshape"] = self.featurizer.shape
            additional_config["state_dict_Li"] = self.featurizer.Li.state_dict()
            additional_config["state_dict_Ro"] = self.featurizer.rotate.state_dict()
            additional_config["requires_grad"] = self.featurizer.trainable


        elif featurizer_class == "SubspaceFeaturizerModule_RotationLinear_SufficientExperiment":
            additional_config["architectureshape"] = self.featurizer.shape
            additional_config["state_dict_Li"] = self.featurizer.Li.state_dict()
            additional_config["state_dict_Ro"] = self.featurizer.rotate.state_dict()
            additional_config["requires_grad"] = self.featurizer.trainable
            additional_config["Suff_RS"] = self.featurizer.Suff_RS
            additional_config["rotation_dims"] = self.featurizer.Suff_dims


        ###############################################################

        model_info = {
            "featurizer_class": featurizer_class,
            "inverse_featurizer_class": inverse_featurizer_class,
            "n_features": self.n_features,
            "additional_config": additional_config,
        }

        torch.save(
            {"model_info": model_info, "state_dict": self.featurizer.state_dict()},
            f"{path}_featurizer",
        )
        torch.save(
            {
                "model_info": model_info,
                "state_dict": self.inverse_featurizer.state_dict(),
            },
            f"{path}_inverse_featurizer",
        )
        return f"{path}_featurizer", f"{path}_inverse_featurizer"

    @classmethod
    def load_modules(cls, path: str) -> "Featurizer":
        """Inverse of :meth:`save_modules`.

        Returns
        -------
        Featurizer
            A *new* instance with reconstructed modules and metadata.
        """
        featurizer_data = torch.load(f"{path}_featurizer")
        inverse_data = torch.load(f"{path}_inverse_featurizer")

        model_info = featurizer_data["model_info"]
        featurizer_class = model_info["featurizer_class"]

        if featurizer_class == "SubspaceFeaturizerModule":
            rot = model_info["additional_config"]["rotation_matrix"]
            requires_grad = model_info["additional_config"]["requires_grad"]

            # Re-build a parametrised orthogonal layer with identical shape.
            in_dim, out_dim = rot.shape
            rotate_layer = pv.models.layers.LowRankRotateLayer(
                in_dim, out_dim, init_orth=False
            )
            rotate_layer.weight.data.copy_(rot)
            rotate_layer = torch.nn.utils.parametrizations.orthogonal(rotate_layer)
            rotate_layer.requires_grad_(requires_grad)

            featurizer = SubspaceFeaturizerModule(rotate_layer)
            inverse = SubspaceInverseFeaturizerModule(rotate_layer)

            # Sanity-check weight shape
            assert (
                featurizer.rotate.weight.shape == rot.shape
            ), "Rotation-matrix shape mismatch after deserialisation."

        #Denis:changed
        #################################################################################
        elif featurizer_class == "SubspaceFeaturizerModule_RevNet":
            trainable = model_info["additional_config"]["requires_grad"]

            # Re-build a parametrised orthogonal layer with identical shape.
            shape=model_info["additional_config"]["architectureshape"]
            RN = RevNet(shape[0], shape[1], shape[2])
            RN.load_state_dict(model_info["additional_config"]["state_dict"])
            RN.requires_grad_(trainable)

            featurizer = SubspaceFeaturizerModule_RevNet(RN,shape,trainable)
            inverse = SubspaceInverseFeaturizerModule_RevNet(RN,shape,trainable)


        elif featurizer_class == "SubspaceFeaturizerModule_RevNet_SufficientExperiment":
            trainable = model_info["additional_config"]["requires_grad"]
            Suff_dims=model_info["additional_config"]["rotation_dims"]
            # Re-build a parametrised orthogonal layer with identical shape.
            shape=model_info["additional_config"]["architectureshape"]
            RN = RevNet(shape[0], shape[1]-Suff_dims, shape[2])
            RN.load_state_dict(model_info["additional_config"]["state_dict"])
            RN.requires_grad_(trainable)

            if "Suff_RS" in model_info["additional_config"]:
                Suff_RS=model_info["additional_config"]["Suff_RS"]
            else:
                Suff_RS=None


            featurizer = SubspaceFeaturizerModule_RevNet_SufficientExperiment(RN,shape,trainable,Suff_RS,Suff_dims)
            inverse = SubspaceInverseFeaturizerModule_RevNet_SufficientExperiment(RN,shape,trainable,Suff_RS,Suff_dims)



        elif featurizer_class == "SubspaceFeaturizerModule_RotationRevNet":
            trainable = model_info["additional_config"]["requires_grad"]

            # Re-build a parametrised orthogonal layer with identical shape.
            shape=model_info["additional_config"]["architectureshape"]
            RN = RevNet(shape[0], shape[3], shape[2])
            RN.load_state_dict(model_info["additional_config"]["state_dict_RN"])
            RN.requires_grad_(trainable)
            rotate = pv.models.layers.LowRankRotateLayer(shape[1],shape[3] , init_orth=True)
            rotate = torch.nn.utils.parametrizations.orthogonal(rotate)
            rotate.load_state_dict(model_info["additional_config"]["state_dict_Ro"])
            rotate.requires_grad_(trainable)

            featurizer = SubspaceFeaturizerModule_RotationRevNet(rotate,RN,shape,trainable)
            inverse = SubspaceInverseFeaturizerModule_RotationRevNet(rotate,RN,shape,trainable)


        elif featurizer_class == "SubspaceFeaturizerModule_RotationRevNet_SufficientExperiment":
            trainable = model_info["additional_config"]["requires_grad"]
            Suff_dims=model_info["additional_config"]["rotation_dims"]
            # Re-build a parametrised orthogonal layer with identical shape.
            shape=model_info["additional_config"]["architectureshape"]
            RN = RevNet(shape[0], shape[3], shape[2])
            RN.load_state_dict(model_info["additional_config"]["state_dict_RN"])
            RN.requires_grad_(trainable)
            rotate = pv.models.layers.LowRankRotateLayer(shape[1]-Suff_dims,shape[3] , init_orth=True)
            rotate = torch.nn.utils.parametrizations.orthogonal(rotate)
            rotate.load_state_dict(model_info["additional_config"]["state_dict_Ro"])
            rotate.requires_grad_(trainable)

            if "Suff_RS" in model_info["additional_config"]:
                Suff_RS=model_info["additional_config"]["Suff_RS"]
            else:
                Suff_RS=None

            featurizer = SubspaceFeaturizerModule_RotationRevNet_SufficientExperiment(rotate,RN,shape,trainable,Suff_RS,Suff_dims)
            inverse = SubspaceInverseFeaturizerModule_RotationRevNet_SufficientExperiment(rotate,RN,shape,trainable,Suff_RS,Suff_dims)


        elif featurizer_class == "SubspaceFeaturizerModule_Linear":
            trainable = model_info["additional_config"]["requires_grad"]
            shape=model_info["additional_config"]["architectureshape"]
            Li = InvertibleLinearStable(shape[0])
            Li.load_state_dict(model_info["additional_config"]["state_dict_Li"])
            Li.requires_grad_(trainable)

            featurizer = SubspaceFeaturizerModule_Linear(Li,shape,trainable)
            inverse = SubspaceInverseFeaturizerModule_Linear(Li,shape,trainable)


        elif featurizer_class == "SubspaceFeaturizerModule_RotationLinear":
            trainable = model_info["additional_config"]["requires_grad"]

            # Re-build a parametrised orthogonal layer with identical shape.
            shape=model_info["additional_config"]["architectureshape"]
            Li = InvertibleLinearStable(shape[1])
            Li.load_state_dict(model_info["additional_config"]["state_dict_Li"])
            Li.requires_grad_(trainable)
            rotate = pv.models.layers.LowRankRotateLayer(shape[0],shape[1] , init_orth=True)
            rotate = torch.nn.utils.parametrizations.orthogonal(rotate)
            rotate.load_state_dict(model_info["additional_config"]["state_dict_Ro"])
            rotate.requires_grad_(trainable)


            featurizer = SubspaceFeaturizerModule_RotationLinear(rotate,Li,shape,trainable)
            inverse = SubspaceInverseFeaturizerModule_RotationLinear(rotate,Li,shape,trainable)


        elif featurizer_class == "SubspaceFeaturizerModule_RotationLinear_SufficientExperiment":
            trainable = model_info["additional_config"]["requires_grad"]
            Suff_dims=model_info["additional_config"]["rotation_dims"]
            # Re-build a parametrised orthogonal layer with identical shape.
            shape=model_info["additional_config"]["architectureshape"]
            Li = InvertibleLinearStable(shape[1])
            Li.load_state_dict(model_info["additional_config"]["state_dict_Li"])
            Li.requires_grad_(trainable)
            rotate = pv.models.layers.LowRankRotateLayer(shape[0]-Suff_dims,shape[1] , init_orth=True)
            rotate = torch.nn.utils.parametrizations.orthogonal(rotate)
            rotate.load_state_dict(model_info["additional_config"]["state_dict_Ro"])
            rotate.requires_grad_(trainable)

            if "Suff_RS" in model_info["additional_config"]:
                Suff_RS=model_info["additional_config"]["Suff_RS"]
            else:
                Suff_RS=None

            featurizer = SubspaceFeaturizerModule_RotationLinear_SufficientExperiment(rotate,Li,shape,trainable,Suff_RS,Suff_dims)
            inverse = SubspaceInverseFeaturizerModule_RotationLinear_SufficientExperiment(rotate,Li,shape,trainable,Suff_RS,Suff_dims)

        #################################################################################
        elif featurizer_class == "IdentityFeaturizerModule":
            featurizer = IdentityFeaturizerModule()
            inverse = IdentityInverseFeaturizerModule()
        else:
            raise ValueError(f"Unknown featurizer class '{featurizer_class}'.")

        featurizer.load_state_dict(featurizer_data["state_dict"])
        inverse.load_state_dict(inverse_data["state_dict"])

        return cls(
            featurizer,
            inverse,
            n_features=model_info["n_features"],
            id=model_info.get("featurizer_id", "loaded"),
        )


# --------------------------------------------------------------------------- #
#  Intervention factory helpers                                               #
# --------------------------------------------------------------------------- #
def build_feature_interchange_intervention(
    featurizer: torch.nn.Module,
    inverse_featurizer: torch.nn.Module,
    featurizer_id: str,
):
    """Return a class implementing PyVENE’s TrainableIntervention."""

    class FeatureInterchangeIntervention(
        pv.TrainableIntervention, pv.DistributedRepresentationIntervention
    ):
        """Swap features between *base* and *source* in the featurized space."""

        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self._featurizer = featurizer
            self._inverse = inverse_featurizer

        def forward(self, base, source, subspaces=None):
            f_base, base_err = self._featurizer(base)
            f_src, _ = self._featurizer(source)
            #print("h1:",subspaces)
            if subspaces is None or _subspace_is_all_none(subspaces):
                f_out = f_src
            else:
                f_out = pv.models.intervention_utils._do_intervention_by_swap(
                    f_base,
                    f_src,
                    "interchange",
                    self.interchange_dim,
                    subspaces,
                    subspace_partition=self.subspace_partition,
                    use_fast=self.use_fast,
                )
            return self._inverse(f_out, base_err).to(base.dtype)

        def __str__(self):  # noqa: D401
            return f"FeatureInterchangeIntervention(id={featurizer_id})"

    return FeatureInterchangeIntervention


def build_feature_collect_intervention(
    featurizer: torch.nn.Module, featurizer_id: str
):
    """Return a `CollectIntervention` operating in feature space."""

    class FeatureCollectIntervention(pv.CollectIntervention):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self._featurizer = featurizer

        def forward(self, base, source=None, subspaces=None):
            f_base, _ = self._featurizer(base)
            return pv.models.intervention_utils._do_intervention_by_swap(
                f_base,
                source,
                "collect",
                self.interchange_dim,
                subspaces,
                subspace_partition=self.subspace_partition,
                use_fast=self.use_fast,
            )

        def __str__(self):  # noqa: D401
            return f"FeatureCollectIntervention(id={featurizer_id})"

    return FeatureCollectIntervention


def build_feature_mask_intervention(
    featurizer: torch.nn.Module,
    inverse_featurizer: torch.nn.Module,
    n_features: int,
    featurizer_id: str,
):
    """Return a trainable mask intervention."""

    class FeatureMaskIntervention(pv.TrainableIntervention):
        """Differential-binary masking in the featurized space."""

        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self._featurizer = featurizer
            self._inverse = inverse_featurizer

            # Learnable parameters
            self.mask = torch.nn.Parameter(torch.zeros(n_features), requires_grad=True)
            self.temperature: Optional[torch.Tensor] = None  # must be set by user

        # -------------------- API helpers -------------------- #
        def get_temperature(self) -> torch.Tensor:
            if self.temperature is None:
                raise ValueError("Temperature has not been set.")
            return self.temperature

        def set_temperature(self, temp: float | torch.Tensor):
            self.temperature = (
                torch.as_tensor(temp, dtype=self.mask.dtype).to(self.mask.device)
            )

        # ------------------------- forward ------------------- #
        def forward(self, base, source, subspaces=None):
            if self.temperature is None:
                raise ValueError("Cannot run forward without a temperature.")

            f_base, base_err = self._featurizer(base)
            f_src, _ = self._featurizer(source)

            # Align devices / dtypes
            mask = self.mask.to(f_base.device)
            temp = self.temperature.to(f_base.device)

            f_base = f_base.to(mask.dtype)
            f_src = f_src.to(mask.dtype)

            if self.training:
                gate = torch.sigmoid(mask / temp)
            else:
                gate = (torch.sigmoid(mask) > 0.5).float()

            f_out = (1.0 - gate) * f_base + gate * f_src
            return self._inverse(f_out.to(base.dtype), base_err).to(base.dtype)

        # ---------------- Sparsity regulariser --------------- #
        def get_sparsity_loss(self) -> torch.Tensor:
            if self.temperature is None:
                raise ValueError("Temperature has not been set.")
            gate = torch.sigmoid(self.mask / self.temperature)
            return torch.norm(gate, p=1)

        def __str__(self):  # noqa: D401
            return f"FeatureMaskIntervention(id={featurizer_id})"

    return FeatureMaskIntervention


# --------------------------------------------------------------------------- #
#  Concrete featurizer implementations                                        #
# --------------------------------------------------------------------------- #
class SubspaceFeaturizerModule(torch.nn.Module):
    """Linear projector onto an orthogonal *rotation* sub-space."""

    def __init__(self, rotate_layer: pv.models.layers.LowRankRotateLayer):
        super().__init__()
        self.rotate = rotate_layer

    def forward(self, x: torch.Tensor):
        r = self.rotate.weight.T  # (out, in)ᵀ
        f = x.to(r.dtype) @ r.T
        error = x - (f @ r).to(x.dtype)
        #print("h2")
        #print(error)
        #print(error.shape)
        #print(f.shape)
        return f, error


class SubspaceInverseFeaturizerModule(torch.nn.Module):
    """Inverse of :class:`SubspaceFeaturizerModule`."""

    def __init__(self, rotate_layer: pv.models.layers.LowRankRotateLayer):
        super().__init__()
        self.rotate = rotate_layer

    def forward(self, f, error):
        r = self.rotate.weight.T
        return (f.to(r.dtype) @ r).to(f.dtype) + error.to(f.dtype)

#Denis:changed
########################################################################
class SubspaceFeaturizerModule_RevNet(torch.nn.Module):
    """Linear projector onto an orthogonal *rotation* sub-space."""

    def __init__(self, RevNet,sh,trainable):
        super().__init__()
        self.RevNet = RevNet
        self.shape=sh
        self.trainable=trainable

    def forward(self, x: torch.Tensor):
        return self.RevNet(x.to(torch.float32)),torch.zeros_like(x)


class SubspaceInverseFeaturizerModule_RevNet(torch.nn.Module):
    """Inverse of :class:`SubspaceFeaturizerModule`."""

    def __init__(self, RevNet,sh,trainable):
        super().__init__()
        self.RevNet = RevNet
        self.shape=sh
        self.trainable=trainable

    def forward(self, f, error):
        return self.RevNet.inverse(f).to(error.dtype) + error


class SubspaceFeaturizerModule_RevNet_SufficientExperiment(torch.nn.Module):
    """Linear projector onto an orthogonal *rotation* sub-space."""

    def __init__(self, RevNet,sh,trainable,Suff_RS,Suff_dims):
        super().__init__()
        self.RevNet = RevNet
        self.shape=sh
        self.trainable=trainable
        self.register_buffer("Suff_RS", Suff_RS)
        self.Suff_dims=Suff_dims

    def forward(self, x: torch.Tensor):
        #print(self.Suff_RS)
        x = x.to(torch.float32)
        if  self.Suff_dims>0:
            x = x @ self.Suff_RS
            x_s = x[:, :Suff_dims]
            x   = x[:, Suff_dims:]
            error = x_s @ self.Suff_RS[:,:self.Suff_dims].T
        else:
            error=torch.zeros_like(x)
        return self.RevNet(x), error


class SubspaceInverseFeaturizerModule_RevNet_SufficientExperiment(torch.nn.Module):
    """Inverse of :class:`SubspaceFeaturizerModule`."""

    def __init__(self, RevNet,sh,trainable,Suff_RS,Suff_dims):
        super().__init__()
        self.RevNet = RevNet
        self.shape=sh
        self.trainable=trainable
        self.register_buffer("Suff_RS", Suff_RS)
        self.Suff_dims=Suff_dims

    def forward(self, f, error):
        #print(self.Suff_RS)
        x = self.RevNet.inverse(f)
        if self.Suff_dims>0:
            x = x @ self.Suff_RS[:,self.Suff_dims:].T
        return x.to(error.dtype) + error



class SubspaceFeaturizerModule_RotationRevNet(torch.nn.Module):
    """Linear projector onto an orthogonal *rotation* sub-space."""

    def __init__(self, Rotation, RevNet,sh,trainable):
        super().__init__()
        self.RevNet = RevNet
        self.rotate = Rotation
        self.shape=sh
        self.trainable=trainable

    def forward(self, x: torch.Tensor):
        #print(self.rotate.weight)
        r = self.rotate.weight.T  # (out, in)ᵀ
        f = x.to(r.dtype) @ r.T
        error = x - (f @ r).to(x.dtype)
        #print("SubspaceFeaturizerModule_RotationRevNet in",flush=True)
        return self.RevNet(f.to(torch.float32)),error


class SubspaceInverseFeaturizerModule_RotationRevNet(torch.nn.Module):
    """Inverse of :class:`SubspaceFeaturizerModule`."""

    def __init__(self, Rotation, RevNet,sh,trainable):
        super().__init__()
        self.RevNet = RevNet
        self.rotate = Rotation
        self.shape=sh
        self.trainable=trainable

    def forward(self, f, error):
        f = self.RevNet.inverse(f).to(error.dtype)
        r = self.rotate.weight.T
        #print("SubspaceInverseFeaturizerModule_RotationRevNet out",flush=True)
        return (f.to(r.dtype) @ r).to(f.dtype) + error.to(f.dtype)


class SubspaceFeaturizerModule_RotationRevNet_SufficientExperiment(torch.nn.Module):
    """Linear projector onto an orthogonal *rotation* sub-space."""

    def __init__(self, Rotation, RevNet,sh,trainable,Suff_RS,Suff_dims):
        super().__init__()
        self.RevNet = RevNet
        self.rotate = Rotation
        self.shape=sh
        self.trainable=trainable
        self.register_buffer("Suff_RS", Suff_RS)
        self.Suff_dims=Suff_dims
        #print("f",self.Suff_dims)

    def forward(self, x: torch.Tensor):
        x = x.to(torch.float32)
        if self.Suff_dims>0:
            x = x @ self.Suff_RS
            x_s = x[:, :self.Suff_dims]
            x   = x[:, self.Suff_dims:]
            error2 = x_s @ self.Suff_RS[:,:self.Suff_dims].T
        else:
            error2=torch.zeros_like(x)
        #print(self.rotate.weight)
        r = self.rotate.weight.T  # (out, in)ᵀ
        f = x.to(r.dtype) @ r.T
        error = x - (f @ r).to(x.dtype)
        if self.Suff_dims>0:
            error  =  error2+(error@ self.Suff_RS[:,self.Suff_dims:].T)
        #print("SubspaceFeaturizerModule_RotationRevNet in",flush=True)
        return self.RevNet(f.to(torch.float32)),error


class SubspaceInverseFeaturizerModule_RotationRevNet_SufficientExperiment(torch.nn.Module):
    """Inverse of :class:`SubspaceFeaturizerModule`."""

    def __init__(self, Rotation, RevNet,sh,trainable,Suff_RS,Suff_dims):
        super().__init__()
        self.RevNet = RevNet
        self.rotate = Rotation
        self.shape=sh
        self.trainable=trainable
        self.register_buffer("Suff_RS", Suff_RS)
        self.Suff_dims=Suff_dims
        #print("i",self.Suff_dims)


    def forward(self, f, error):
        f = self.RevNet.inverse(f).to(error.dtype)
        r = self.rotate.weight.T
        #print("SubspaceInverseFeaturizerModule_RotationRevNet out",flush=True)
        x = f.to(r.dtype) @ r
        #print(x.shape)
        #print()
        if self.Suff_dims>0:
           x=x@ self.Suff_RS[:,self.Suff_dims:].T
        return x.to(f.dtype) + error.to(f.dtype)


class SubspaceFeaturizerModule_Linear(torch.nn.Module):
    """Linear projector onto an orthogonal *rotation* sub-space."""

    def __init__(self, Li, sh, trainable):
        super().__init__()
        self.Li = Li
        self.shape=sh
        self.trainable=trainable

    def forward(self, x: torch.Tensor):
        return self.Li(x.to(torch.float32)),torch.zeros_like(x)


class SubspaceInverseFeaturizerModule_Linear(torch.nn.Module):
    """Inverse of :class:`SubspaceFeaturizerModule`."""

    def __init__(self, Li, sh, trainable):
        super().__init__()
        self.Li = Li
        self.shape=sh
        self.trainable=trainable

    def forward(self, f, error):
        return self.Li.inverse(f).to(error.dtype) + error


class SubspaceFeaturizerModule_RotationLinear(torch.nn.Module):
    """Linear projector onto an orthogonal *rotation* sub-space."""

    def __init__(self, Rotation, Li, sh, trainable):
        super().__init__()
        self.Li = Li
        self.rotate = Rotation
        self.shape=sh
        self.trainable=trainable

    def forward(self, x: torch.Tensor):
        #print(self.rotate.weight)
        r = self.rotate.weight.T  # (out, in)ᵀ
        f = x.to(r.dtype) @ r.T
        error = x - (f @ r).to(x.dtype)
        #print("SubspaceFeaturizerModule_RotationRevNet in",flush=True)
        return self.Li(f.to(torch.float32)),error


class SubspaceInverseFeaturizerModule_RotationLinear(torch.nn.Module):
    """Inverse of :class:`SubspaceFeaturizerModule`."""

    def __init__(self, Rotation, Li,sh,trainable):
        super().__init__()
        self.Li = Li
        self.rotate = Rotation
        self.shape=sh
        self.trainable=trainable

    def forward(self, f, error):

        f = self.Li.inverse(f).to(error.dtype)
        r = self.rotate.weight.T
        #print("SubspaceInverseFeaturizerModule_RotationRevNet out",flush=True)
        return (f.to(r.dtype) @ r).to(f.dtype) + error.to(f.dtype)


class SubspaceFeaturizerModule_RotationLinear_SufficientExperiment(torch.nn.Module):
    """Linear projector onto an orthogonal *rotation* sub-space."""

    def __init__(self, Rotation, Li, sh, trainable,Suff_RS,Suff_dims):
        super().__init__()
        self.Li = Li
        self.rotate = Rotation
        self.shape=sh
        self.trainable=trainable
        self.register_buffer("Suff_RS", Suff_RS)
        self.Suff_dims=Suff_dims


    def forward(self, x: torch.Tensor):
        x = x.to(torch.float32)
        if self.Suff_dims>0:
            x = x @ self.Suff_RS
            x_s = x[:, :self.Suff_dims]
            x   = x[:, self.Suff_dims:]
            error2 = x_s @ self.Suff_RS[:,:self.Suff_dims].T
        else:
            error2=torch.zeros_like(x)
        #print(self.rotate.weight)
        r = self.rotate.weight.T  # (out, in)ᵀ
        f = x.to(r.dtype) @ r.T
        error = x - (f @ r).to(x.dtype)
        if self.Suff_dims>0:
            error  =  error2+(error@ self.Suff_RS[:,self.Suff_dims:].T)
        #print("SubspaceFeaturizerModule_RotationRevNet in",flush=True)
        return self.Li(f.to(torch.float32)),error


class SubspaceInverseFeaturizerModule_RotationLinear_SufficientExperiment(torch.nn.Module):
    """Inverse of :class:`SubspaceFeaturizerModule`."""

    def __init__(self, Rotation, Li,sh,trainable,Suff_RS,Suff_dims):
        super().__init__()
        self.Li = Li
        self.rotate = Rotation
        self.shape=sh
        self.trainable=trainable
        self.register_buffer("Suff_RS", Suff_RS)
        self.Suff_dims=Suff_dims


    def forward(self, f, error):
        f = self.Li.inverse(f).to(error.dtype)
        r = self.rotate.weight.T
        x = f.to(r.dtype) @ r
        #print(x.shape)
        #print()
        if self.Suff_dims>0:
           x=x@ self.Suff_RS[:,self.Suff_dims:].T
        return x.to(f.dtype) + error.to(f.dtype)


########################################################################


class SubspaceFeaturizer(Featurizer):
    """Orthogonal linear sub-space featurizer."""

    def __init__(
        self,
        *,
        shape: Tuple[int, int] | None = None,
        rotation_subspace: torch.Tensor | None = None,
        trainable: bool = True,
        id: str = "subspace",
    ):
        assert (
            shape is not None or rotation_subspace is not None
        ), "Provide either `shape` or `rotation_subspace`."

        if shape is not None:
            rotate = pv.models.layers.LowRankRotateLayer(*shape, init_orth=True)
        else:
            shape = rotation_subspace.shape
            rotate = pv.models.layers.LowRankRotateLayer(*shape, init_orth=False)
            rotate.weight.data.copy_(rotation_subspace)

        rotate = torch.nn.utils.parametrizations.orthogonal(rotate)
        rotate.requires_grad_(trainable)
        #print("Loaded Prelearned rotation!!!!!!!!!!!!!!!!!!!")
        #model_info=torch.load(f"Featurizer_Prelearned")
        #rotate.load_state_dict(model_info["model_info"]["additional_config"]["state_dict_Ro"])
        rotate.requires_grad_(trainable)
        super().__init__(
            SubspaceFeaturizerModule(rotate),
            SubspaceInverseFeaturizerModule(rotate),
            n_features=rotate.weight.shape[1],
            id=id,
        )

#Denis:changed
################################################################################
class SubspaceFeaturizerRevNet(Featurizer):
    """RevNet sub-space featurizer."""

    def __init__(
        self,
        *,
        shape: Tuple[int, int] | None = None,
        rotation_subspace: torch.Tensor | None = None,
        trainable: bool = True,
        id: str = "subspace",
    ):

        RN = RevNet(shape[0], shape[1], shape[2])
        RN.requires_grad_(trainable)
        super().__init__(
            SubspaceFeaturizerModule_RevNet(RN,shape,trainable),
            SubspaceInverseFeaturizerModule_RevNet(RN,shape,trainable),
            n_features=shape[1],
            id=id,
        )


class SubspaceFeaturizerRevNet_SufficientExperiment(Featurizer):
    """RevNet sub-space featurizer."""

    def __init__(
        self,
        *,
        shape: Tuple[int, int] | None = None,
        rotation_subspace: torch.Tensor | None = None,
        trainable: bool = True,
        id: str = "subspace",
    ):
        if os.path.exists("./Featurizer_Suff"):
            #print("RevNet Suff","!"*100)
            Suff_rotate=torch.load(f"Featurizer_Suff")
            Suff_dims=Suff_rotate.shape[1]
            Suff_rotate=Suff_rotate.numpy()
            #print(rotation.shape)
            rotation_full, _ = np.linalg.qr(Suff_rotate, mode='complete')
            #print(rotation_full.shape)
            rotation_project=torch.tensor(rotation_full)
        else:
            Suff_dims=0
            rotation_project=None

        RN = RevNet(shape[0], shape[1]-Suff_dims, shape[2])
        RN.requires_grad_(trainable)

        #a=SubspaceFeaturizerModule_RevNet_SufficientExperiment(RN,shape,trainable,rotation_project)
        #ai=SubspaceInverseFeaturizerModule_RevNet_SufficientExperiment(RN,shape,trainable,rotation_project)
        #tens=torch.rand((2,4096))
        #print(tens)
        #lat,err=a.forward(tens)
        #print(lat)
        #retens=ai.forward(lat,err)
        #print(retens)
        #exit()
        super().__init__(
            SubspaceFeaturizerModule_RevNet_SufficientExperiment(RN,shape,trainable,rotation_project,Suff_dims),
            SubspaceInverseFeaturizerModule_RevNet_SufficientExperiment(RN,shape,trainable,rotation_project,Suff_dims),
            n_features=shape[1],
            id=id,
        )


class SubspaceFeaturizerRotationRevNet(Featurizer):
    """RevNet sub-space featurizer."""

    def __init__(
        self,
        *,
        shape: Tuple[int, int] | None = None,
        rotation_subspace: torch.Tensor | None = None,
        trainable: bool = True,
        id: str = "subspace",
    ):
        #print(shape,flush=True)
        RN = RevNet(shape[0], shape[3], shape[2])
        RN.requires_grad_(trainable)
        rotate = pv.models.layers.LowRankRotateLayer(shape[1],shape[3] , init_orth=True)
        #print(RN)
        rotate = torch.nn.utils.parametrizations.orthogonal(rotate)
        rotate.requires_grad_(trainable)
        #print(rotate)
        super().__init__(
            SubspaceFeaturizerModule_RotationRevNet(rotate,RN,shape,trainable),
            SubspaceInverseFeaturizerModule_RotationRevNet(rotate,RN,shape,trainable),
            n_features=shape[1],
            id=id,
        )



class SubspaceFeaturizerRotationRevNet_SufficientExperiment(Featurizer):
    """RevNet sub-space featurizer."""

    def __init__(
        self,
        *,
        shape: Tuple[int, int] | None = None,
        rotation_subspace: torch.Tensor | None = None,
        trainable: bool = True,
        id: str = "subspace",
    ):
        if os.path.exists("./Featurizer_Suff"):
            #print("I am in")
            #print("RevNet Suff","!"*100)
            Suff_rotate=torch.load(f"Featurizer_Suff")
            Suff_dims=Suff_rotate.shape[1]
            Suff_rotate=Suff_rotate.numpy()
            #print(rotation.shape)
            rotation_full, _ = np.linalg.qr(Suff_rotate, mode='complete')
            #print(rotation_full.shape)
            rotation_project=torch.tensor(rotation_full)
        else:
            Suff_dims=0
            rotation_project=None
        #print(shape,flush=True)
        RN = RevNet(shape[0], shape[3], shape[2])
        RN.requires_grad_(trainable)
        rotate = pv.models.layers.LowRankRotateLayer(shape[1]-Suff_dims,shape[3] , init_orth=True)
        #print(RN)
        rotate = torch.nn.utils.parametrizations.orthogonal(rotate)
        rotate.requires_grad_(trainable)
        #print(rotate)
        #a=SubspaceFeaturizerModule_RotationRevNet_SufficientExperiment(rotate,RN,shape,trainable,rotation_project)
        #ai=SubspaceInverseFeaturizerModule_RotationRevNet_SufficientExperiment(rotate,RN,shape,trainable,rotation_project)
        #tens=torch.rand((2,4098))
        #print(tens)
        #lat,err=a.forward(tens)
        #print(lat)
        #retens=ai.forward(lat,err)
        #print(retens)
        #exit()

        super().__init__(
            SubspaceFeaturizerModule_RotationRevNet_SufficientExperiment(rotate,RN,shape,trainable,rotation_project,Suff_dims),
            SubspaceInverseFeaturizerModule_RotationRevNet_SufficientExperiment(rotate,RN,shape,trainable,rotation_project,Suff_dims),
            n_features=shape[1],
            id=id,
        )



class SubspaceFeaturizerLinear(Featurizer):
    """RevNet sub-space featurizer."""

    def __init__(
        self,
        *,
        shape: Tuple[int, int] | None = None,
        rotation_subspace: torch.Tensor | None = None,
        trainable: bool = True,
        id: str = "subspace",
    ):
        #print(shape,flush=True)
        Li = InvertibleLinearStable(shape[0])
        Li.requires_grad_(trainable)
        #print(rotate)
        super().__init__(
            SubspaceFeaturizerModule_Linear(Li,shape,trainable),
            SubspaceInverseFeaturizerModule_Linear(Li,shape,trainable),
            n_features=shape[0],
            id=id,
        )


class SubspaceFeaturizerRotationLinear(Featurizer):
    """RevNet sub-space featurizer."""

    def __init__(
        self,
        *,
        shape: Tuple[int, int] | None = None,
        rotation_subspace: torch.Tensor | None = None,
        trainable: bool = True,
        id: str = "subspace",
    ):
        #print(shape,flush=True)
        Li = InvertibleLinearStable(shape[1])
        Li.requires_grad_(trainable)
        rotate = pv.models.layers.LowRankRotateLayer(shape[0],shape[1] , init_orth=True)
        #print(RN)
        rotate = torch.nn.utils.parametrizations.orthogonal(rotate)
        rotate.requires_grad_(trainable)
        #print(rotate)
        super().__init__(
            SubspaceFeaturizerModule_RotationLinear(rotate,Li,shape,trainable),
            SubspaceInverseFeaturizerModule_RotationLinear(rotate,Li,shape,trainable),
            n_features=shape[0],
            id=id,
        )

class SubspaceFeaturizerRotationLinear_SufficientExperiment(Featurizer):
    """RevNet sub-space featurizer."""

    def __init__(
        self,
        *,
        shape: Tuple[int, int] | None = None,
        rotation_subspace: torch.Tensor | None = None,
        trainable: bool = True,
        id: str = "subspace",
    ):
        if os.path.exists("./Featurizer_Suff"):
            #print("I am in")
            #print("RevNet Suff","!"*100)
            Suff_rotate=torch.load(f"Featurizer_Suff")
            Suff_dims=Suff_rotate.shape[1]
            Suff_rotate=Suff_rotate.numpy()
            #print(rotation.shape)
            rotation_full, _ = np.linalg.qr(Suff_rotate, mode='complete')
            #print(rotation_full.shape)
            rotation_project=torch.tensor(rotation_full)
        else:
            Suff_dims=0
            rotation_project=None

        #print(shape,flush=True)
        Li = InvertibleLinearStable(shape[1])
        Li.requires_grad_(trainable)
        #print(Suff_dims)
        rotate = pv.models.layers.LowRankRotateLayer(shape[0]-Suff_dims,shape[1] , init_orth=True)
        #print(RN)
        rotate = torch.nn.utils.parametrizations.orthogonal(rotate)
        rotate.requires_grad_(trainable)
        #print(rotate)
        #print(rotate)
        #a=SubspaceFeaturizerModule_RotationLinear_SufficientExperiment(rotate,Li,shape,trainable,rotation_project,Suff_dims)
        #ai=SubspaceInverseFeaturizerModule_RotationLinear_SufficientExperiment(rotate,Li,shape,trainable,rotation_project,Suff_dims)
        #tens=torch.rand((2,4096))
        #print(tens)
        #lat,err=a.forward(tens)
        #print(lat)
        #retens=ai.forward(lat,err)
        #print(retens)
        #exit()

        super().__init__(
            SubspaceFeaturizerModule_RotationLinear_SufficientExperiment(rotate,Li,shape,trainable,rotation_project,Suff_dims),
            SubspaceInverseFeaturizerModule_RotationLinear_SufficientExperiment(rotate,Li,shape,trainable,rotation_project,Suff_dims),
            n_features=shape[0],
            id=id,
        )

################################################################################


class SAEFeaturizerModule(torch.nn.Module):
    """Wrapper around a *Sparse Autoencoder*’s encode() / decode() pair."""

    def __init__(self, sae):
        super().__init__()
        self.sae = sae

    def forward(self, x):
        features = self.sae.encode(x.to(self.sae.dtype))
        error = x - self.sae.decode(features).to(x.dtype)
        return features.to(x.dtype), error


class SAEInverseFeaturizerModule(torch.nn.Module):
    """Inverse for :class:`SAEFeaturizerModule`."""

    def __init__(self, sae):
        super().__init__()
        self.sae = sae

    def forward(self, features, error):
        return (
            self.sae.decode(features.to(self.sae.dtype)).to(features.dtype)
            + error.to(features.dtype)
        )


class SAEFeaturizer(Featurizer):
    """Featurizer backed by a pre-trained sparse auto-encoder.

    Notes
    -----
    Serialisation is *disabled* for SAE featurizers – saving will raise
    ``NotImplementedError``.
    """

    def __init__(self, sae, *, trainable: bool = False):
        sae.requires_grad_(trainable)
        super().__init__(
            SAEFeaturizerModule(sae),
            SAEInverseFeaturizerModule(sae),
            n_features=sae.cfg.to_dict()["d_sae"],
            id="sae",
        )


# --------------------------------------------------------------------------- #
#  Utility helpers                                                            #
# --------------------------------------------------------------------------- #
def _subspace_is_all_none(subspaces) -> bool:
    """Return ``True`` if *every* element of *subspaces* is ``None``."""
    return subspaces is None or all(
        inner is None or all(elem is None for elem in inner) for inner in subspaces
    )






# --------------------------------------------------------------------------- #
#  RevNet (Denis:changed)                                                     #
# --------------------------------------------------------------------------- #





class MLP(nn.Module):
    """
    A Multi-Layer Perceptron (MLP) with configurable depth and width.

    This implementation allows for dynamic specification of the network architecture
    through the hidden_sizes parameter, which determines both the width and depth
    of the network.

    Args:
        input_size (int): Dimensionality of the input features
        output_size (int): Dimensionality of the output features
        hidden_sizes (list of int): List specifying the size of each hidden layer
        activation (nn.Module, optional): Activation function to use between layers.
                                         Defaults to nn.ReLU()
        dropout_rate (float, optional): Dropout probability. Defaults to 0.0
    """
    def __init__(self, input_size, output_size, hidden_sizes, activation=nn.ReLU(), dropout_rate=0.0):
        super(MLP, self).__init__()

        # Validate inputs
        if not isinstance(hidden_sizes, list) or len(hidden_sizes) == 0:
            raise ValueError("hidden_sizes must be a non-empty list of integers")

        # Build the layers
        layers = []

        # Input layer
        layers.append(nn.Linear(input_size, hidden_sizes[0]))
        layers.append(activation)
        if dropout_rate > 0:
            layers.append(nn.Dropout(dropout_rate))

        # Hidden layers
        for i in range(len(hidden_sizes) - 1):
            layers.append(nn.Linear(hidden_sizes[i], hidden_sizes[i+1]))
            layers.append(activation)
            if dropout_rate > 0:
                layers.append(nn.Dropout(dropout_rate))

        # Output layer
        layers.append(nn.Linear(hidden_sizes[-1], output_size))

        # Combine all layers
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        """Forward pass through the MLP."""
        return self.network(x)


class RevNet_Block(nn.Module):
    def __init__(self, in_features, hidden_size, depth=1):
        super(RevNet_Block, self).__init__()
        self.half_in_features=in_features//2

        self.F = MLP(self.half_in_features, self.half_in_features, [hidden_size]*depth)
        self.G = MLP(self.half_in_features, self.half_in_features, [hidden_size]*depth)

    def forward(self, x):
        x_1 = x[:,:self.half_in_features]
        x_2 = x[:,self.half_in_features:]
        F_O = self.F(x_2)
        y_1 = x_1 + F_O
        G_O = self.G(y_1)
        y_2 = x_2 + G_O
        y   = torch.cat((y_1, y_2), dim=1)
        return y

    def inverse(self, y):
        y_1 = y[:,:self.half_in_features]
        y_2 = y[:,self.half_in_features:]
        G_O = self.G(y_1)
        x_2 = y_2 - G_O
        F_O = self.F(x_2)
        x_1 = y_1 - F_O
        x   = torch.cat((x_1, x_2), dim=1)
        return x


class RevNet(nn.Module):
    def __init__(self, number_blocks, in_features, hidden_size, depth=1):
        super(RevNet, self).__init__()
        Model_Layers = []
        #print("h1",in_features)
        for i in range(number_blocks):
            Model_Layers.append(RevNet_Block(in_features, hidden_size, depth))
        self.Model_Layers = nn.ModuleList(Model_Layers)

    def forward(self, x):
        for ac_layer in self.Model_Layers:
            x = ac_layer(x)
        return x

    def inverse(self, y):
        """Applies inverse transformation with high precision."""
        for ac_layer in reversed(self.Model_Layers):
            y = ac_layer.inverse(y)
        return y



# --------------------------------------------------------------------------- #
#  ManualIntervention (Denis:changed)                                         #
# --------------------------------------------------------------------------- #



class ManualClassifier(nn.Module):
    def __init__(self, planes_json_path, mode="g"):
        super().__init__()
        assert mode in ["g", "o", "go"], "Mode must be 'g', 'o', or 'go'"
        self.mode = mode

        with open(planes_json_path, "r") as f:
            planes = json.load(f)

        self.cp = nn.Parameter(torch.tensor(planes["cp"], dtype=torch.float32))
        self.v1 = nn.Parameter(torch.tensor(planes["v1"], dtype=torch.float32))
        self.v2 = nn.Parameter(torch.tensor(planes["v2"], dtype=torch.float32))

    def point_line_distance(self, point, direction):
        # Compute distance from point to infinite line through cp along direction vector
        cp_to_point = point - self.cp
        direction_norm = direction / torch.norm(direction)
        projection_length = (cp_to_point @ direction_norm)
        projection = projection_length.unsqueeze(-1) * direction_norm
        closest_point = self.cp + projection
        return torch.norm(point - closest_point, dim=-1)

    def forward(self, x):
        """
        x: Tensor of shape (batch_size, 2)
        returns: Tensor of shape (batch_size, 2)
        """

        d1 = self.point_line_distance(x, self.v1)
        d2 = self.point_line_distance(x, self.v2)
        result = torch.zeros((x.size(0), 2), device=x.device)

        # Set [1.0, 0.0] for items where d1 < d2
        mask = d1 < d2
        result[mask] = torch.tensor([1.0, 1.0], device=x.device)
        print("Forward Manual",result[:10])
        return result

    def sample_point_on_line(self, direction, constraint_axis, constraint_range):
        direction = direction / torch.norm(direction)
        while True:
            t = random.uniform(-50, 50)
            point = self.cp + t * direction
            if constraint_axis == 0:  # x
                if constraint_range[0] <= point[0].item() <= constraint_range[1]:
                    return point
            else:  # y
                if constraint_range[0] <= point[1].item() <= constraint_range[1]:
                    return point

    def inverse(self, x):
        """
        x: Tensor of shape (batch_size, 2)
        Returns: Tensor of shape (batch_size, 2)
        """
        print("inverse Manual",x[:10])
        batch_out = []
        for val in x:
            label = val[0].item()

            if self.mode == "g":
                if label == 0.0:
                    p = self.sample_point_on_line(self.v2, 1, (5, 20))
                else:
                    p = self.sample_point_on_line(self.v1, 0, (5, 20))
            elif self.mode == "o":
                if label == 0.0:
                    p = self.sample_point_on_line(self.v2, 1, (-20, -5))
                else:
                    p = self.sample_point_on_line(self.v1, 0, (-20, -5))
            elif self.mode == "go":
                if label == 0.0:
                    if random.random() < 0.5:
                        p = self.sample_point_on_line(self.v2, 1, (-20, -5))
                    else:
                        p = self.sample_point_on_line(self.v2, 1, (5, 20))
                else:
                    if random.random() < 0.5:
                        p = self.sample_point_on_line(self.v1, 0, (-20, -5))
                    else:
                        p = self.sample_point_on_line(self.v1, 0, (5, 20))
            batch_out.append(p)

        return torch.stack(batch_out)





# --------------------------------------------------------------------------- #
#  LinearIntervention (Denis:changed)                                         #
# --------------------------------------------------------------------------- #



class InvertibleLinearLU(nn.Module):
    def __init__(self, features):
        super().__init__()
        self.features = features

        # Initialize LU components
        P, L, U = torch.lu_unpack(*torch.lu(torch.randn(features, features)))
        self.register_buffer("P", P)
        self.L_params = nn.Parameter(L.tril(-1))  # strictly lower
        self.U_params = nn.Parameter(U.triu())    # upper including diagonal

        self.bias = nn.Parameter(torch.randn(features))

    def _assemble_W(self):
        L = self.L_params + torch.eye(self.features, device=self.L_params.device)
        U = self.U_params
        return self.P @ L @ U

    def forward(self, x):
        #print("Li input")
        W = self._assemble_W()
        return x @ W.T + self.bias

    def inverse(self, y):
        #print("Li output")
        W = self._assemble_W()
        W_inv = torch.linalg.inv(W)
        return (y - self.bias) @ W_inv.T


class InvertibleLinearEx(nn.Module):
    def __init__(self, features):
        super().__init__()
        self.features = features

        # Initialize a matrix A for parameterization
        # W = expm(A) will be invertible
        self.A = nn.Parameter(torch.randn(features, features))

        # Bias can be anything
        self.bias = nn.Parameter(torch.randn(features))

    def forward(self, x):
        # Compute W as a matrix exponential to ensure invertibility
        W = torch.matrix_exp(self.A)
        return x @ W.T + self.bias

    def inverse(self, y):
        # Compute W and its inverse
        W = torch.matrix_exp(self.A)
        W_inv = torch.linalg.inv(W)
        return (y - self.bias) @ W_inv.T


class InvertibleLinearStable(nn.Module):
    def __init__(self, features):
        super().__init__()
        self.features = features

        # Unconstrained matrix for Q (orthogonal basis)
        self.Q_base = nn.Parameter(torch.randn(features, features))

        # Log-diagonal values (ensures positivity)
        self.log_diag = nn.Parameter(torch.randn(features))

        #self.bias = nn.Parameter(torch.randn(features))

    def _compute_W(self):
        # QR for orthogonal matrix Q
        Q, _ = torch.linalg.qr(self.Q_base)
        D = torch.diag(torch.exp(self.log_diag))  # ensure positive values
        return Q @ D @ Q.T

    def forward(self, x):
        #print("li for")
        W = self._compute_W()
        return x @ W.T

    def inverse(self, y):
        #print("Li inv")
        W = self._compute_W()
        W_inv = torch.linalg.inv(W)  # should be very stable now
        return y  @ W_inv.T
