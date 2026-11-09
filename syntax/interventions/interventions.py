import torch
from pyvene.models.intervention_utils import _do_intervention_by_swap
from pyvene.models.interventions import TrainableIntervention, DistributedRepresentationIntervention, LowRankRotatedSpaceIntervention
from pyvene.models.layers import RotateLayer
import normflows as nf

from interventions.layers import StretchLayer, RealNVPLayer, RevNetLayer, UnconstrainedLayer, InvertibleLinearStable
import os


class LowRankRotatedSpaceInterventionCustomDim(LowRankRotatedSpaceIntervention):
    """Low-rank projection intervention with configurable number of swap dimensions.

    This intervention projects representations to a low-rank subspace and swaps
    a configurable number of dimensions from source to base.

    Args:
        intervention_dim: Number of dimensions to swap (must be <= low_rank_dimension)
        **kwargs: Additional arguments passed to LowRankRotatedSpaceIntervention
    """
    def __init__(self, intervention_dim=1, **kwargs):
        super().__init__(**kwargs)
        self.register_buffer('interchange_dim', torch.tensor(intervention_dim))

    def forward(self, base, source, subspaces=None, **kwargs):
        # Project to low-rank subspace
        rotated_base = self.rotate_layer(base)
        rotated_source = self.rotate_layer(source)

        # Interchange in rotated space
        _rb = rotated_base.clone()
        _rs = rotated_source.clone()
        intervened_rotated = _do_intervention_by_swap(
            _rb,
            _rs,
            "interchange",
            self.interchange_dim,
            subspaces,
            subspace_partition=self.subspace_partition,
            use_fast=self.use_fast,
        )

        # Compute intervention in rotated space
        rotated_intervention = intervened_rotated - rotated_base

        # Map back to original space and add to base
        output = base + torch.matmul(rotated_intervention, self.rotate_layer.weight.T)
        return output.to(base.dtype)

    def __str__(self):
        low_rank_dim = self.rotate_layer.weight.shape[1]
        return f"LowRankRotatedSpaceInterventionCustomDim(low_rank_dimension={low_rank_dim}, interchange_dim={self.interchange_dim.item()})"


class LowRankInvertibleRotatedIntervention(LowRankRotatedSpaceIntervention):
    """Intervention in a low-rank subspace with a RevNet transformation."""
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        rotate_layer = RotateLayer(2)
        self.rotate_layer_two = torch.nn.utils.parametrizations.orthogonal(rotate_layer)
        self.register_buffer('interchange_dim', torch.tensor(1)) # defaulting to a 1d intervention

    def forward(self, base, source, subspaces=None, **kwargs):
        # Project to low-rank subspace
        rotated_base = self.rotate_layer(base)
        rotated_source = self.rotate_layer(source)

        # Apply RevNet transformation
        transformed_base = self.rotate_layer_two(rotated_base)
        transformed_source = self.rotate_layer_two(rotated_source)

        # Interchange in transformed space
        _tb = transformed_base.clone()
        _ts = transformed_source.clone()
        intervened_transformed = _do_intervention_by_swap(
            _tb,
            _ts,
            "interchange",
            self.interchange_dim,
            subspaces,
            subspace_partition=self.subspace_partition,
            use_fast=self.use_fast,
        )

        # Inverse transform back to rotated space
        intervened_rotated = torch.matmul(intervened_transformed, self.rotate_layer_two.weight.T)

        # Compute intervention in rotated space
        rotated_intervention = intervened_rotated - rotated_base

        # Map back to original space and add to base
        output = base + torch.matmul(rotated_intervention, self.rotate_layer.weight.T)
        return output.to(base.dtype)

    def __str__(self):
        return f"LowRankRotatedIntervention(low_rank_dimension={self.low_rank_dimension})"


class LowRankInvertibleRevNetIntervention(LowRankRotatedSpaceIntervention):
    """Intervention in a low-rank subspace with a RevNet transformation."""
    def __init__(self, n_layers=4, latent_dim=None, hidden_dim=None, **kwargs):
        super().__init__(**kwargs)
        self.revNetLayer = RevNetLayer(n_layers, 2, hidden_dim)
        self.register_buffer('interchange_dim', torch.tensor(1)) # defaulting to a 1d intervention

    def forward(self, base, source, subspaces=None, **kwargs):
        # Project to low-rank subspace
        rotated_base = self.rotate_layer(base)
        rotated_source = self.rotate_layer(source)

        # Apply RevNet transformation
        transformed_base = self.revNetLayer(rotated_base)
        transformed_source = self.revNetLayer(rotated_source)

        # Interchange in transformed space
        _tb = transformed_base.clone()
        _ts = transformed_source.clone()
        intervened_transformed = _do_intervention_by_swap(
            _tb,
            _ts,
            "interchange",
            self.interchange_dim,
            subspaces,
            subspace_partition=self.subspace_partition,
            use_fast=self.use_fast,
        )

        # Inverse transform back to rotated space
        intervened_rotated = self.revNetLayer.revnet.inverse(intervened_transformed)

        # Compute intervention in rotated space
        rotated_intervention = intervened_rotated - rotated_base

        # Map back to original space and add to base
        output = base + torch.matmul(rotated_intervention, self.rotate_layer.weight.T)
        return output.to(base.dtype)

    def __str__(self):
        return f"LowRankInvertibleRevNetIntervention(low_rank_dimension={self.low_rank_dimension})"


class LowRankInvertibleRealNVPIntervention(LowRankRotatedSpaceIntervention):
    """Intervention in a low-rank subspace with an unconstrained invertible transformation."""
    def __init__(self, n_layers=4, latent_dim=None, hidden_dim=None, **kwargs):
        super().__init__(**kwargs)
        self.realNVPlayer = RealNVPLayer(n_layers, 2, hidden_dim)
        self.register_buffer('interchange_dim', torch.tensor(1)) # defaulting to a 1d intervention

    def forward(self, base, source, subspaces=None, **kwargs):
        # Project to low-rank subspace
        rotated_base = self.rotate_layer(base)
        rotated_source = self.rotate_layer(source)

        # Apply unconstrained transformation
        transformed_base = self.realNVPlayer(rotated_base)
        transformed_source = self.realNVPlayer(rotated_source)

        # Interchange in transformed space
        _tb = transformed_base.clone()
        _ts = transformed_source.clone()
        intervened_transformed = _do_intervention_by_swap(
            _tb,
            _ts,
            "interchange",
            self.interchange_dim,
            subspaces,
            subspace_partition=self.subspace_partition,
            use_fast=self.use_fast,
        )

        # Inverse transform back to rotated space
        intervened_rotated = self.realNVPlayer.realnvp.inverse(intervened_transformed)

        # Compute intervention in rotated space
        rotated_intervention = intervened_rotated - rotated_base

        # Map back to original space and add to base
        output = base + torch.matmul(rotated_intervention, self.rotate_layer.weight.T)
        return output.to(base.dtype)

    def __str__(self):
        return f"LowRankInvertibleRealNVPIntervention(low_rank_dimension={self.low_rank_dimension})"


class LowRankInvertibleUnconstrainedIntervention(LowRankRotatedSpaceIntervention):
    """Intervention in a low-rank subspace with an unconstrained invertible transformation."""
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.unconstrained_layer = InvertibleLinearStable(2) # hardcode to 2 for visualization purposes
        self.register_buffer('interchange_dim', torch.tensor(1)) # defaulting to a 1d intervention

    def forward(self, base, source, subspaces=None, **kwargs):
        # Project to low-rank subspace
        rotated_base = self.rotate_layer(base)
        rotated_source = self.rotate_layer(source)

        # Apply unconstrained transformation
        transformed_base = self.unconstrained_layer(rotated_base)
        transformed_source = self.unconstrained_layer(rotated_source)

        # Interchange in transformed space
        _tb = transformed_base.clone()
        _ts = transformed_source.clone()
        intervened_transformed = _do_intervention_by_swap(
            _tb,
            _ts,
            "interchange",
            self.interchange_dim,
            subspaces,
            subspace_partition=self.subspace_partition,
            use_fast=self.use_fast,
        )

        # Inverse transform back to rotated space
        intervened_rotated = self.unconstrained_layer.inverse(intervened_transformed)

        # Compute intervention in rotated space
        rotated_intervention = intervened_rotated - rotated_base

        # Map back to original space and add to base
        output = base + torch.matmul(rotated_intervention, self.rotate_layer.weight.T)
        return output.to(base.dtype)

    def __str__(self):
        low_rank_dim = self.rotate_layer.weight.shape[1]
        return f"LowRankInvertibleUnconstrainedIntervention(low_rank_dimension={low_rank_dim}, interchange_dim={self.interchange_dim.item()})"

class LowRankInvertibleUnconstrained2DIntervention(LowRankRotatedSpaceIntervention):
    """Intervention in a low-rank subspace with an unconstrained invertible transformation. Difference is intervene on 2d rather than 1d!"""
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.unconstrained_layer = InvertibleLinearStable(2) # hardcode to 2 for visualization purposes
        self.register_buffer('interchange_dim', torch.tensor(2)) # defaulting to a *2d intervention*

    def forward(self, base, source, subspaces=None, **kwargs):
        # Project to low-rank subspace
        rotated_base = self.rotate_layer(base)
        rotated_source = self.rotate_layer(source)

        # Apply unconstrained transformation
        transformed_base = self.unconstrained_layer(rotated_base)
        transformed_source = self.unconstrained_layer(rotated_source)

        # Interchange in transformed space
        _tb = transformed_base.clone()
        _ts = transformed_source.clone()
        intervened_transformed = _do_intervention_by_swap(
            _tb,
            _ts,
            "interchange",
            self.interchange_dim,
            subspaces,
            subspace_partition=self.subspace_partition,
            use_fast=self.use_fast,
        )

        # Inverse transform back to rotated space
        intervened_rotated = self.unconstrained_layer.inverse(intervened_transformed)

        # Compute intervention in rotated space
        rotated_intervention = intervened_rotated - rotated_base

        # Map back to original space and add to base
        output = base + torch.matmul(rotated_intervention, self.rotate_layer.weight.T)
        return output.to(base.dtype)

    def __str__(self):
        low_rank_dim = self.rotate_layer.weight.shape[1]
        return f"LowRankInvertibleUnconstrained2DIntervention(low_rank_dimension={low_rank_dim}, interchange_dim={self.interchange_dim.item()})"

class InvertibleUnconstrainedIntervention(TrainableIntervention, DistributedRepresentationIntervention):
    """Unconstrained linear intervention."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        unconstrained_layer = InvertibleLinearStable(self.embed_dim)
        self.unconstrained_layer = unconstrained_layer


    def forward(self, base, source, subspaces=None, **kwargs):
        transformed_base = self.unconstrained_layer(base)
        transformed_source = self.unconstrained_layer(source)
        # interchange
        # clone to avoid in-place modifications that can break autograd
        _tb = transformed_base.clone()
        _ts = transformed_source.clone()
        transformed_base = _do_intervention_by_swap(
            _tb,
            _ts,
            "interchange",
            self.interchange_dim,
            subspaces,
            subspace_partition=self.subspace_partition,
            use_fast=self.use_fast,
        )
        # inverse base
        output = self.unconstrained_layer.inverse(transformed_base)
        return output.to(base.dtype)


    def __str__(self):
        return f"InvertibleUnconstrainedIntervention()"


class UnconstrainedIntervention(TrainableIntervention, DistributedRepresentationIntervention):
    """Unconstrained linear intervention."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        unconstrained_layer = UnconstrainedLayer(self.embed_dim)
        self.unconstrained_layer = unconstrained_layer


    def forward(self, base, source, subspaces=None, **kwargs):
        transformed_base = self.unconstrained_layer(base)
        transformed_source = self.unconstrained_layer(source)
        # interchange
        # clone to avoid in-place modifications that can break autograd
        _tb = transformed_base.clone()
        _ts = transformed_source.clone()
        transformed_base = _do_intervention_by_swap(
            _tb,
            _ts,
            "interchange",
            self.interchange_dim,
            subspaces,
            subspace_partition=self.subspace_partition,
            use_fast=self.use_fast,
        )
        # inverse base
        output = torch.linalg.pinv(self.unconstrained_layer.weight) @ transformed_base.T
        return output.T.to(base.dtype)


    def __str__(self):
        return f"UnconstrainedIntervention()"


class StretchedSpaceIntervention(TrainableIntervention, DistributedRepresentationIntervention):

    """Intervention in the scaled space."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        stretch_layer = StretchLayer(self.embed_dim)
        self.stretch_layer = stretch_layer


    def forward(self, base, source, subspaces=None, **kwargs):
        stretched_base = self.stretch_layer(base)
        stretched_source = self.stretch_layer(source)
        # interchange
        # clone to avoid in-place modifications that can break autograd
        _sb = stretched_base.clone()
        _ss = stretched_source.clone()
        stretched_base = _do_intervention_by_swap(
            _sb,
            _ss,
            "interchange",
            self.interchange_dim,
            subspaces,
            subspace_partition=self.subspace_partition,
            use_fast=self.use_fast,
        )
        # inverse base
        output = stretched_base / self.stretch_layer.weight
        return output.to(base.dtype)



    def __str__(self):
        return f"StretchedSpaceIntervention()"



class RealNVPIntervention(TrainableIntervention, DistributedRepresentationIntervention):
    """
        Intervention using RealNVP for the scaled space.
    """

    def __init__(self, n_layers=4, latent_dim=None, hidden_dim=None, **kwargs):
        super().__init__(**kwargs)
        self.realNVPlayer = RealNVPLayer(n_layers, latent_dim, hidden_dim)
        self.register_buffer('interchange_dim', torch.tensor(1))

    def forward(self, base, source, subspaces=None, **kwargs):
        flowed_base = self.realNVPlayer(base)
        flowed_source = self.realNVPlayer(source)
        # clone to avoid in-place modifications that can break autograd
        _fb = flowed_base.clone()
        _fs = flowed_source.clone()
        flowed_base = _do_intervention_by_swap(
            _fb,
            _fs,
            "interchange",
            self.interchange_dim,
            subspaces,
            subspace_partition=self.subspace_partition,
            use_fast=self.use_fast,
        )
        # inverse base
        output = self.realNVPlayer.realnvp.inverse(flowed_base)
        return output.to(base.dtype)

    def __str__(self):
        return f"RealNVPIntervention()"



class RevNetIntervention(TrainableIntervention, DistributedRepresentationIntervention):
    """
        Intervention using RevNet for the scaled space.
    """

    def __init__(self, n_layers=None, latent_dim=None, hidden_dim=None, **kwargs):
        super().__init__(**kwargs)
        self.revNetLayer = RevNetLayer(n_layers, latent_dim, hidden_dim)
        self.register_buffer('interchange_dim', torch.tensor(1))

    def forward(self, base, source, subspaces=None, **kwargs):
        revnet_base = self.revNetLayer(base)
        revnet_source = self.revNetLayer(source)
        # Optional debug hook: if REVNET_DEBUG=1 in the environment register a
        # backward hook on the revnet representation tensor so we can observe
        # the gradient that flows back to the representation during training.
        try:
            if os.environ.get("REVNET_DEBUG", "0") == "1":
                def _show_grad(g):
                    try:
                        nz = int((g.abs() > 0).sum().item()) if g.numel() > 0 else 0
                        print(f"REVNET_BASE grad: max={g.abs().max().item():.6e} mean={g.abs().mean().item():.6e} nonzero={nz} dtype={g.dtype} device={g.device}")
                    except Exception as e:
                        print("REVNET debug hook error:", e)

                # register_hook returns a removable handle but we don't store it here
                revnet_base.register_hook(_show_grad)
        except Exception:
            # don't fail forward if debugging hook can't be registered
            pass
        # interchange
        # clone to avoid in-place modifications that can break autograd
        _rb = revnet_base.clone()
        _rs = revnet_source.clone()
        revnet_base = _do_intervention_by_swap(
            _rb,
            _rs,
            "interchange",
            self.interchange_dim,
            subspaces,
            subspace_partition=self.subspace_partition,
            use_fast=self.use_fast,
        )
        # inverse base
        output = self.revNetLayer.revnet.inverse(revnet_base)
        return output.to(base.dtype)

    def __str__(self):
        return f"RevNetIntervention()"
