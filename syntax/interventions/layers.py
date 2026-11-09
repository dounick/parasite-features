import torch
import normflows as nf
from interventions.neural import RevNet, RealNVP


class UnconstrainedLayer(torch.nn.Module):
    """
    A simple unconstrained linear transformation layer with improved numerical stability.
    """
    def __init__(self, n):
        super().__init__()
        # Initialize closer to identity with small random perturbations
        self.weight = torch.nn.Parameter(
            torch.eye(n) + 0.01 * torch.randn(n, n)
        )

    def forward(self, x):
        # Apply gradient clipping during forward pass
        with torch.no_grad():
            if self.weight.grad is not None:
                torch.nn.utils.clip_grad_norm_(self.weight, max_norm=1.0)
        return x.to(self.weight.dtype) @ self.weight


class StretchLayer(torch.nn.Module):
    """
    A linear stretching transformation, constrained to be a diagonal matrix.
    Only diagonal elements are learnable.
    """
    def __init__(self, n, init_val=1.0):
        super().__init__()
        scale = torch.full((n,), init_val)  # 1D vector
        self.weight = torch.nn.Parameter(scale, requires_grad=True)

    def forward(self, x):
        return x.to(self.weight.dtype) * self.weight


class RealNVPLayer(torch.nn.Module):
    """
    A RealNVP layer for normalizing flows.
    """
    def __init__(self, n_layers, latent_dim, hidden_dim, device='cuda:0'):
        super().__init__()
        print(f"Creating RealNVP layer with {n_layers} layers, latent_dim={latent_dim}, hidden_dim={hidden_dim}, device={device}")
        self.realnvp = RealNVP(n_layers, latent_dim, hidden_dim)
        device = device if torch.cuda.is_available() else "cpu"
        self.realnvp.to(device)


    def forward(self, x):
        """
        Forward pass through the RealNVP layer.
        """
        return self.realnvp(x)


class InvertibleLinearStable(torch.nn.Module):
    def __init__(self, features):
        super().__init__()
        self.features = features

        # Unconstrained matrix for Q (orthogonal basis)
        self.Q_base = torch.nn.Parameter(torch.randn(features, features))

        # Log-diagonal values (ensures positivity)
        self.log_diag = torch.nn.Parameter(torch.randn(features))

        #self.bias = nn.Parameter(torch.randn(features))

    def _compute_W(self):
        # QR for orthogonal matrix Q
        Q, _ = torch.linalg.qr(self.Q_base)
        D = torch.diag(torch.exp(self.log_diag))
        return Q @ D @ Q.T

    def forward(self, x):
        W = self._compute_W()
        return x @ W.T

    def inverse(self, y):
        W = self._compute_W()
        W_inv = torch.linalg.inv(W)
        return y  @ W_inv.T


class RevNetLayer(torch.nn.Module):
    """
    A RevNet layer for reversible neural networks.
    """
    def __init__(self, num_layers, input_dim, hidden_dim, device='cuda:0'):
        super().__init__()
        self.revnet = RevNet(number_blocks=num_layers, in_features=input_dim, hidden_size=hidden_dim)
        device = device if torch.cuda.is_available() else "cpu"
        self.revnet.to(device)

    def forward(self, x):
        """
        Forward pass through the RevNet layer.
        """
        return self.revnet(x)
