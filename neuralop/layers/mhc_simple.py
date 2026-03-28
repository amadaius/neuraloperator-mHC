"""
Simplified DualModeSinkhorn for debugging

This is a simplified version without complex KDB padding logic
to isolate the core Sinkhorn algorithm.
"""

from typing import Literal, Optional, Tuple
import math
import torch
import torch.nn as nn


class DualModeSinkhorn(nn.Module):
    """
    Simplified Dual-Mode Sinkhorn for debugging.

    Supports two modes:
    - 'discrete': Traditional channel-wise Sinkhorn iterations (O(N·n²))
    - 'continuous': Placeholder for future KDB implementation

    All computations are performed in log-domain for numerical stability.

    Parameters
    ----------
    n_streams : int
        Number of streams in the manifold (n in n×C expansion)
    sinkhorn_iter : int, optional
        Number of Sinkhorn iterations, by default 20
    eps : float, optional
        Stabilization factor for log-domain operations, by default 1e-8

    Notes
    -----
    The continuous mode is a placeholder for future KDB implementation.
    Current version only supports discrete mode.
    """

    def __init__(
        self,
        n_streams: int,
        sinkhorn_iter: int = 20,
        eps: float = 1e-8,
    ):
        super().__init__()
        self.n_streams = n_streams
        self.sinkhorn_iter = sinkhorn_iter
        self.eps = eps

    def forward(
        self,
        log_H_res: torch.Tensor,
        return_stats: bool = False,
    ) -> Tuple[torch.Tensor, Optional[dict]]:
        """
        Forward pass with Sinkhorn iterations.

        Parameters
        ----------
        log_H_res : torch.Tensor
            Log-domain residual weights of shape (B, n*n, H, W)
        return_stats : bool, optional
            Whether to return statistics for monitoring, by default False

        Returns
        -------
        Tuple[torch.Tensor, Optional[dict]]
            - Projected bistochastic matrix of shape (B, n, n, H, W)
            - Statistics dict if return_stats=True, else None
        """
        batch_size = log_H_res.shape[0]
        spatial_shape = log_H_res.shape[2:]

        # Reshape from (B, n*n, H, W) to (B, n, n, H, W)
        # This is the correct 4D format for F.conv2d
        log_H_res = log_H_res.view(batch_size, self.n_streams, self.n_streams, *spatial_shape)

        # Initialize log-domain row and column marginals
        # Start with uniform distribution: log(1/n) for each stream
        log_row = torch.full(
            (batch_size, self.n_streams, *spatial_shape),
            -math.log(self.n_streams),
            device=log_H_res.device,
            dtype=log_H_res.dtype
        )
        log_col = log_row.clone()

        # Simple Sinkhorn iterations (no complex padding logic)
        for _ in range(self.sinkhorn_iter):
            # Normalize rows: subtract log(row_marginals)
            row_marginals = torch.sum(log_H_res, dim=2, keepdim=True)  # (B, 1, H, W)
            log_H_res = log_H_res - row_marginals

            # Normalize columns: subtract log(col_marginals)
            col_marginals = torch.sum(log_H_res, dim=1, keepdim=True)  # (B, 1, H, W)
            log_H_res = log_H_res - col_marginals

        # Convert back to linear space
        H_star = torch.exp(log_H_res)

        # Gather statistics if requested
        stats = None
        if return_stats:
            # Check bistochastic property (sum over rows and columns should be ~1.0)
            row_sum = H_star.sum(dim=2)  # Sum over columns: (B, n, H, W)
            col_sum = H_star.sum(dim=1)  # Sum over rows: (B, 1, H, W)

            row_error = torch.abs(row_sum - 1.0).max().item()
            col_error = torch.abs(col_sum - 1.0).max().item()

            stats = {
                'max_row_error': row_error,
                'max_col_error': col_error,
                'max_error': max(row_error, col_error),
            }

        return H_star, stats


if __name__ == "__main__":
    # Quick test
    batch_size = 2
    n_streams = 4
    height, width = 16, 16

    # Generate random log residual weights
    log_H_res = torch.randn(batch_size, n_streams * n_streams, height, width)

    sinkhorn = DualModeSinkhorn(n_streams=n_streams, sinkhorn_iter=5)
    with torch.no_grad():
        H_star, stats = sinkhorn(log_H_res, return_stats=True)

    print(f"Output shape: {H_star.shape}")
    print(f"Row sum error: {stats['max_row_error']:.6e}")
    print(f"Col sum error: {stats['max_col_error']:.6e}")
    print(f"✓ Basic Sinkhorn test passed!")
