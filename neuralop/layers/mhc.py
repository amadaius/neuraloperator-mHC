"""
Manifold Hyper-Connection (mHC) Module for FNO

This module implements the dual-mode Sinkhorn-Knopp algorithm with continuous
kernel density balancing (KDB) for neural operators.

References:
    - mHC paper: Manifold Hyper-Connections for Neural Operators
"""

from typing import Literal, Optional, Tuple
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class DualModeSinkhorn(nn.Module):
    """
    Dual-Mode Sinkhorn-Knopp algorithm with continuous Kernel Density Balancing.

    This module supports two modes:
    - 'discrete': Traditional channel-wise Sinkhorn iterations (O(N·n²))
    - 'continuous': Continuous KDB with spatial smoothing using depthwise conv (O(N·K²))

    All computations are performed in log-domain for numerical stability.

    Parameters
    ----------
    n_streams : int
        Number of streams in the manifold (n in n×C expansion)
    sinkhorn_iter : int, optional
        Number of Sinkhorn iterations, by default 20
    kdb_bandwidth : float, optional
        Bandwidth for kernel density balancing, by default 1.0
    kernel_size : int, optional
        Size of the truncated kernel for depthwise convolution, by default 5
    mode : Literal["discrete", "continuous"], optional
        Mode of operation, by default "continuous"
    eps : float, optional
        Stabilization factor for log-domain operations, by default 1e-8

    Notes
    -----
    In continuous mode, we use depthwise separable convolution to approximate
    the continuous kernel density balancing, reducing complexity from O(N²) to O(K²)
    where K is the kernel size.
    """

    def __init__(
        self,
        n_streams: int,
        sinkhorn_iter: int = 20,
        kdb_bandwidth: float = 0.5,  # Reduced from 1.0 to reduce smoothing conflict
        kernel_size: int = 5,
        mode: Literal["discrete", "continuous"] = "continuous",
        padding_mode: Literal["circular", "replicate", "zeros"] = "circular",  # Fixed: use circular to prevent mass leakage
        eps: float = 1e-8,
    ):
        super().__init__()
        self.n_streams = n_streams
        self.sinkhorn_iter = sinkhorn_iter
        self.kdb_bandwidth = kdb_bandwidth
        self.kernel_size = kernel_size
        self.mode = mode
        self.padding_mode = padding_mode
        self.eps = eps

        # Validate kernel size is odd for symmetric padding
        assert kernel_size % 2 == 1, f"kernel_size must be odd, got {kernel_size}"

        # Generate Gaussian kernel for continuous mode
        if mode == "continuous":
            self.register_buffer(
                "gaussian_kernel",
                self._create_gaussian_kernel(kernel_size, kdb_bandwidth)
            )
            self.padding = kernel_size // 2

            # Setup padding mode
            self.use_padding_layer = (padding_mode in ["replicate", "reflection"])
            if self.use_padding_layer:
                if padding_mode == "replicate":
                    self.row_pad = nn.ReplicationPad2d(self.padding)
                    self.col_pad = nn.ReplicationPad2d(self.padding)
                elif padding_mode == "reflection":
                    self.row_pad = nn.ReflectionPad2d(self.padding)
                    self.col_pad = nn.ReflectionPad2d(self.padding)

    def _create_gaussian_kernel(self, kernel_size: int, bandwidth: float) -> torch.Tensor:
        """Create a 2D Gaussian kernel for depthwise convolution."""
        # Create coordinate grid
        x = torch.arange(kernel_size, dtype=torch.float32) - kernel_size // 2
        y = torch.arange(kernel_size, dtype=torch.float32) - kernel_size // 2
        xx, yy = torch.meshgrid(x, y, indexing='ij')

        # Gaussian function
        kernel = torch.exp(-(xx**2 + yy**2) / (2 * bandwidth**2))

        # Normalize to sum to 1 (probability kernel)
        kernel = kernel / kernel.sum()

        # Shape: (1, 1, kernel_size, kernel_size)
        return kernel.unsqueeze(0).unsqueeze(0)

    def _log_sum_exp_spatial(
        self,
        log_values: torch.Tensor,
        log_weights: Optional[torch.Tensor] = None,
        kernel: Optional[torch.Tensor] = None,
        padding: int = 0,
    ) -> torch.Tensor:
        """
        Compute log-space spatial sum-exp with optional Gaussian smoothing.

        Parameters
        ----------
        log_values : torch.Tensor
            Log-domain values of shape (B, n, H, W)
        log_weights : torch.Tensor, optional
            Log-domain weights of shape (B, 1, H, W). If None, no weighting.
        kernel : torch.Tensor, optional
            Gaussian kernel for smoothing, by default None
        padding : int, optional
            Padding amount, by default 0

        Returns
        -------
        torch.Tensor
            Log-domain result of shape (B, 1, H, W)
        """
        # Apply weighting if provided
        if log_weights is not None:
            # Broadcast log_weights to match log_values: (B, 1, H, W) -> (B, n, H, W)
            log_values = log_values + log_weights.expand_as(log_values)

        # Compute log-space sum: log(Σ exp(log_values))
        # Using torch.logsumexp for numerical stability
        log_marginal = torch.logsumexp(
            log_values,
            dim=1,  # Sum over streams dimension
            keepdim=True  # Keep dimension for broadcasting
        )

        if kernel is not None:
            # Continuous mode: smooth with Gaussian kernel in linear space
            # Convert to linear space
            marginal_linear = torch.exp(log_marginal)

            # Apply Gaussian smoothing via depthwise convolution
            # marginal_linear: (B, 1, H, W)
            # kernel: (1, 1, kernel_size, kernel_size)
            marginal_smoothed = F.conv2d(
                marginal_linear,
                kernel,
                padding=padding,
                groups=1
            )

            # Add stabilization factor and convert back to log space
            log_marginal = torch.log(marginal_smoothed + self.eps)

        return log_marginal

    def _apply_circular_pad(self, x: torch.Tensor, padding: int) -> torch.Tensor:
        """
        Apply circular padding to a 4D tensor (B, C, H, W).

        This creates a larger tensor where the edges wrap around,
        simulating periodic boundary conditions.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (B, C, H, W)
        padding : int
            Amount of padding on each side

        Returns
        -------
        torch.Tensor
            Padded tensor of shape (B, C, H + 2*padding, W + 2*padding)
        """
        B, C, H, W = x.shape

        # Create padded tensor
        padded_h = H + 2 * padding
        padded_w = W + 2 * padding
        padded = torch.zeros((B, C, padded_h, padded_w), dtype=x.dtype, device=x.device)

        # Fill center region with original
        padded[:, :, padding:padding+H, padding:padding+W] = x

        # Circular wrapping for top/bottom
        padded[:, :, :padding, padding:padding+W] = x[:, :, -padding:, :]  # Bottom wraps to top
        padded[:, :, padding+H:, padding:padding+W] = x[:, :, :padding, :]  # Top wraps to bottom

        # Circular wrapping for left/right (including corners)
        padded[:, :, padding:padding+H, :padding] = x[:, :, :, -padding:]  # Right wraps to left
        padded[:, :, padding:padding+H, padding+W:] = x[:, :, :, :padding]  # Left wraps to right

        # Handle corners (circular wrapping in both dimensions)
        padded[:, :, :padding, :padding] = x[:, :, -padding:, -padding:]  # Bottom-right to top-left
        padded[:, :, :padding, padding+W:] = x[:, :, -padding:, :padding]  # Bottom-left to top-right
        padded[:, :, padding+H:, :padding] = x[:, :, :padding, -padding:]  # Top-right to bottom-left
        padded[:, :, padding+H:, padding+W:] = x[:, :, :padding, :padding]  # Top-left to bottom-right

        return padded

    def forward(
        self,
        log_H_res: torch.Tensor,
        return_stats: bool = False
    ) -> Tuple[torch.Tensor, Optional[dict]]:
        """
        Perform dual-mode Sinkhorn iterations on log-domain residual weights.

        Parameters
        ----------
        log_H_res : torch.Tensor
            Log-domain residual weights of shape (B, n*n, H, W)
        return_stats : bool, optional
            Whether to return statistics for monitoring, by default False

        Returns
        -------
        Tuple[torch.Tensor, Optional[dict]]
            - Projected matrix H_star of shape (B, n, n, H, W)
            - Statistics dict if return_stats=True, else None
        """
        batch_size = log_H_res.shape[0]
        spatial_shape = log_H_res.shape[2:]

        # Reshape from (B, n*n, H, W) to (B, n, n, H, W)
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

        # Kernel and padding for continuous mode
        kernel = self.gaussian_kernel if self.mode == "continuous" else None
        padding = self.padding if self.mode == "continuous" else 0

        # Sinkhorn iterations using linear space for numerical stability
        # Convert to linear space for computations
        H_linear = torch.exp(log_H_res)

        for _ in range(self.sinkhorn_iter):
            # Normalize rows: divide each row by its sum
            # Compute row marginals with optional KDB smoothing
            row_marginals = torch.sum(H_linear, dim=2, keepdim=True)  # (B, n, 1, H, W)

            # Apply KDB smoothing in continuous mode (prevent mass leakage)
            if kernel is not None:
                # Reshape for spatial conv: (B, n, 1, H, W) -> (B*n, 1, H, W)
                B, n, _, H, W = row_marginals.shape
                row_marginals_conv = row_marginals.reshape(B * n, 1, H, W)

                # Apply padding manually based on mode
                if self.padding_mode == "circular":
                    # Manual circular padding on both spatial dimensions
                    padded = self._apply_circular_pad(row_marginals_conv, self.padding)
                elif self.padding_mode == "replicate":
                    # Replicate padding: pad (left, right, top, bottom)
                    padded = F.pad(row_marginals_conv, (self.padding, self.padding, self.padding, self.padding), mode='replicate')
                else:  # zeros (constant padding)
                    padded = F.pad(row_marginals_conv, (self.padding, self.padding, self.padding, self.padding), mode='constant')

                # Apply convolution with padding=0 since we pre-padded
                row_marginals = F.conv2d(
                    padded,
                    kernel,
                    stride=1,
                    padding=0,
                )  # (B*n, 1, H, W)
                row_marginals = row_marginals.reshape(B, n, 1, H, W)

            # Normalize rows
            H_linear = H_linear / (row_marginals + self.eps)

            # Normalize columns: divide each column by its sum
            # Compute column marginals with optional KDB smoothing
            col_marginals = torch.sum(H_linear, dim=1, keepdim=True)  # (B, 1, n, H, W)

            # Apply KDB smoothing in continuous mode (prevent mass leakage)
            if kernel is not None:
                # Reshape for spatial conv: (B, 1, n, H, W) -> (B*n, 1, H, W)
                B, _, n, H, W = col_marginals.shape
                col_marginals_conv = col_marginals.reshape(B * n, 1, H, W)

                # Apply padding manually based on mode
                if self.padding_mode == "circular":
                    # Manual circular padding on both spatial dimensions
                    padded = self._apply_circular_pad(col_marginals_conv, self.padding)
                elif self.padding_mode == "replicate":
                    # Replicate padding: pad (left, right, top, bottom)
                    padded = F.pad(col_marginals_conv, (self.padding, self.padding, self.padding, self.padding), mode='replicate')
                else:  # zeros (constant padding)
                    padded = F.pad(col_marginals_conv, (self.padding, self.padding, self.padding, self.padding), mode='constant')

                # Apply convolution with padding=0 since we pre-padded
                col_marginals = F.conv2d(
                    padded,
                    kernel,
                    stride=1,
                    padding=0,
                )  # (B*n, 1, H, W)
                col_marginals = col_marginals.reshape(B, 1, n, H, W)

            # Normalize columns
            H_linear = H_linear / (col_marginals + self.eps)

        # Convert back to log space for final output
        log_H_res = torch.log(H_linear + self.eps)

        # Convert to linear space for final matrix
        H_star = torch.exp(log_H_res)

        # Gather statistics if requested
        stats = None
        if return_stats:
            # Check bistochastic property: sum over rows and columns should be ~1.0
            row_sum = H_star.sum(dim=2)  # Sum over columns: (B, n, H, W)
            col_sum = H_star.sum(dim=1)  # Sum over rows: (B, n, H, W)

            row_error = torch.abs(row_sum - 1.0).max().item()
            col_error = torch.abs(col_sum - 1.0).max().item()

            stats = {
                "max_row_error": row_error,
                "max_col_error": col_error,
                "max_error": max(row_error, col_error),
            }

        return H_star, stats


class mHC_FNO_Block(nn.Module):
    """
    Manifold Hyper-Connection Block for FNO.

    This block integrates the mHC branch as a third parallel pathway in FNO,
    alongside spectral convolution and 1x1 convolution branches.

    The mHC branch consists of:
    1. Pre-mapping: Expand channels from C to n×C using 1x1 convolution
    2. Weight generation: Generate residual weights from hidden state
    3. Sinkhorn projection: Project to bistochastic manifold
    4. Post-mapping: Compress channels back using 1x1 convolution
    5. Stream mixing: Apply projected matrix for feature mixing

    Parameters
    ----------
    in_channels : int
        Number of input channels
    out_channels : int
        Number of output channels (typically equals in_channels for FNO)
    mhc_expansion_ratio : int, optional
        Expansion ratio for manifold (n in n×C), by default 4
    sinkhorn_iter : int, optional
        Number of Sinkhorn iterations, by default 20
    kdb_bandwidth : float, optional
        Bandwidth for kernel density balancing, by default 1.0
    kernel_size : int, optional
        Size of truncated kernel for KDB, by default 5
    mode : Literal["discrete", "continuous"], optional
        Mode of operation, by default "continuous"

    Notes
    -----
    The block uses 1x1 convolutions (nn.Conv2d with kernel_size=1) for
    pre/post-mapping to maintain spatial alignment with FNO's point-wise
    operations and preserve spatial topology of PDE data.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        mhc_expansion_ratio: int = 4,
        sinkhorn_iter: int = 20,
        kdb_bandwidth: float = 1.0,
        kernel_size: int = 5,
        mode: Literal["discrete", "continuous"] = "continuous",
        padding_mode: Literal["circular", "replicate", "zeros"] = "circular",
    ):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.expansion_ratio = mhc_expansion_ratio
        self.n_streams = mhc_expansion_ratio
        self.mode = mode

        # Pre-mapping: C -> n×C using 1x1 convolution
        self.pre_mapping = nn.Conv2d(
            in_channels,
            in_channels * mhc_expansion_ratio,
            kernel_size=1,
            bias=False
        )

        # Weight generation: generate residual weights from hidden state
        # Shape: (B, C, H, W) -> (B, n*n, H, W)
        self.weight_generator = nn.Conv2d(
            in_channels,
            mhc_expansion_ratio ** 2,
            kernel_size=1,
            bias=False
        )

        # Dual-mode Sinkhorn for bistochastic projection
        self.sinkhorn = DualModeSinkhorn(
            n_streams=mhc_expansion_ratio,
            sinkhorn_iter=sinkhorn_iter,
            kdb_bandwidth=kdb_bandwidth,
            kernel_size=kernel_size,
            mode=mode,
            padding_mode=padding_mode,
        )

        # Post-mapping: n×C -> C using 1x1 convolution
        self.post_mapping = nn.Conv2d(
            in_channels * mhc_expansion_ratio,
            out_channels,
            kernel_size=1,
            bias=False
        )

    def forward(
        self,
        x: torch.Tensor,
        return_stats: bool = False
    ) -> Tuple[torch.Tensor, Optional[dict]]:
        """
        Forward pass of mHC FNO block.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (B, C, H, W)
        return_stats : bool, optional
            Whether to return Sinkhorn statistics, by default False

        Returns
        -------
        Tuple[torch.Tensor, Optional[dict]]
            - Output tensor of shape (B, C, H, W)
            - Statistics dict if return_stats=True, else None
        """
        batch_size = x.shape[0]
        spatial_shape = x.shape[2:]

        # Step 1: Pre-mapping - Expand channels
        # (B, C, H, W) -> (B, n×C, H, W)
        x_expanded = self.pre_mapping(x)

        # Reshape for stream operations
        # (B, n×C, H, W) -> (B, n, C, H, W)
        x_expanded = x_expanded.view(batch_size, self.n_streams, self.in_channels, *spatial_shape)

        # Step 2: Generate residual weights from hidden state
        # (B, C, H, W) -> (B, n*n, H, W) in log-domain
        log_H_res = self.weight_generator(x)

        # Step 3: Sinkhorn projection to bistochastic manifold
        H_star, sinkhorn_stats = self.sinkhorn(log_H_res, return_stats=return_stats)

        # Step 4: Stream mixing using einsum
        # Apply H_star to mix features across streams
        # x_expanded: (B, n, C, H, W), H_star: (B, n, n, H, W)
        # Result: (B, n, C, H, W) where each output stream is weighted sum of input streams
        x_mixed = torch.einsum('bnchw,bnnhw->bnchw', x_expanded, H_star)

        # Reshape back
        # (B, n, C, H, W) -> (B, n×C, H, W)
        x_mixed = x_mixed.reshape(batch_size, self.in_channels * self.n_streams, *spatial_shape)

        # Step 5: Post-mapping - Compress channels
        # (B, n×C, H, W) -> (B, C, H, W)
        output = self.post_mapping(x_mixed)

        # Gather all statistics
        stats = None
        if return_stats and sinkhorn_stats is not None:
            stats = sinkhorn_stats

        # Return output directly if not requesting stats, otherwise return tuple
        if return_stats:
            return output, stats
        else:
            return output
