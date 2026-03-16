import numpy as np
import torch


def find_homographies_4_points(
    src_points: np.ndarray | torch.Tensor,
    dst_points: np.ndarray | torch.Tensor,
) -> np.ndarray | torch.Tensor:
    """
    Find homographies from batched 4-point correspondences using DLT.

    Args:
        src_points: (batch_size, 4, 3) — homogeneous source points
        dst_points: (batch_size, 4, 3) — homogeneous destination points

    Returns:
        (batch_size, 3, 3) homography matrices mapping src -> dst
    """
    is_torch = isinstance(src_points, torch.Tensor)
    batch_size = src_points.shape[0]
    if is_torch:
        A = torch.zeros((batch_size, 8, 9), dtype=src_points.dtype, device=src_points.device)
    else:
        A = np.zeros((batch_size, 8, 9), dtype=src_points.dtype)

    for j in range(4):
        s = src_points[:, j, :]
        u = dst_points[:, j, 0]
        v = dst_points[:, j, 1]
        w = dst_points[:, j, 2]

        # v*(h3·s) - w*(h2·s) = 0
        A[:, 2 * j, 3:6] = -w[..., None] * s
        A[:, 2 * j, 6:9] = v[..., None] * s

        # w*(h1·s) - u*(h3·s) = 0
        A[:, 2 * j + 1, 0:3] = w[..., None] * s
        A[:, 2 * j + 1, 6:9] = -u[..., None] * s

    if is_torch:
        _, _, Vh = torch.linalg.svd(A)
    else:
        _, _, Vh = np.linalg.svd(A)
    h = Vh[:, -1, :]  # (B, 9)

    return h.reshape(batch_size, 3, 3)


def find_homography_ransac(
    src_points: np.ndarray | torch.Tensor,
    dst_points: np.ndarray | torch.Tensor,
    num_iters: int = 1000,
    eps: float = 0.005,
) -> np.ndarray | torch.Tensor | None:
    """
    RANSAC-style homography estimation with a soft cosine-similarity scoring rule.

    Samples 4 random correspondences ``num_iters`` times, fits a homography
    for each sample, and scores it by summing a soft exponential function of
     the cosine similarity between ``H @ src`` and ``dst``, i.e.
     ``exp((cos_sim - 1) / eps)`` over all points. The homography with the
     highest total score is returned.

    Args:
        src_points: (N, 3) source points in homogeneous coords
        dst_points: (N, 3) destination points in homogeneous coords
        num_iters:  number of RANSAC iterations
        eps:        softness/scale parameter for the exponential cosine-similarity
                     score; smaller values make the score more sharply peaked near
                     perfect cosine similarity (cos_sim = 1).

    Returns:
        (3, 3) best homography, or None if fewer than 4 points
    """

    is_torch = isinstance(src_points, torch.Tensor)

    N = src_points.shape[0]
    if N < 4:
        return None

    if is_torch:
        device = src_points.device
        idx_list = [torch.randperm(N, device=device)[:4] for _ in range(num_iters)]
        indices = torch.stack(idx_list, dim=0)  # (I, 4)

        src_samples = src_points[indices]  # (I, 4, 3)
        dst_samples = dst_points[indices]  # (I, 4, 3)

        Hs = find_homographies_4_points(src_samples, dst_samples)  # (I, 3, 3)

        projected = Hs @ src_points.T  # (I, 3, N)
        dst_T = dst_points.T  # (3, N)

        dot_prod = (projected * dst_T.unsqueeze(0)).sum(dim=1)  # (I, N)
        norm_proj = torch.linalg.norm(projected, dim=1)  # (I, N)
        norm_dst = torch.linalg.norm(dst_T, dim=0)  # (N,)
        cos_sim = dot_prod / (norm_proj * norm_dst.unsqueeze(0) + 1e-12)

        rewards = torch.exp((cos_sim - 1) / eps).sum(dim=1)  # (I,)
        best_idx = rewards.argmax()
        return Hs[best_idx]

    # NumPy branch
    indices = np.stack(
        [np.random.choice(N, 4, replace=False) for _ in range(num_iters)],
        axis=0,
    )  # (I, 4)

    src_samples = src_points[indices]  # (I, 4, 3)
    dst_samples = dst_points[indices]  # (I, 4, 3)

    Hs = find_homographies_4_points(src_samples, dst_samples)  # (I, 3, 3)

    projected = Hs @ src_points.T  # (I, 3, N)

    dst_T = dst_points.T  # (3, N)
    dot_prod = (projected * dst_T[None]).sum(axis=1)  # (I, N)
    norm_proj = np.linalg.norm(projected, axis=1)  # (I, N)
    norm_dst = np.linalg.norm(dst_T, axis=0)  # (N,)
    cos_sim = dot_prod / (norm_proj * norm_dst[None] + 1e-12)  # (I, N)

    rewards = np.exp((cos_sim - 1) / eps).sum(axis=1)  # (I,)
    best_idx = np.argmax(rewards)
    return Hs[best_idx]
