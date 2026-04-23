"""Patch m7_garment_texture.py compute() to use eigenvalue thresholding."""

filepath = r"d:\Pretrained_Metrics_Evaluation\pretrained_metrics\metrics\m7_garment_texture.py"

with open(filepath, "r", encoding="utf-8") as f:
    content = f.read()

# Normalize
content_n = content.replace('\r\n', '\n')

OLD = '''    # ------------------------------------------------------------------ #
    def compute(self) -> Dict[str, float]:
        D = self._encoder.embed_dim
        N = len(self._embeddings)
        if N < 2:
            return {
                "garment_diversity_logdet": float("nan"),
                "garment_variance_total":   float("nan"),
                "garment_embed_dim":        float(D),
            }

        E  = np.stack(self._embeddings, axis=0)        # (N, D)
        mu = E.mean(axis=0, keepdims=True)
        Ec = E - mu                                     # centred (N, D)

        # -- PCA via thin SVD --
        # Ec = U S Vt  ->  eigenvalues of Cov = S**2 / (N-1)
        # We keep only the top-k components where k = min(N-1, D, n_components).
        # This avoids summing log over hundreds of near-zero null-space dims
        # that arise because (a) N-1 < D and/or (b) L2 normalisation constrains
        # embeddings to a low-dimensional sub-manifold.
        k = min(N - 1, D, self.n_components)
        _, S, _ = np.linalg.svd(Ec, full_matrices=False)   # S shape: (min(N,D),)
        S = S[:k]                                           # top-k singular values
        eigvals = (S ** 2) / max(N - 1, 1)                 # (k,) eigenvalues

        # Light absolute regularisation (just prevents log(0) for tiny eigvals)
        reg_eigvals = eigvals + self.eps

        log_det   = float(np.sum(np.log(reg_eigvals)))
        total_var = float(eigvals.sum())

        return {
            "garment_diversity_logdet": log_det,
            "garment_variance_total":   total_var,
            "garment_embed_dim":        float(k),   # effective dims used
        }

    def reset(self):
        self._embeddings.clear()
'''

# The file might use unicode box-drawing chars for comments. Search for the actual text.
idx = content_n.find("def compute(self) -> Dict[str, float]:")
if idx < 0:
    print("ERROR: compute method not found")
    exit(1)

# Find start (4 spaces before the line with #--)
line_start = content_n.rfind("\n", 0, idx) + 1
# Check if there's a comment line before
prev_line_start = content_n.rfind("\n", 0, line_start - 1) + 1
prev_line = content_n[prev_line_start:line_start].strip()
if prev_line.startswith("# -"):
    line_start = prev_line_start

# Find end (after reset method)
reset_idx = content_n.find("def reset(self):", idx)
if reset_idx < 0:
    print("ERROR: reset method not found")
    exit(1)
# Find end of reset
end_idx = content_n.find("\n", content_n.find("self._embeddings.clear()", reset_idx))
if end_idx < 0:
    end_idx = len(content_n)
else:
    end_idx += 1  # include the newline

old_block = content_n[line_start:end_idx]
print(f"Found block from char {line_start} to {end_idx}")
print(f"Block starts with: {repr(old_block[:80])}")
print(f"Block ends with: {repr(old_block[-80:])}")

NEW_BLOCK = '''    # ------------------------------------------------------------------ #
    def compute(self) -> Dict[str, float]:
        D = self._encoder.embed_dim
        N = len(self._embeddings)
        if N < 2:
            return {
                "garment_diversity_logdet":      float("nan"),
                "garment_diversity_normalized":  float("nan"),
                "garment_variance_total":        float("nan"),
                "garment_embed_dim":             float(D),
                "garment_effective_rank":         float("nan"),
            }

        E  = np.stack(self._embeddings, axis=0)        # (N, D)
        mu = E.mean(axis=0, keepdims=True)
        Ec = E - mu                                     # centred (N, D)

        # ── PCA via thin SVD ──────────────────────────────────────────────────
        k_max = min(N - 1, D, self.n_components)
        _, S, _ = np.linalg.svd(Ec, full_matrices=False)
        S = S[:k_max]
        eigvals = (S ** 2) / max(N - 1, 1)

        # ── Eigenvalue thresholding ───────────────────────────────────────────
        # L2-normalised embeddings lie on a low-dimensional manifold, so most
        # eigenvalues are near-zero.  Summing log(~0) over 100+ dimensions
        # produces huge negative log-dets (e.g. -800).
        # Keep only eigenvalues above a meaningful threshold.
        EIGVAL_THRESHOLD = 1e-4
        sig_mask = eigvals > EIGVAL_THRESHOLD
        sig_eigvals = eigvals[sig_mask]
        effective_rank = int(sig_mask.sum())

        if effective_rank == 0:
            return {
                "garment_diversity_logdet":      float("-inf"),
                "garment_diversity_normalized":  float("-inf"),
                "garment_variance_total":        float(eigvals.sum()),
                "garment_embed_dim":             float(k_max),
                "garment_effective_rank":         0.0,
            }

        # Log-det over significant eigenvalues only
        reg_eigvals = sig_eigvals + self.eps
        log_det     = float(np.sum(np.log(reg_eigvals)))
        total_var   = float(eigvals.sum())

        # Normalised log-det (per effective dimension)
        log_det_norm = log_det / effective_rank

        return {
            "garment_diversity_logdet":      log_det,
            "garment_diversity_normalized":  log_det_norm,
            "garment_variance_total":        total_var,
            "garment_embed_dim":             float(effective_rank),
            "garment_effective_rank":         float(effective_rank),
        }

    def reset(self):
        self._embeddings.clear()
'''

content_n = content_n[:line_start] + NEW_BLOCK + content_n[end_idx:]

with open(filepath, "w", encoding="utf-8", newline='\n') as f:
    f.write(content_n)
print("SUCCESS: m7_garment_texture.py patched.")
