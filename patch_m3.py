"""Patch m3_background.py compute() and reset() methods."""
import os

filepath = r"d:\Pretrained_Metrics_Evaluation\pretrained_metrics\metrics\m3_background.py"

with open(filepath, "r", encoding="utf-8") as f:
    content = f.read()

OLD_COMPUTE = '''    def compute(self) -> Dict[str, float]:
        ent = np.array([v for v in self._entropies if not math.isnan(v)])
        obj = np.array(self._obj_counts, dtype=float)
        sem_ent = np.array([v for v in self._semantic_entropies if not math.isnan(v)])
        sem_uniq = np.array(self._semantic_uniques, dtype=float)
        sem_global = _semantic_entropy(self._semantic_all)

        return {
            "bg_entropy_mean":        float(ent.mean()) if len(ent) else float("nan"),
            "bg_entropy_var":         float(ent.var())  if len(ent) else float("nan"),
            "bg_object_density_mean": float(obj.mean()) if len(obj) else float("nan"),
            "bg_semantic_entropy_mean": float(sem_ent.mean()) if len(sem_ent) else float("nan"),
            "bg_semantic_entropy_var":  float(sem_ent.var())  if len(sem_ent) else float("nan"),
            "bg_semantic_unique_mean":  float(sem_uniq.mean()) if len(sem_uniq) else float("nan"),
            "bg_semantic_unique_var":   float(sem_uniq.var())  if len(sem_uniq) else float("nan"),
            "bg_complexity_3A":       float(ent.mean()) if len(ent) else float("nan"),
            "bg_complexity_3B":       float(obj.mean()) if len(obj) else float("nan"),
            "bg_complexity_semantic": float(sem_ent.mean()) if len(sem_ent) else float("nan"),
            "bg_semantic_entropy_global": float(sem_global),
        }

    def reset(self):
        self._entropies.clear()
        self._semantic_entropies.clear()
        self._semantic_uniques.clear()
        self._semantic_all.clear()
        self._obj_counts.clear()
'''

NEW_COMPUTE = '''    def compute(self) -> Dict[str, float]:
        ent = np.array([v for v in self._entropies if not math.isnan(v)])
        obj = np.array(self._obj_counts, dtype=float)
        sem_ent = np.array([v for v in self._semantic_entropies if not math.isnan(v)])
        sem_uniq = np.array(self._semantic_uniques, dtype=float)
        sem_global = _semantic_entropy(self._semantic_all)

        # ── Sub-component means ───────────────────────────────────────────
        ent_mean  = float(ent.mean())  if len(ent)  else float("nan")
        ent_var   = float(ent.var())   if len(ent)  else float("nan")
        obj_mean  = float(obj.mean())  if len(obj)  else float("nan")
        sem_mean  = float(sem_ent.mean()) if len(sem_ent) else float("nan")
        sem_var   = float(sem_ent.var())  if len(sem_ent) else float("nan")
        uniq_mean = float(sem_uniq.mean()) if len(sem_uniq) else float("nan")
        uniq_var  = float(sem_uniq.var())  if len(sem_uniq) else float("nan")

        # ── Overall background complexity score (0-1) ─────────────────────
        #   Three normalised pillars combined with weights:
        #     1. Texture entropy  (max ~ 8 for 8-bit grayscale patches)
        #     2. Object density   (clamped at 20 objects)
        #     3. Semantic entropy  (max = log2(91 COCO classes) ~ 6.5)
        #
        #   Higher -> more complex / cluttered background.
        MAX_TEXTURE_ENT = 8.0
        MAX_OBJ_COUNT   = 20.0
        MAX_SEM_ENT     = math.log2(91)  # COCO has 91 class ids

        parts, weights = [], []
        if not math.isnan(ent_mean):
            parts.append(min(ent_mean / MAX_TEXTURE_ENT, 1.0))
            weights.append(0.35)
        if not math.isnan(obj_mean):
            parts.append(min(obj_mean / MAX_OBJ_COUNT, 1.0))
            weights.append(0.35)
        if not math.isnan(sem_mean):
            parts.append(min(sem_mean / MAX_SEM_ENT, 1.0))
            weights.append(0.30)

        if parts:
            w_sum = sum(weights)
            overall = sum(p * w for p, w in zip(parts, weights)) / w_sum
        else:
            overall = float("nan")

        # Human-readable difficulty label
        if math.isnan(overall):
            label_val = float("nan")
        elif overall < 0.30:
            label_val = 1.0   # simple
        elif overall < 0.60:
            label_val = 2.0   # moderate
        else:
            label_val = 3.0   # complex

        return {
            # ── Overall score ─────────────────────────────────────────────
            "bg_overall_complexity":       overall,        # 0-1 composite
            "bg_difficulty_label":         label_val,      # 1=simple 2=moderate 3=complex

            # ── Texture (3A) ──────────────────────────────────────────────
            "bg_entropy_mean":             ent_mean,
            "bg_entropy_var":              ent_var,
            "bg_complexity_3A":            ent_mean,

            # ── Object density (3B) ───────────────────────────────────────
            "bg_object_density_mean":      obj_mean,
            "bg_complexity_3B":            obj_mean,

            # ── Semantic diversity ────────────────────────────────────────
            "bg_semantic_entropy_mean":    sem_mean,
            "bg_semantic_entropy_var":     sem_var,
            "bg_semantic_unique_mean":     uniq_mean,
            "bg_semantic_unique_var":      uniq_var,
            "bg_complexity_semantic":      sem_mean,
            "bg_semantic_entropy_global":  float(sem_global),
        }

    def reset(self):
        self._entropies.clear()
        self._semantic_entropies.clear()
        self._semantic_uniques.clear()
        self._semantic_all.clear()
        self._obj_counts.clear()
'''

# Normalize line endings for matching
content_n = content.replace('\r\n', '\n')
old_n = OLD_COMPUTE.replace('\r\n', '\n')
new_n = NEW_COMPUTE.replace('\r\n', '\n')

if old_n in content_n:
    content_n = content_n.replace(old_n, new_n)
    with open(filepath, "w", encoding="utf-8", newline='\n') as f:
        f.write(content_n)
    print("SUCCESS: m3_background.py patched.")
else:
    print("ERROR: Could not find target text in file.")
    idx = content_n.find("def compute(self)")
    if idx >= 0:
        print(f"Found compute at char {idx}")
        print(repr(content_n[idx:idx+200]))
