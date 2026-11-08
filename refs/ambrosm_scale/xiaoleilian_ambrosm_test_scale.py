def load_gray(path: Path) -> np.ndarray:
    im = read_pil(path)
    return to_gray_u8(im)

def load_rgb(path: Path) -> np.ndarray:
    im = read_pil(path)
    return np.asarray(im.convert("RGB"))

def tick_profile(gray: np.ndarray, side: str, band=14) -> np.ndarray:
    if side == "left":
        return gray[:, :band].astype(np.float32).mean(axis=1)
    if side == "right":
        return gray[:, -band:].astype(np.float32).mean(axis=1)
    if side == "bottom":
        return gray[-band:, :].astype(np.float32).mean(axis=0)
    if side == "top":
        return gray[:band, :].astype(np.float32).mean(axis=0)
    raise ValueError(side)

def rising_edges(prof: np.ndarray, thr: float, min_gap=8) -> np.ndarray:
    b = (prof > thr).astype(np.int8)
    idx = np.where((b[1:] == 1) & (b[:-1] == 0))[0] + 1
    if len(idx) == 0:
        return idx
    keep = [idx[0]]
    for x in idx[1:]:
        if x - keep[-1] >= min_gap:
            keep.append(x)
    return np.asarray(keep, dtype=int)

def acorr_period(prof: np.ndarray, min_lag=15, max_lag=250) -> float | None:
    x = prof - prof.mean()
    if x.std() < 1e-6:
        return None
    max_lag = min(max_lag, len(x) // 3)
    if max_lag <= min_lag:
        return None
    c = np.correlate(x, x, mode="full")
    mid = len(c) // 2
    seg = c[mid + min_lag: mid + max_lag]
    if len(seg) == 0:
        return None
    return float(min_lag + int(np.argmax(seg)))

def ambrosm_test_scale(path: Path, a: np.ndarray) -> dict:
    """Port of AmbrosM collect_metadata for TEST images. Returns px_per_cm + bbox."""
    h, w = a.shape[:2]
    filetype = path.suffix.lower().lstrip(".")
    px_per_cm = -1.0
    l = t = r = b = -1
    note = ""
    try:
        if filetype == "png":
            first_tick = int(np.argmax(a[:, 6].mean(axis=-1) > 50)) if a.ndim == 3 else int(np.argmax(a[:, 6] > 50))
            col = a[:, 6].mean(axis=-1) if a.ndim == 3 else a[:, 6]
            col9 = a[:, 9].mean(axis=-1) if a.ndim == 3 else a[:, 9]
            second_minor = 150 + int(np.argmax(col[150:] > 50))
            second_major = 150 + int(np.argmax(col9[150:] > 50))
            last_tick = len(a) - 1 - int(np.argmax(col[::-1] > 50))
            if (second_major - first_tick) < 3 * (second_minor - first_tick):
                px_per_cm = float(second_major - first_tick)  # 10 mm major
            else:
                px_per_cm = float(second_minor - first_tick)  # 10 mm minor
            hw = a.shape[1] // 2
            s = a[:, hw:].sum(axis=(0, 2)) if a.ndim == 3 else a[:, hw:].sum(axis=0)
            w2 = int(np.argmin(s))
            l, t, r, b = hw - w2, first_tick, hw + w2, last_tick
            note = "png"
        elif a.shape[:2] == (800, 1200) and a.ndim == 3:
            if (a[87, 1147:1157] == 175).all() or (a[87, 1147:1157].mean() > 150):
                col = a[:, 1150].mean(axis=-1)
                first_tick = int(np.argmax(col > 50))
                second_major = first_tick + 20 + int(np.argmax(col[first_tick + 20:] > 50))
                last_tick = len(a) - 1 - int(np.argmax(col[::-1] > 50))
                n_ticks = (last_tick - first_tick) / max(second_major - first_tick, 1)
                n = int(round(n_ticks))
                table = {7: (142, 91, 1058), 8: (163, 91, 1037), 9: (211, 91, 989),
                         10: (249, 91, 951), 11: (282, 91, 918), 12: (308, 91, 892),
                         13: (331, 91, 869), 14: (349, 91, 851), 15: (142, 91, 1058)}
                if n <= 14:
                    px_per_cm = (last_tick - first_tick) / n * 2  # 2 ticks / cm
                elif n == 15:
                    px_per_cm = (last_tick - first_tick) / 3
                if n in table:
                    l, t, r = table[n]
                    b = last_tick
                note = f"800x1200_right n={n}"
            elif (a[42, 67:74, 0] > 115).all():
                px_per_cm = (783 - 42) / 5
                l, t, r, b = 171, 42, 1029, 798
                note = "800x1200_left"
            else:
                note = "800x1200_unparsed"
        elif a.shape[:2] == (644, 1088):
            px_per_cm = 630.5 / 5
            l, t, r, b = 140, 0, 947, 643
            note = "644x1088"
        elif a.shape[0] in (512, 513) and a.ndim == 3:
            px_per_cm = (442 - 53) / 5
            l, t, r, b = 0, 0, w, h - 10
            note = f"h{a.shape[0]}_bottom"
        elif a.shape[0] == 853 and a.ndim == 2:
            if a[-5, 100] == 170 and a[-5, 934] == 170:
                px_per_cm = (934 - 100) / 5
            elif a[-5, 44] == 170 and a[-5, 879] == 170:
                px_per_cm = (879 - 44) / 5
            l, t, r, b = 0, 0, w, h
            note = "h853"
        else:
            note = f"unknown {a.shape}"
    except Exception as e:
        note = f"err {e}"
    return dict(px_per_cm=float(px_per_cm), l=int(l), t=int(t), r=int(r), b=int(b), scale_note=note)

def generic_scale(gray: np.ndarray) -> dict:
    """Autocorr + edge ticks on all four sides. Returns candidate px_per_cm."""
    out = {}
    for side in ("left", "right", "bottom", "top"):
        prof = tick_profile(gray, side)
        thr = max(40.0, float(np.percentile(prof, 85)))
        edges = rising_edges(prof, thr)
        period = acorr_period(prof)
        gaps = np.diff(edges) if len(edges) >= 3 else np.array([])
        med_gap = float(np.median(gaps)) if len(gaps) else np.nan
        out[f"{side}_n"] = int(len(edges))
        out[f"{side}_med_gap"] = med_gap
        out[f"{side}_acorr"] = period if period is not None else np.nan
    # prefer the side with the most regular gaps
    best_side, best_gap, best_n = None, np.nan, 0
    for side in ("left", "right", "bottom"):
        g, n = out[f"{side}_med_gap"], out[f"{side}_n"]
        if n >= 4 and np.isfinite(g) and g > 8:
            if best_side is None or n > best_n:
                best_side, best_gap, best_n = side, g, n
    out["generic_side"] = best_side or ""
    out["generic_gap_px"] = float(best_gap) if best_side else np.nan
    # Host: many bottom ticks are 1 cm. Left/right often 5 or 10 mm.
    # We leave the physical unit as a hypothesis; AmbrosM is the test authority.
    return out

scale_rows = []
test_paths = splits["test"]
print(f"scale-parsing {len(test_paths)} test images")
for i, p in enumerate(test_paths):
    rgb = load_rgb(p)
    gray = rgb.mean(axis=2).astype(np.uint8) if rgb.ndim == 3 else rgb
    rec = dict(name=p.name, height=gray.shape[0], width=gray.shape[1], ndim=rgb.ndim)
    rec.update(ambrosm_test_scale(p, rgb if rgb.ndim == 3 else gray))
    rec.update(generic_scale(gray))
    rec["px_per_mm"] = rec["px_per_cm"] / 10.0 if rec["px_per_cm"] > 0 else np.nan
    rec["fov_w_mm"] = (rec["r"] - rec["l"]) / rec["px_per_mm"] if rec["px_per_mm"] > 0 and rec["r"] > rec["l"] else np.nan
    rec["fov_h_mm"] = (rec["b"] - rec["t"]) / rec["px_per_mm"] if rec["px_per_mm"] > 0 and rec["b"] > rec["t"] else np.nan
    rec["pixel_aspect_if_square"] = 1.0
    scale_rows.append(rec)
    if (i + 1) % 50 == 0:
        print(f"  {i+1}/{len(test_paths)}", flush=True)

S = pd.DataFrame(scale_rows)
print("\nAmbrosM coverage: px_per_cm>0  "
      f"{(S.px_per_cm>0).mean():.1%}  bbox {(S.l>=0).mean():.1%}")
print(S[["name", "height", "width", "px_per_cm", "px_per_mm", "fov_w_mm", "fov_h_mm", "scale_note"]]
      .describe(include="all").transpose().head(20))
print("\npx_per_mm by AmbrosM note:")
print(S.groupby("scale_note")["px_per_mm"].agg(["count", "mean", "std", "min", "max"]).to_string())
S.to_csv(OUT / "test_scale.csv", index=False)

# sanity: PNG FOV width should be ~28.5 mm; 800x1200 right-tick ~57.5 mm
print("\nFOV width mm by note (AmbrosM asserts 28.5 / 57.5):")
print(S.groupby("scale_note")["fov_w_mm"].agg(["count", "mean", "min", "max"]).to_string())
