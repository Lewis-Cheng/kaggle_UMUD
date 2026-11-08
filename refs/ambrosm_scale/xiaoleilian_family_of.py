def family_of(r) -> str:
    h, w = r.get("height", np.nan), r.get("width", np.nan)
    mode = r.get("mode", "")
    suf = r.get("suffix", "")
    if suf == ".png":
        return "png_lumify"
    if (h, w) == (800, 1200):
        rt = r.get("fp_right_tick", np.nan)
        lt = r.get("fp_left_tick", np.nan)
        if pd.notna(rt) and rt > 100:
            return "rgb_800x1200_right_ticks"
        if pd.notna(lt) and lt > 100:
            return "rgb_800x1200_left_ticks"
        return "rgb_800x1200_other"
    if (h, w) == (644, 1088):
        return "rgb_644x1088"
    if h in (512, 513):
        return f"h{int(h)}_bottom_ticks"
    if h == 853 and mode in ("L", "I;16", "I"):
        return "gray_853_bottom_ticks"
    if pd.notna(h) and pd.notna(w):
        return f"other_{int(h)}x{int(w)}_{mode}"
    return "unknown"

C["family"] = C.apply(family_of, axis=1)
fam = C.groupby(["split", "family"]).size().rename("n").reset_index()
print(fam.sort_values(["split", "n"], ascending=[True, False]).to_string(index=False))

print("\nTrain vs test family overlap:")
tr = set(C.loc[C.split.isin(["apo_img", "fasc_img"]), "family"])
te = set(C.loc[C.split.eq("test"), "family"])
print("  train families", sorted(tr))
print("  test  families", sorted(te))
print("  test-only     ", sorted(te - tr))
print("  train-only    ", sorted(tr - te)[:20], "..." if len(tr-te) > 20 else "")
