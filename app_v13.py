"""
Oligonucleotide Diastereomer Analyzer  v13
==========================================
Tab 1 — Raw Data       : Sequence + Rp/Sp input + Peak Merging + LC chromatogram + LC params
Tab 2 — Data Processing: CLR statistics for current settings
Tab 3 — Scenario Manager: Save / label Reference vs Sample
Tab 4 — Comparison Dashboard: % area, CLR, Euclidean distance, contribution analysis
"""

import streamlit as st
import numpy as np
import matplotlib
matplotlib.rcParams['font.family'] = 'DejaVu Sans'
matplotlib.rcParams['font.sans-serif'] = ['DejaVu Sans']
import matplotlib.pyplot as plt
import pandas as pd
from io import BytesIO

def show_fig(fig, dpi=100):
    """Save figure to BytesIO and display with st.image — avoids Streamlit pyplot DPI bug."""
    from io import BytesIO as _BytesIO
    buf = _BytesIO()
    fig.savefig(buf, format="png", dpi=dpi)
    buf.seek(0)
    import streamlit as _st
    _st.image(buf, use_container_width=True)
    import matplotlib.pyplot as _plt
    _plt.close(fig)


# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Oligo DS Analyzer v13",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Colour palette ────────────────────────────────────────────────────────────
REF_COL = "#185FA5"
SMP_COL = "#D85A30"
HI_COL  = "#E24B4A"
OK_COL  = "#639922"
NEU_COL = "#888780"
BG_COL  = "#F8FAFC"

# ── Session state ─────────────────────────────────────────────────────────────
if "scenarios" not in st.session_state:
    st.session_state.scenarios = {}
if "reference" not in st.session_state:
    st.session_state.reference = None

# ══════════════════════════════════════════════════════════════════════════════
# HELPER FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════════

def parse_sequence(seq_str):
    parts = [p.strip() for p in seq_str.split("-")]
    return sum(1 for p in parts if p.upper() == "PS")

def build_combos(n_ps, rp_vals):
    combos = []
    for mask in range(2 ** n_ps):
        label, prob, rc = "", 1.0, 0
        for b in range(n_ps):
            is_r = bool((mask >> (n_ps - 1 - b)) & 1)
            label += "R" if is_r else "S"
            rp = rp_vals[b] / 100.0
            prob *= rp if is_r else (1.0 - rp)
            if is_r:
                rc += 1
        combos.append({"label": label, "prob": prob, "rp_count": rc})
    df = pd.DataFrame(combos)
    df = df.sort_values(["rp_count", "label"], ascending=[False, True]).reset_index(drop=True)
    df["pct"] = df["prob"] / df["prob"].sum() * 100
    return df

def apply_merge(combo_df, merge_map):
    combo_df = combo_df.copy()
    combo_df["peak_id"] = combo_df["label"].map(merge_map)
    merged = (combo_df.groupby("peak_id")["pct"]
              .sum().reset_index()
              .sort_values("peak_id").reset_index(drop=True))
    merged["peak_label"] = merged["peak_id"].apply(lambda x: f"Peak {x}")
    return merged

def clr_calc(pcts):
    arr = np.array(pcts, dtype=float)
    if np.any(arr <= 0):
        return None, None
    gm = np.exp(np.mean(np.log(arr)))
    return np.log(arr / gm), gm

def euclidean_distance(clr_a, clr_b):
    delta  = np.array(clr_b) - np.array(clr_a)
    d2     = delta ** 2
    dist   = float(np.sqrt(d2.sum()))
    contrib = d2 / d2.sum() * 100
    return dist, delta, contrib

def simulate_lc(peaks_df, rs, sigma):
    spacing = rs * 4 * sigma
    n_peaks = len(peaks_df)
    n_pts   = max(700, int(50 + n_peaks * spacing + 6 * sigma))
    t = np.arange(n_pts, dtype=float)
    y = np.zeros(n_pts)
    centers = [50 + i * spacing for i in range(n_peaks)]
    for i, row in peaks_df.iterrows():
        amp = row["pct"] / 100.0
        c   = centers[i]
        y  += amp * np.exp(-0.5 * ((t - c) / sigma) ** 2)
    return t, y, centers

def draw_lc(peaks_df, rs, sigma):
    t, y, centers = simulate_lc(peaks_df, rs, sigma)
    n_peaks = len(peaks_df)
    fig_w   = max(12, n_peaks * 0.9)
    fig, ax = plt.subplots(figsize=(fig_w, 5.0), dpi=100, facecolor=BG_COL)
    ax.set_facecolor(BG_COL)
    ax.plot(t, y, color=REF_COL, lw=1.5)
    ax.fill_between(t, y, alpha=0.08, color=REF_COL)
    for i, row in peaks_df.iterrows():
        c   = centers[i]
        top = float(row["pct"]) / 100
        ax.text(c, top + max(y) * 0.18,
                f"{row['peak_label']}\n{row['pct']:.1f}%",
                ha="center", va="bottom", fontsize=8,
                fontweight="bold", color=REF_COL, linespacing=1.4)
    ax.set_xlim(0, len(t) - 1)
    ax.set_ylim(0, max(y) * 1.40)
    ax.axis("off")
    show_fig(fig)

# ══════════════════════════════════════════════════════════════════════════════
# TABS
# ══════════════════════════════════════════════════════════════════════════════

st.caption("Oligo DS Analyzer — v13")

tab_raw, tab_proc, tab_sc, tab_cmp = st.tabs([
    "🔬 Raw Data",
    "📈 Data Processing",
    "📁 Scenario Manager",
    "📊 Comparison Dashboard",
])

# ─────────────────────────────────────────────────────────────────────────────
# TAB 1 — Raw Data
# ─────────────────────────────────────────────────────────────────────────────
with tab_raw:
    st.subheader("Raw Data — Sequence, Rp/Sp & LC Simulation")

    # ── Sequence input ────────────────────────────────────────────────────────
    st.markdown("#### 1. Sequence")
    seq_input = st.text_input(
        "Sequence (nucleotides separated by -, PS = phosphorothioate linkage)",
        value="mA-PS-mC-PS-mG-PS-mA-PS-mC",
        help="Example: mA-PS-mC-PS-mG",
        key="seq_input"
    )
    n_ps = parse_sequence(seq_input)

    if n_ps == 0:
        st.warning("No PS linkages detected. Please check the sequence format.")
        st.stop()

    st.info(f"Detected **{n_ps}** PS linkages → **{2**n_ps}** theoretical DS combinations")

    # ── Rp/Sp number inputs ───────────────────────────────────────────────────
    st.markdown("#### 2. Rp/Sp Ratio per PS Linkage")
    st.caption("Enter the % Rp for each PS linkage (Sp % = 100 − Rp %)")

    rp_vals = []
    cols_rp = st.columns(min(n_ps, 6))
    for i in range(n_ps):
        with cols_rp[i % len(cols_rp)]:
            rp = st.number_input(
                f"PS {i+1} Rp %",
                min_value=1, max_value=99, value=50, step=1,
                key=f"rp_{i}",
                help=f"Position {i+1} from 5' end"
            )
            st.caption(f"Sp = {100 - rp}%")
            rp_vals.append(rp)

    # ── Build combos ──────────────────────────────────────────────────────────
    combo_df = build_combos(n_ps, rp_vals)

    with st.expander("Show all DS combinations", expanded=False):
        disp = combo_df[["label", "rp_count", "pct"]].copy()
        disp.columns = ["DS Combination", "Rp Count", "Theoretical % Area"]
        disp["Theoretical % Area"] = disp["Theoretical % Area"].map("{:.2f}%".format)
        st.dataframe(disp, use_container_width=True, hide_index=True)

    # ── Peak Merging ──────────────────────────────────────────────────────────
    st.markdown("#### 3. Peak Merging")

    if n_ps > 6:
        st.warning(
            "Peak Merging is disabled for sequences with more than 6 PS linkages "
            "(2^n > 64 combinations). Each DS combination is treated as an individual peak."
        )
        enable_merge = False
    else:
        st.caption("Assign each DS combination to an observable peak number. Same number = co-eluting (merged).")
        enable_merge = st.toggle("Enable Peak Merging", value=True, key="enable_merge_toggle")

    st.session_state["enable_merge"] = enable_merge

    if enable_merge:
        merge_key = f"merge_map_{n_ps}"
        default_merge = {row["label"]: (i + 1) for i, row in combo_df.iterrows()}
        if merge_key not in st.session_state:
            st.session_state[merge_key] = default_merge.copy()
        for lbl, did in default_merge.items():
            if lbl not in st.session_state[merge_key]:
                st.session_state[merge_key][lbl] = did

        new_merge = {}
        n_cols = min(6, len(combo_df))
        merge_cols = st.columns(n_cols)
        for idx, row in combo_df.iterrows():
            with merge_cols[idx % n_cols]:
                default_val = st.session_state[merge_key].get(row["label"], idx + 1)
                pid = st.number_input(
                    f"{row['label']}",
                    min_value=1, max_value=len(combo_df),
                    value=int(default_val), step=1,
                    key=f"merge_{row['label']}",
                    help=f"Theoretical: {row['pct']:.1f}%"
                )
                new_merge[row["label"]] = pid
        st.session_state[merge_key] = new_merge
        peaks_df = apply_merge(combo_df, new_merge)
    else:
        # No merging: each DS combination is an individual peak
        peaks_df = combo_df[["label", "pct"]].copy()
        peaks_df["peak_id"] = range(1, len(peaks_df) + 1)
        peaks_df["peak_label"] = peaks_df["label"]
        peaks_df = peaks_df.reset_index(drop=True)

    st.markdown("**Merged observable peaks:**")
    prev_cols = st.columns(min(len(peaks_df), 8))
    for i, row in peaks_df.iterrows():
        with prev_cols[i % len(prev_cols)]:
            st.metric(row["peak_label"], f"{row['pct']:.2f}%")

    # ── LC Chromatogram ───────────────────────────────────────────────────────
    st.markdown("#### 4. Simulated LC Chromatogram")
    draw_lc(peaks_df,
            st.session_state.get("rs_val", 1.2),
            st.session_state.get("sigma_val", 7))

    # ── LC Parameters (below chart) ───────────────────────────────────────────
    st.markdown("#### 5. LC Simulation Parameters")
    lc_c1, lc_c2, lc_c3 = st.columns([2, 2, 3])

    with lc_c1:
        rs_val = st.number_input(
            "Resolution (Rs)",
            min_value=0.2, max_value=5.0, value=1.2, step=0.1,
            key="rs_val",
            help="Rs < 1.0 = overlapping | Rs = 1.5 = baseline separation | Rs > 1.5 = fully resolved"
        )
    with lc_c2:
        sigma_val = st.number_input(
            "Peak Width (σ)",
            min_value=1, max_value=30, value=7, step=1,
            key="sigma_val",
            help="Small = narrow peak (high efficiency) | Large = broad peak (low efficiency)"
        )
    with lc_c3:
        if rs_val < 0.6:
            st.error("⚠️ Rs < 0.6: Severe overlap")
        elif rs_val < 1.0:
            st.warning("Rs 0.6–1.0: Partial overlap")
        elif rs_val < 1.5:
            st.success("Rs 1.0–1.5: Near baseline separation")
        else:
            st.info("Rs > 1.5: Fully resolved")

# ─────────────────────────────────────────────────────────────────────────────
# TAB 2 — Data Processing (CLR for current settings)
# ─────────────────────────────────────────────────────────────────────────────
with tab_proc:
    st.subheader("Data Processing — CLR Statistics")

    # Rebuild from session state so this tab always reflects Tab 1 settings
    try:
        combo_df2  = build_combos(n_ps, rp_vals)
        enable_merge2 = st.session_state.get("enable_merge", True)
        if enable_merge2:
            merge_key2 = f"merge_map_{n_ps}"
            merge_map2 = st.session_state.get(merge_key2, {r["label"]: (i+1) for i, r in combo_df2.iterrows()})
            peaks_df2  = apply_merge(combo_df2, merge_map2)
        else:
            peaks_df2 = combo_df2[["label", "pct"]].copy()
            peaks_df2["peak_id"] = range(1, len(peaks_df2) + 1)
            peaks_df2["peak_label"] = peaks_df2["label"]
            peaks_df2 = peaks_df2.reset_index(drop=True)
        pcts2      = peaks_df2["pct"].tolist()
        clr_vals2, gm2 = clr_calc(pcts2)
    except Exception:
        st.warning("Please configure your sequence in the Raw Data tab first.")
        st.stop()

    if clr_vals2 is None:
        st.error("CLR cannot be calculated — a peak area is 0.")
    else:
        m1, m2, m3 = st.columns(3)
        m1.metric("Geometric Mean", f"{gm2:.3f}%")
        m2.metric("Observable Peaks", len(pcts2))
        m3.metric("CLR Sum (≈ 0)", f"{sum(clr_vals2):.6f}")

        st.markdown("---")
        st.markdown("#### CLR Values")

        clr_tbl = peaks_df2[["peak_label", "pct"]].copy()
        clr_tbl["CLR"]    = clr_vals2
        clr_tbl["Status"] = clr_tbl["CLR"].apply(
            lambda v: "▲ Above mean" if v > 0.05 else ("▼ Below mean" if v < -0.05 else "≈ Near mean"))
        clr_tbl.columns = ["Peak", "% Area", "CLR", "Status"]
        clr_tbl["% Area"] = clr_tbl["% Area"].map("{:.2f}%".format)
        clr_tbl["CLR"]    = clr_tbl["CLR"].map("{:+.4f}".format)
        st.caption("CLR status is relative to the Geometric Mean of all peaks")
        st.dataframe(clr_tbl, use_container_width=True, hide_index=True)

        st.markdown("---")
        st.markdown("#### CLR Bar Chart")
        n_bars = len(pcts2)
        fig_c, ax_c = plt.subplots(figsize=(12, 4), dpi=100, facecolor=BG_COL)
        ax_c.set_facecolor(BG_COL)
        bar_colors = [OK_COL if v >= 0 else HI_COL for v in clr_vals2]
        x_pos = list(range(n_bars))
        ax_c.bar(x_pos, clr_vals2, color=bar_colors, alpha=0.85)
        ax_c.axhline(0, color="#333", lw=1.2)
        ax_c.set_xticks(x_pos)
        ax_c.set_xticklabels(peaks_df2["peak_label"].tolist(), fontsize=8, rotation=0, ha="center")
        for i, v in enumerate(clr_vals2):
            ax_c.text(i, v + (0.01 if v >= 0 else -0.02),
                      f"{v:+.3f}", ha="center", fontsize=8, color="#333")
        ax_c.set_ylabel("CLR value", fontsize=9)
        ax_c.spines["top"].set_visible(False)
        ax_c.spines["right"].set_visible(False)
        fig_c.patch.set_facecolor(BG_COL)
        show_fig(fig_c)

        st.markdown("---")
        st.markdown("#### % Peak Area Bar Chart")
        fig_p, ax_p = plt.subplots(figsize=(12, 4), dpi=100, facecolor=BG_COL)
        ax_p.set_facecolor(BG_COL)
        ax_p.bar(x_pos, pcts2, color=REF_COL, alpha=0.85)
        ax_p.set_xticks(x_pos)
        ax_p.set_xticklabels(peaks_df2["peak_label"].tolist(), fontsize=8, rotation=0, ha="center")
        for i, v in enumerate(pcts2):
            ax_p.text(i, v + max(pcts2) * 0.01,
                      f"{v:.1f}%", ha="center", fontsize=8, color="#333")
        ax_p.set_ylabel("% Area", fontsize=9)
        ax_p.spines["top"].set_visible(False)
        ax_p.spines["right"].set_visible(False)
        fig_p.patch.set_facecolor(BG_COL)
        show_fig(fig_p)

# ─────────────────────────────────────────────────────────────────────────────
# TAB 3 — Scenario Manager
# ─────────────────────────────────────────────────────────────────────────────
with tab_sc:
    st.subheader("Scenario Manager")
    st.markdown("Save the current Raw Data settings as a named scenario for comparison.")

    sc_name = st.text_input(
        "Scenario Name",
        value=f"Scenario {chr(65 + len(st.session_state.scenarios))}"
    )

    if st.button("💾 Save Current Settings as Scenario", type="primary"):
        enable_merge_sc = st.session_state.get("enable_merge", True)
        if enable_merge_sc:
            mk = f"merge_map_{n_ps}"
            mm = st.session_state.get(mk, {r["label"]: (i+1) for i, r in combo_df.iterrows()})
            pk = apply_merge(combo_df, mm)
        else:
            pk = combo_df[["label", "pct"]].copy()
            pk["peak_id"] = range(1, len(pk) + 1)
            pk["peak_label"] = pk["label"]
            pk = pk.reset_index(drop=True)
        st.session_state.scenarios[sc_name] = {
            "pcts":    pk["pct"].tolist(),
            "labels":  pk["peak_label"].tolist(),
            "rp_vals": rp_vals.copy(),
            "n_ps":    n_ps,
        }
        if st.session_state.reference is None:
            st.session_state.reference = sc_name
        st.success(f"✅ Saved: '{sc_name}'")

    st.markdown("---")

    if not st.session_state.scenarios:
        st.info("No scenarios saved yet.")
    else:
        st.markdown("#### Saved Scenarios")

        ref_options     = list(st.session_state.scenarios.keys())
        current_ref_idx = ref_options.index(st.session_state.reference) \
                          if st.session_state.reference in ref_options else 0
        new_ref = st.selectbox("🔵 Set as Reference", ref_options, index=current_ref_idx)
        st.session_state.reference = new_ref

        for name, sc in st.session_state.scenarios.items():
            is_ref = (name == st.session_state.reference)
            badge  = "🔵 REF" if is_ref else "🟠 Sample"
            with st.expander(f"{badge}  {name}  ({len(sc['pcts'])} peaks)", expanded=False):
                tbl = pd.DataFrame({
                    "Peak":   sc["labels"],
                    "% Area": [f"{p:.2f}%" for p in sc["pcts"]]
                })
                st.dataframe(tbl, use_container_width=True, hide_index=True)
                if st.button(f"🗑️ Delete '{name}'", key=f"del_{name}"):
                    del st.session_state.scenarios[name]
                    if st.session_state.reference == name:
                        remaining = list(st.session_state.scenarios.keys())
                        st.session_state.reference = remaining[0] if remaining else None
                    st.rerun()

# ─────────────────────────────────────────────────────────────────────────────
# TAB 4 — Comparison Dashboard  (1 Reference vs N Samples + Replicate Groups)
# ─────────────────────────────────────────────────────────────────────────────
with tab_cmp:
    st.subheader("Comparison Dashboard")

    scenarios = st.session_state.scenarios
    if len(scenarios) < 2:
        st.info("Save at least **2 scenarios** in the Scenario Manager to enable comparison.")
        st.stop()

    sc_names = list(scenarios.keys())
    ref_name = st.session_state.reference or sc_names[0]

    # Unified colour palette — index 0 = Reference, 1+ = Samples/Groups (in order)
    SCENARIO_COLORS = [
        "#185FA5", "#D85A30", "#639922", "#7F77DD",
        "#1D9E75", "#E24B4A", "#F5A623", "#8B4513",
    ]

    # ── Reference & Sample selection ──────────────────────────────────────────
    sel_c1, sel_c2 = st.columns([1, 2])
    with sel_c1:
        ref_sel = st.selectbox(
            "🔵 Reference",
            sc_names,
            index=sc_names.index(ref_name) if ref_name in sc_names else 0,
        )
    with sel_c2:
        smp_opts = [n for n in sc_names if n != ref_sel]
        smp_sels = st.multiselect(
            "🟠 Samples (select one or more)",
            smp_opts,
            default=smp_opts[: min(3, len(smp_opts))],
        )

    if not smp_sels:
        st.warning("Please select at least one Sample.")
        st.stop()

    # ── Measurement %RSD ──────────────────────────────────────────────────────
    pct_rsd = st.number_input(
        "Measurement %RSD",
        min_value=0.1, max_value=20.0, value=2.0, step=0.1,
        help="Applied to individual samples. SD = mean × %RSD / 100",
    )
    clr_sd_rsd = pct_rsd / 100.0

    # ── Replicate Groups ──────────────────────────────────────────────────────
    st.markdown("#### Replicate Groups (optional)")
    st.caption("Group multiple scenarios as replicates to calculate mean ± SD")

    n_groups = int(st.number_input(
        "Number of replicate groups", min_value=0, max_value=6, value=0, step=1,
        key="n_rep_groups",
    ))
    groups = []
    for gi in range(n_groups):
        gc1, gc2 = st.columns([1, 2])
        with gc1:
            gname = st.text_input(
                f"Group {gi + 1} name", value=f"Group {gi + 1}", key=f"grp_name_{gi}"
            )
        with gc2:
            gmembers = st.multiselect(
                f"Group {gi + 1} members", smp_sels, key=f"grp_members_{gi}"
            )
        if gname and gmembers:
            groups.append({"name": gname, "members": gmembers})

    assigned = set()
    for g in groups:
        assigned.update(g["members"])
    individual_sels = [s for s in smp_sels if s not in assigned]

    # ── Core setup ────────────────────────────────────────────────────────────
    sc_ref = scenarios[ref_sel]

    def _label_sort_key(lbl):
        try:
            return (0, int(lbl.split()[-1]), lbl)
        except (ValueError, IndexError):
            return (1, 0, lbl)

    all_label_set = set(sc_ref["labels"])
    for sn in smp_sels:
        all_label_set |= set(scenarios[sn]["labels"])
    all_labels = sorted(all_label_set, key=_label_sort_key)
    n_peaks = len(all_labels)

    def get_pcts(sc, labels):
        return [sc["pcts"][sc["labels"].index(l)] if l in sc["labels"] else 0.001
                for l in labels]

    pcts_ref = get_pcts(sc_ref, all_labels)
    clr_ref, gm_ref = clr_calc(pcts_ref)
    if clr_ref is None:
        st.error("CLR calculation failed for Reference: a peak area is 0.")
        st.stop()

    # ── Build plot_items ──────────────────────────────────────────────────────
    # Each item: type | name | color | pcts_mean | pcts_err | clr_mean | clr_err
    #            | dist_mean | dist_err | delta_mean | pct_diff_mean
    color_idx = 1  # index 0 reserved for Reference
    plot_items = []

    for g in groups:
        pcts_list, clr_list, dist_list, delta_list = [], [], [], []
        ok = True
        for m in g["members"]:
            p = get_pcts(scenarios[m], all_labels)
            cv, _ = clr_calc(p)
            if cv is None:
                st.error(f"CLR calculation failed for {m}: a peak area is 0.")
                ok = False
                break
            dv, deltav, _ = euclidean_distance(clr_ref, cv)
            pcts_list.append(p)
            clr_list.append(cv)
            dist_list.append(dv)
            delta_list.append(deltav)
        if not ok:
            st.stop()

        n_m       = len(g["members"])
        pcts_arr  = np.array(pcts_list)
        clr_arr   = np.array(clr_list)
        delta_arr = np.array(delta_list)

        pcts_mean  = pcts_arr.mean(axis=0).tolist()
        clr_mean   = clr_arr.mean(axis=0).tolist()
        delta_mean = delta_arr.mean(axis=0).tolist()
        dist_mean  = float(np.mean(dist_list))

        if n_m > 1:
            pcts_sd   = pcts_arr.std(axis=0, ddof=1).tolist()
            clr_sd_g  = clr_arr.std(axis=0, ddof=1).tolist()
            dist_sd   = float(np.std(dist_list, ddof=1))
        else:                                          # single member → fall back to %RSD
            pcts_sd   = [v * clr_sd_rsd for v in pcts_mean]
            clr_sd_g  = [clr_sd_rsd] * n_peaks
            dist_sd   = float(np.sqrt(n_peaks * 2)) * clr_sd_rsd

        col = SCENARIO_COLORS[color_idx % len(SCENARIO_COLORS)]
        color_idx += 1
        plot_items.append({
            "type":          "group",
            "name":          g["name"],
            "members":       g["members"],
            "color":         col,
            "pcts_mean":     pcts_mean,
            "pcts_err":      [3 * s for s in pcts_sd],
            "clr_mean":      clr_mean,
            "clr_err":       [3 * s for s in clr_sd_g],
            "dist_mean":     dist_mean,
            "dist_err":      3 * dist_sd,
            "delta_mean":    delta_mean,
            "pct_diff_mean": (np.array(pcts_mean) - np.array(pcts_ref)).tolist(),
        })

    for sn in individual_sels:
        p = get_pcts(scenarios[sn], all_labels)
        cv, gm_s = clr_calc(p)
        if cv is None:
            st.error(f"CLR calculation failed for {sn}: a peak area is 0.")
            st.stop()
        dv, deltav, contribv = euclidean_distance(clr_ref, cv)
        dist_sd_ind = float(np.sqrt(n_peaks * 2)) * clr_sd_rsd

        col = SCENARIO_COLORS[color_idx % len(SCENARIO_COLORS)]
        color_idx += 1
        plot_items.append({
            "type":          "individual",
            "name":          sn,
            "color":         col,
            "pcts_mean":     p,
            "pcts_err":      [3 * v * pct_rsd / 100 for v in p],
            "clr_mean":      list(cv),
            "clr_err":       [3 * clr_sd_rsd] * n_peaks,
            "dist_mean":     dv,
            "dist_err":      3 * dist_sd_ind,
            "delta_mean":    list(deltav),
            "pct_diff_mean": (np.array(p) - np.array(pcts_ref)).tolist(),
            "contrib":       contribv,
            "gm":            gm_s,
        })

    if not plot_items:
        st.warning("No samples or groups to compare.")
        st.stop()

    n_items = len(plot_items)
    x       = np.arange(n_peaks)

    # ── 1. Euclidean Distance Overview ────────────────────────────────────────
    st.markdown("---")
    st.markdown("#### Euclidean Distance vs Reference")
    st.caption("Deviation is statistically significant if distance > 3SD")

    dists    = [it["dist_mean"] for it in plot_items]
    dist_err = [it["dist_err"]  for it in plot_items]
    colors_d = [it["color"]     for it in plot_items]

    fig_d, ax_d = plt.subplots(
        figsize=(10, max(2.5, n_items * 0.7 + 1.0)), dpi=100, facecolor=BG_COL
    )
    ax_d.set_facecolor(BG_COL)
    ypos = list(range(n_items))
    ax_d.barh(ypos, dists, color=colors_d, alpha=0.85,
              xerr=[np.zeros(n_items), dist_err],
              error_kw=dict(ecolor="#555", capsize=4, lw=1.2))
    ax_d.set_yticks(ypos)
    ax_d.set_yticklabels([it["name"] for it in plot_items], fontsize=9)
    ax_d.invert_yaxis()
    max_d = max(d + e for d, e in zip(dists, dist_err)) if dists else 1
    for i, (d, e) in enumerate(zip(dists, dist_err)):
        ax_d.text(d + e + max_d * 0.02, i,
                  f"{d:.4f} ±{e:.4f} (3SD)", va="center", fontsize=8, color="#333")
    ax_d.set_xlabel("Euclidean Distance (CLR space)", fontsize=9)
    ax_d.spines["top"].set_visible(False)
    ax_d.spines["right"].set_visible(False)
    fig_d.patch.set_facecolor(BG_COL)
    show_fig(fig_d)

    # ── 2. % Peak Area Grouped Bar Chart ──────────────────────────────────────
    st.markdown("---")
    st.markdown("#### % Peak Area Comparison")
    n_bars  = 1 + n_items
    total_w = 0.75
    bw      = total_w / n_bars
    offsets = np.linspace(-(total_w / 2) + bw / 2, (total_w / 2) - bw / 2, n_bars)
    fig_w   = max(10, n_peaks * (n_bars * 0.35 + 0.4))
    err_kw  = dict(ecolor="#555", capsize=3, lw=1.0)

    sd3_ref_pct = [3 * v * pct_rsd / 100 for v in pcts_ref]

    fig_pct, ax_pct = plt.subplots(figsize=(fig_w, 4.5), dpi=100, facecolor=BG_COL)
    ax_pct.set_facecolor(BG_COL)
    ax_pct.bar(x + offsets[0], pcts_ref, bw, color=SCENARIO_COLORS[0], alpha=0.85,
               label=ref_sel, yerr=sd3_ref_pct, error_kw=err_kw)
    for j, it in enumerate(plot_items):
        ax_pct.bar(x + offsets[1 + j], it["pcts_mean"], bw, color=it["color"], alpha=0.85,
                   label=it["name"], yerr=it["pcts_err"], error_kw=err_kw)
    ax_pct.set_xticks(x)
    ax_pct.set_xticklabels(all_labels, fontsize=8, rotation=0, ha="center")
    ax_pct.set_ylabel("% Peak Area", fontsize=9)
    ax_pct.legend(fontsize=8, framealpha=0.7)
    ax_pct.spines["top"].set_visible(False)
    ax_pct.spines["right"].set_visible(False)
    fig_pct.patch.set_facecolor(BG_COL)
    show_fig(fig_pct)

    # ── 3. CLR Grouped Bar Chart ──────────────────────────────────────────────
    st.markdown("---")
    st.markdown("#### CLR Value Comparison")
    sd3_ref_clr = np.full(n_peaks, 3 * clr_sd_rsd)

    fig_clr, ax_clr = plt.subplots(figsize=(fig_w, 4.5), dpi=100, facecolor=BG_COL)
    ax_clr.set_facecolor(BG_COL)
    ax_clr.bar(x + offsets[0], clr_ref, bw, color=SCENARIO_COLORS[0], alpha=0.85,
               label=ref_sel, yerr=sd3_ref_clr, error_kw=err_kw)
    for j, it in enumerate(plot_items):
        ax_clr.bar(x + offsets[1 + j], it["clr_mean"], bw, color=it["color"], alpha=0.85,
                   label=it["name"], yerr=it["clr_err"], error_kw=err_kw)
    ax_clr.axhline(0, color="#333", lw=1.2)
    ax_clr.set_xticks(x)
    ax_clr.set_xticklabels(all_labels, fontsize=8, rotation=0, ha="center")
    ax_clr.set_ylabel("CLR value", fontsize=9)
    ax_clr.legend(fontsize=8, framealpha=0.7)
    ax_clr.spines["top"].set_visible(False)
    ax_clr.spines["right"].set_visible(False)
    fig_clr.patch.set_facecolor(BG_COL)
    show_fig(fig_clr)

    # ── 4. ΔCLR Contribution Table ────────────────────────────────────────────
    st.markdown("---")
    st.markdown("#### ΔCLR by Peak (vs Reference)")
    dclr_data = {"Peak": all_labels, "REF %": [f"{v:.2f}%" for v in pcts_ref]}
    for it in plot_items:
        dclr_data[f"{it['name']} ΔCLR"] = [f"{v:+.4f}" for v in it["delta_mean"]]
    dclr_df = pd.DataFrame(dclr_data)
    delta_matrix = np.column_stack([it["delta_mean"] for it in plot_items])
    if delta_matrix.ndim == 1:
        delta_matrix = delta_matrix.reshape(-1, 1)
    max_abs = np.max(np.abs(delta_matrix), axis=1)
    dclr_df["_sort"] = max_abs
    dclr_df = dclr_df.sort_values("_sort", ascending=False).drop(columns="_sort").reset_index(drop=True)
    st.dataframe(dclr_df, use_container_width=True, hide_index=True)

    # ── 5. Summary by Sample / Group ─────────────────────────────────────────
    st.markdown("---")
    st.markdown("#### Summary by Sample / Group")
    for it in plot_items:
        top_idx  = int(np.argmax(np.abs(it["delta_mean"])))
        top_peak = all_labels[top_idx]
        top_dclr = it["delta_mean"][top_idx]
        top_dpct = it["pct_diff_mean"][top_idx]
        type_tag = " *(group)*" if it["type"] == "group" else ""
        c1, c2, c3, c4, c5 = st.columns([2, 1.2, 1.5, 1.2, 1.2])
        c1.markdown(f"**{it['name']}**{type_tag}")
        c2.metric("Distance", f"{it['dist_mean']:.4f}")
        c3.metric("Top Peak", top_peak)
        c4.metric("ΔCLR", f"{top_dclr:+.4f}")
        c5.metric("Δ%", f"{top_dpct:+.2f}%")
        exp_label = f"Full per-peak data — {it['name']}"
        if it["type"] == "group":
            exp_label += f"  (members: {', '.join(it['members'])})"
        with st.expander(exp_label):
            full_df = pd.DataFrame({
                "Peak":                 all_labels,
                "REF %":                [f"{v:.2f}%" for v in pcts_ref],
                f"{it['name']} mean %": [f"{v:.2f}%" for v in it["pcts_mean"]],
                "Δ%":                   [f"{v:+.2f}%" for v in it["pct_diff_mean"]],
                "ΔCLR":                 [f"{v:+.4f}" for v in it["delta_mean"]],
            })
            st.dataframe(full_df, use_container_width=True, hide_index=True)

    # ── Excel export ──────────────────────────────────────────────────────────
    st.markdown("---")
    buf = BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as writer:
        dclr_df.to_excel(writer, sheet_name="ΔCLR_summary", index=False)
        clr_sheet = pd.DataFrame({"Peak": all_labels, "CLR_REF": clr_ref})
        for it in plot_items:
            clr_sheet[f"CLR_{it['name']}"] = it["clr_mean"]
        clr_sheet.to_excel(writer, sheet_name="CLR_values", index=False)
        pd.DataFrame({
            "Name":               [it["name"] for it in plot_items],
            "Type":               [it["type"] for it in plot_items],
            "Euclidean_Distance": [it["dist_mean"] for it in plot_items],
        }).to_excel(writer, sheet_name="Distances", index=False)
    buf.seek(0)
    st.download_button(
        "⬇️ Download Excel Report",
        data=buf,
        file_name=f"DS_comparison_{ref_sel}.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )
