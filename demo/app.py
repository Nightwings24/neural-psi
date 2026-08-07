"""
Neural-PSI - interactive pipeline demonstration.

Run from this directory:
    ~/bt-demo/bin/streamlit run app.py
"""

import time

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

import pipeline as pl

st.set_page_config(
    page_title="Neural-PSI - Privacy-Preserving Fingerprint Authentication",
    layout="wide",
    initial_sidebar_state="expanded",
)

ACCENT = "#1f4e79"
MUTED = "#5b6b7c"
OK = "#1b7f5a"
BAD = "#b23b3b"

st.markdown(
    f"""
    <style>
      .block-container {{ padding-top: 2.1rem; padding-bottom: 3rem; max-width: 1420px; }}
      h1, h2, h3, h4 {{ color: #16283d; font-weight: 600; letter-spacing: -0.01em; }}
      .np-title {{ font-size: 1.9rem; font-weight: 650; color: #16283d; margin-bottom: .15rem; }}
      .np-sub {{ color: {MUTED}; font-size: 1.02rem; margin-bottom: .3rem; }}
      .np-flow {{ font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
                  color: {ACCENT}; background: #f4f6f9; border: 1px solid #e2e8f0;
                  border-radius: 6px; padding: .55rem .9rem; font-size: .86rem;
                  display: inline-block; margin-top: .35rem; }}
      .np-stage {{ text-transform: uppercase; letter-spacing: .08em; font-size: .74rem;
                   color: {MUTED}; font-weight: 600; margin-bottom: .35rem; }}
      .np-verdict-ok {{ border-left: 5px solid {OK}; background: #f1f8f4;
                        padding: 1rem 1.2rem; border-radius: 6px; }}
      .np-verdict-bad {{ border-left: 5px solid {BAD}; background: #fdf3f3;
                         padding: 1rem 1.2rem; border-radius: 6px; }}
      .np-verdict-h {{ font-size: 1.3rem; font-weight: 650; margin-bottom: .2rem; }}
      .np-mono {{ font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
                  font-size: .76rem; color: #33475b; word-break: break-all; }}
      div[data-testid="stMetricValue"] {{ font-size: 1.3rem; color: #16283d; }}
      div[data-testid="stMetricLabel"] {{ color: {MUTED}; }}
      hr {{ margin: 1.1rem 0; border-color: #e8edf3; }}
    </style>
    """,
    unsafe_allow_html=True,
)


# --------------------------------------------------------------------------- #
# Cached resources
# --------------------------------------------------------------------------- #
@st.cache_resource(show_spinner=False)
def get_model():
    return pl.load_model()


@st.cache_data(show_spinner=False)
def get_index():
    real, altered = pl.index_dataset()
    return real, altered, pl.held_out_identities()


@st.cache_resource(show_spinner=False)
def get_quantizer(n_calib: int, seed: int):
    """Public constants (mu, P, tau) fitted on calibration identities only."""
    model, device = get_model()
    real, _altered, test_ids = get_index()
    test_set = set(test_ids)
    calib_ids = sorted(i for i in real if i not in test_set)
    rng = np.random.default_rng(seed)
    pick = rng.choice(len(calib_ids), size=min(n_calib, len(calib_ids)), replace=False)
    quant, _ = pl.fit_quantizer(model, device,
                                [real[calib_ids[i]] for i in sorted(pick)], seed=seed)
    return quant


@st.cache_data(show_spinner=False)
def usable_fingers():
    """Held-out fingers that have both an enrolment (Real) and a probe (Altered) capture."""
    real, altered, test_ids = get_index()
    return [i for i in test_ids if i in real and i in altered]


@st.cache_data(show_spinner=False)
def build_database(db_size: int, seed: int, n_calib: int):
    """Enrol `db_size` held-out fingers: Real capture -> embedding -> 128-bit code."""
    model, device = get_model()
    real, _altered, _t = get_index()
    quant = get_quantizer(n_calib, seed)

    pool = usable_fingers()
    rng = np.random.default_rng(seed)
    pick = rng.choice(len(pool), size=min(db_size, len(pool)), replace=False)
    enrolled = [pool[i] for i in sorted(pick)]

    emb = pl.embed_paths(model, device, [real[f] for f in enrolled])
    codes = np.stack([quant.transform(e) for e in emb]).astype(np.uint8)
    outsiders = [f for f in pool if f not in set(enrolled)]
    return enrolled, codes, outsiders


@st.cache_data(show_spinner=False)
def encode_probe(path: str, n_calib: int, seed: int):
    model, device = get_model()
    quant = get_quantizer(n_calib, seed)
    img = pl.load_gray(path)
    t0 = time.perf_counter()
    emb = pl.embed_images(model, device, img[None])[0]
    t_cnn = (time.perf_counter() - t0) * 1000
    t0 = time.perf_counter()
    code = quant.transform(emb).astype(np.uint8)
    t_quant = (time.perf_counter() - t0) * 1e6
    return emb, code, t_cnn, t_quant


# --------------------------------------------------------------------------- #
# Sidebar
# --------------------------------------------------------------------------- #
with st.sidebar:
    st.markdown("### Configuration")

    db_size = st.slider("Fingers enrolled on server", 10, 200, 40, step=10,
                        help="Held-out identities enrolled as 128-bit codes.")

    engine = st.radio(
        "Stage 3 backend",
        ["Closed-form protocol model", "Real cryptography (flash-psi)"],
        help=("The closed-form model is the exact sub-sampling analysis used in the "
              "evaluation. The real option executes the actual masked-OPRF, garbled "
              "circuit, VOLE and Shamir protocol binary."),
    )

    st.markdown("**Operating point**")
    weight = st.slider("Mask weight (w)", 8, 40, 14, step=2,
                       help="Bit positions selected by each sub-sampling mask. "
                            "Larger w is stricter.")
    t_thr = st.slider("Shamir threshold (t)", 2, 4, 2,
                      help="Sub-samples that must collide before the record is released.")
    T_masks = st.select_slider("Sub-samples (T)", options=[32, 64, 128], value=64)

    st.caption(
        "Default w = 14, t = 2, T = 64 is the operating point validated in the report: "
        "true-accept 99 percent at false-accept 4.5 percent per comparison, confirmed "
        "against the real protocol binary."
    )

    st.divider()
    seed = st.number_input("Enrolment seed", 0, 9999, 7, step=1)
    N_CALIB = 300
    _m, _dev = get_model()
    st.caption(f"Feature extractor: 224 x 224 model on {_dev.upper()}  |  "
               f"quantiser calibrated on {N_CALIB} non-test identities")


# --------------------------------------------------------------------------- #
# Header
# --------------------------------------------------------------------------- #
st.markdown('<div class="np-title">Neural-PSI</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="np-sub">Privacy-preserving fingerprint authentication: a convolutional '
    'feature extractor, a Super-Bit binarisation bridge, and Fuzzy-Labelled Private Set '
    'Intersection in place of homomorphic encryption.</div>',
    unsafe_allow_html=True,
)
st.markdown(
    '<div class="np-flow">fingerprint image &nbsp;&rarr;&nbsp; CNN &nbsp;&rarr;&nbsp; '
    'e &isin; R<sup>16</sup> &nbsp;&rarr;&nbsp; Super-Bit &nbsp;&rarr;&nbsp; '
    'b &isin; {0,1}<sup>128</sup> &nbsp;&rarr;&nbsp; FLPSI &nbsp;&rarr;&nbsp; '
    'label or reject</div>',
    unsafe_allow_html=True,
)
st.write("")

with st.spinner("Loading feature extractor, calibrating quantiser, enrolling database..."):
    real_idx, altered_idx, _test_ids = get_index()
    enrolled, db_codes, outsiders = build_database(db_size, int(seed), N_CALIB)

m1, m2, m3, m4 = st.columns(4)
m1.metric("Enrolled fingers", f"{len(enrolled)}")
m2.metric("Template size", "16 bytes", help="One 128-bit code per finger")
m3.metric("Server storage", f"{len(enrolled) * 16 / 1024:.2f} KB")
m4.metric("Operating point", f"w={weight}, t={t_thr}, T={T_masks}")

tab_db, tab_auth, tab_about = st.tabs(["Server database", "Authentication", "How it works"])


# --------------------------------------------------------------------------- #
# Tab 1 - enrolled database
# --------------------------------------------------------------------------- #
with tab_db:
    st.markdown("#### Enrolled records on the server")
    st.markdown(
        f"Each enrolled finger is reduced to a **{pl.D_BITS}-bit code (16 bytes)**. The server "
        "holds nothing else: no image, no floating-point embedding, and no per-user key. "
        "The homomorphic-encryption baseline stores roughly 1.67 KB per user plus a one-time "
        "117 MB key."
    )

    show_n = st.slider("Preview records", 4, min(24, len(enrolled)),
                       min(8, len(enrolled)), key="prev")
    cols = st.columns(4)
    for k in range(show_n):
        with cols[k % 4]:
            st.image(pl.fingerprint_image(real_idx[enrolled[k]], 150), width=150)
            st.markdown(f"**Record {k}** &nbsp; `{enrolled[k]}`")
            st.markdown(
                f'<span class="np-mono">{pl.bits_to_str(db_codes[k])[:32]}…</span>',
                unsafe_allow_html=True)
            st.write("")

    st.divider()
    st.markdown("##### Stored template table")
    st.dataframe(
        pd.DataFrame({
            "Record": range(len(enrolled)),
            "Identity label": enrolled,
            "128-bit template (truncated)": [pl.bits_to_str(c)[:48] + "…" for c in db_codes],
            "Bits set": [int(c.sum()) for c in db_codes],
        }),
        hide_index=True, width="stretch", height=320,
    )


# --------------------------------------------------------------------------- #
# Tab 2 - authentication
# --------------------------------------------------------------------------- #
with tab_auth:
    st.markdown("#### Authenticate against the enrolled database")
    st.caption("The user presents a finger **and claims an identity**; the protocol verifies "
               "the presented finger against that claimed record.")

    left, right = st.columns([1, 1.15])

    with left:
        scenario = st.radio(
            "Scenario",
            ["Genuine - an enrolled finger, fresh capture",
             "Impostor - a finger that is not enrolled"],
            key="scenario",
        )
        genuine = scenario.startswith("Genuine")

        if genuine:
            probe_id = st.selectbox(
                "Finger presented at the reader", enrolled,
                format_func=lambda f: f"Record {enrolled.index(f)} · {f}")
            claimed = probe_id
        else:
            if not outsiders:
                st.error("No unenrolled fingers available. Reduce the database size.")
                st.stop()
            probe_id = st.selectbox("Finger presented at the reader", outsiders,
                                    format_func=lambda f: f"Unenrolled · {f}")
            claimed = st.selectbox(
                "Identity being claimed", enrolled,
                format_func=lambda f: f"Record {enrolled.index(f)} · {f}",
                help="The impostor asserts they are this enrolled user.")

        claimed_ix = enrolled.index(claimed)

        variants = altered_idx.get(probe_id, [])
        vix = st.selectbox("Capture distortion", range(len(variants)),
                           format_func=lambda i: pl.VARIANT_LABEL.get(variants[i][0],
                                                                      variants[i][0])
                           ) if variants else None
        probe_path = variants[vix][1] if variants else real_idx[probe_id]

        run = st.button("Run authentication", type="primary", width="stretch")

    with right:
        st.markdown('<div class="np-stage">Stage 0 - captured image</div>',
                    unsafe_allow_html=True)
        pcol, ecol = st.columns(2)
        with pcol:
            st.image(pl.fingerprint_image(probe_path, 185), width=185)
            st.caption(f"Presented: `{probe_id}`")
        with ecol:
            st.image(pl.fingerprint_image(real_idx[claimed], 185), width=185)
            st.caption(f"Claimed record {claimed_ix}: `{claimed}`")
        if genuine:
            st.caption("A different capture of the same finger already enrolled.")
        else:
            st.caption("This finger has never been enrolled; it is claiming someone else's record.")

    if run:
        st.divider()

        emb, probe_code, t_cnn, t_quant = encode_probe(probe_path, N_CALIB, int(seed))

        s1, s2 = st.columns(2)
        with s1:
            st.markdown('<div class="np-stage">Stage 1 - CNN feature extraction</div>',
                        unsafe_allow_html=True)
            st.caption(f"16-dimensional embedding computed on the client in {t_cnn:.1f} ms. "
                       "Neither the image nor this embedding leaves the device.")
            edf = pd.DataFrame({"dimension": np.arange(1, pl.EMB_DIM + 1),
                                "value": emb.astype(float)})
            st.altair_chart(
                alt.Chart(edf).mark_bar(size=14).encode(
                    x=alt.X("dimension:O", title="embedding dimension",
                            axis=alt.Axis(labelAngle=0)),
                    y=alt.Y("value:Q", title="value"),
                    color=alt.condition(alt.datum.value > 0,
                                        alt.value(ACCENT), alt.value("#c26a4a")),
                    tooltip=["dimension", alt.Tooltip("value:Q", format=".2f")],
                ).properties(height=215),
                width="stretch")

        with s2:
            st.markdown('<div class="np-stage">Stage 2 - Super-Bit binarisation</div>',
                        unsafe_allow_html=True)
            st.caption(f"128-bit code from one matrix multiply and a threshold, "
                       f"{t_quant:.0f} microseconds. Bits set: {int(probe_code.sum())} of 128.")
            st.image(pl.bit_grid_image(probe_code, rows=8, cell=20), width=340)
            st.markdown(f'<span class="np-mono">{pl.bits_to_str(probe_code)}</span>',
                        unsafe_allow_html=True)

        # ---------------- Stage 3 ----------------
        st.divider()
        st.markdown('<div class="np-stage">Stage 3 - Fuzzy-Labelled PSI match</div>',
                    unsafe_allow_html=True)

        dists = pl.hamming(db_codes, probe_code).astype(int)
        probs = np.array([pl.accept_probability(int(h), weight, t_thr, T_masks) for h in dists])

        real_ms = None
        if engine.startswith("Real"):
            try:
                with st.spinner("Executing masked-OPRF, garbled circuits, VOLE and Shamir..."):
                    res = pl.run_real_flpsi(probe_code, db_codes, weight, t_thr, T_masks)
                real_ms = res.duration_ms
                passes = np.array([i in set(res.matched_indices)
                                   for i in range(len(db_codes))])
            except Exception as exc:
                st.warning(f"Real protocol unavailable, using the closed-form model. ({exc})")
                passes = probs > 0.5
        else:
            passes = probs > 0.5

        accepted = bool(passes[claimed_ix])
        h_claim = int(dists[claimed_ix])
        nearest = int(np.argmin(dists))
        truth_ok = accepted if genuine else (not accepted)

        vcol, mcol = st.columns([1.45, 1])
        with vcol:
            if accepted:
                st.markdown(
                    f'<div class="np-verdict-ok"><div class="np-verdict-h">Access granted</div>'
                    f'Verified against record <b>{claimed_ix}</b> &nbsp;·&nbsp; label released: '
                    f'<b>{claimed}</b><br>Hamming distance <b>{h_claim}</b> / {pl.D_BITS}</div>',
                    unsafe_allow_html=True)
            else:
                st.markdown(
                    f'<div class="np-verdict-bad"><div class="np-verdict-h">Access denied</div>'
                    f'The presented finger did not clear the threshold against claimed record '
                    f'<b>{claimed_ix}</b>. Hamming distance <b>{h_claim}</b> / {pl.D_BITS}.</div>',
                    unsafe_allow_html=True)
            if truth_ok:
                st.success("Outcome matches the ground truth for this scenario.")
            else:
                st.error("Outcome does not match the ground truth for this scenario.")

        with mcol:
            st.metric("Hamming to claimed record", f"{h_claim} / {pl.D_BITS}")
            st.metric("Closest record in database", f"{int(dists[nearest])} / {pl.D_BITS}",
                      help=f"Record {nearest} · {enrolled[nearest]}")
            if real_ms is not None:
                st.metric("Protocol execution", f"{real_ms:.0f} ms",
                          help="Full setup and online phase of the real flash-psi protocol.")

        st.markdown("##### Hamming distance to every enrolled record")
        ddf = pd.DataFrame({
            "record": np.arange(len(dists)),
            "identity": enrolled,
            "hamming": dists,
            "role": np.where(np.arange(len(dists)) == claimed_ix,
                             "claimed record", "other enrolled records"),
        })
        st.altair_chart(
            alt.Chart(ddf).mark_bar().encode(
                x=alt.X("record:O", title="enrolled record",
                        axis=alt.Axis(labelAngle=0, labelOverlap=True)),
                y=alt.Y("hamming:Q", title="Hamming distance (of 128)"),
                color=alt.Color("role:N",
                                scale=alt.Scale(domain=["claimed record",
                                                        "other enrolled records"],
                                                range=[OK, "#c8d2dd"]),
                                legend=alt.Legend(title=None, orient="top")),
                tooltip=["record", "identity", "hamming", "role"],
            ).properties(height=235),
            width="stretch")
        st.caption("Two captures of the same finger differ in roughly 8 of 128 bits; unrelated "
                   "fingers differ in roughly 64 of 128. The protocol tests exactly that "
                   "separation, without the server seeing the code.")

        st.markdown("##### Code comparison against the claimed record")
        diff = (probe_code != db_codes[claimed_ix])
        g1, g2, g3 = st.columns(3)
        with g1:
            st.caption("Presented finger")
            st.image(pl.bit_grid_image(probe_code, rows=8, cell=18), width=300)
        with g2:
            st.caption(f"Claimed record {claimed_ix} - `{claimed}`")
            st.image(pl.bit_grid_image(db_codes[claimed_ix], rows=8, cell=18), width=300)
        with g3:
            st.caption(f"Differing positions: {int(diff.sum())} of {pl.D_BITS}")
            st.image(pl.bit_grid_image(probe_code, rows=8, cell=18, diff_mask=diff), width=300)

        with st.expander("Per-record protocol detail"):
            st.dataframe(
                pd.DataFrame({
                    "Record": np.arange(len(enrolled)),
                    "Identity": enrolled,
                    "Hamming": dists,
                    "Sub-sample collision q(H)": [
                        f"{pl.q_clean(int(h), weight):.3e}" for h in dists],
                    "Accept probability": [f"{p:.4f}" for p in probs],
                }).sort_values("Hamming"),
                hide_index=True, width="stretch", height=300)


# --------------------------------------------------------------------------- #
# Tab 3 - explanation
# --------------------------------------------------------------------------- #
with tab_about:
    st.markdown("#### The three stages")
    a, b, c = st.columns(3)
    with a:
        st.markdown('<div class="np-stage">Stage 1 - feature extraction</div>',
                    unsafe_allow_html=True)
        st.markdown(
            "A five-block convolutional network maps a 224 x 224 grayscale fingerprint to a "
            "**16-dimensional embedding**. It is trained in a Siamese configuration so that "
            "captures of the same finger land close together and different fingers land far "
            "apart. The network runs on the client; no image or embedding reaches the server.")
    with b:
        st.markdown('<div class="np-stage">Stage 2 - binarisation</div>',
                    unsafe_allow_html=True)
        st.markdown(
            "The embedding is centred, projected through a public **Super-Bit** matrix of "
            "eight orthonormalised 16 x 16 blocks, and thresholded at per-bit medians, giving "
            "a balanced **128-bit code**. Expected Hamming distance follows "
            "arccos(cosine similarity) / pi, so angular similarity becomes bit agreement.")
    with c:
        st.markdown('<div class="np-stage">Stage 3 - private matching</div>',
                    unsafe_allow_html=True)
        st.markdown(
            "Fuzzy-Labelled PSI draws **T** random masks of weight **w**. Two codes sub-match "
            "on a mask when they agree on every selected position; the record is released once "
            "**t** sub-samples collide. Masks are evaluated through a masked OPRF, so the "
            "server never sees the query code in the clear.")

    st.divider()
    st.markdown("#### Measured results behind this demonstration")
    r1, r2 = st.columns(2)
    with r1:
        st.markdown("**Accuracy on held-out identities**")
        st.dataframe(pd.DataFrame({
            "Representation": ["Floating-point embedding (ceiling)",
                               "Naive sign, 16 bits", "ITQ, 16 bits", "Super-Bit, 128 bits"],
            "Equal error rate": ["0.33%", "10.12%", "5.55%", "1.88%"],
        }), hide_index=True, width="stretch")
        st.caption("1,200 genuine and 240,000 impostor pairs (200 sampled impostors per probe) "
                   "on SOCOFing Altered-Easy. This is an intentionally easy, single-dataset "
                   "protocol; cross-sensor evaluation remains future work.")
    with r2:
        st.markdown("**Efficiency against the homomorphic-encryption baseline**")
        st.dataframe(pd.DataFrame({
            "Property": ["Template per user", "One-time key", "Latency, single server",
                         "Per-record match cost"],
            "Blind-Touch (HE)": ["~1.67 KB", "117 MB", "1,334 ms", "capped by slot packing"],
            "Neural-PSI": ["16 B", "none", "~824 ms", "~0.16 ms"],
        }), hide_index=True, width="stretch")
        st.caption("Blind-Touch figures from Choi, Woo & Kim, AAAI 2024. The real protocol was "
                   "validated against the closed-form model on 100 genuine and 9,900 impostor "
                   "decisions, agreeing to two significant figures.")
