"""
KLASTERISASI CACAT PRODUK MANUFAKTUR — Panel Rekayasa
Aplikasi Streamlit - Deployment Project (Pertemuan ke-12)
Varian tema: "Blueprint Engineering" — navigasi dropdown, palet biru-oranye teknikal.

Fitur:
1. Eksplorasi dataset cacat produk (defects_data.csv)
2. Rekayasa fitur per produk & klasterisasi K-Means
3. Diagnostik model (elbow method & silhouette score)
4. Rekomendasi aksi otomatis berdasarkan profil tiap klaster
"""

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA

# --------------------------------------------------------------------------------
# IDENTITAS PENYUSUN
# --------------------------------------------------------------------------------
NAMA = "Muhammad Ghany Putra S"
NIM = "E12.2024.02002"

# --------------------------------------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------------------------------------
st.set_page_config(
    page_title="Panel Rekayasa — Klasterisasi Cacat Produk",
    page_icon="📐",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --------------------------------------------------------------------------------
# THEME / CUSTOM CSS  -- "Blueprint Engineering" technical look
# --------------------------------------------------------------------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=IBM+Plex+Sans:wght@400;500;600;700&display=swap');

:root{
    --bg:        #F4F7FB;
    --panel:     #FFFFFF;
    --panel-2:   #E9EFF7;
    --line:      #C7D6E8;
    --text:      #14213D;
    --muted:     #5C7091;
    --blue:      #0B3D91;
    --orange:    #FF6B35;
    --cyan:      #0090A8;
    --amber:     #E8A628;
}

html, body, [class*="css"]  { font-family: 'IBM Plex Sans', 'Segoe UI', sans-serif; }
.stApp {
    background-color: var(--bg);
    background-image: linear-gradient(var(--line) 1px, transparent 1px), linear-gradient(90deg, var(--line) 1px, transparent 1px);
    background-size: 34px 34px;
    background-attachment: fixed;
    color: var(--text);
}

section[data-testid="stSidebar"] {
    background-color: #FFFFFF;
    border-right: 2px solid var(--blue);
}
section[data-testid="stSidebar"] * { color: var(--text) !important; }

h1, h2, h3 { font-family: 'Space Mono', monospace; letter-spacing: -0.01em; }
h1 { color: var(--blue); font-weight: 700; }
h2, h3 { color: var(--orange); font-weight: 700; }

/* Ruler divider - blueprint signature element */
.ruler-divider { display: flex; align-items: center; gap: 10px; margin: 0.5rem 0 1.3rem 0; }
.ruler-divider .track { flex:1; height: 6px; position: relative;
    background: repeating-linear-gradient(90deg, var(--blue) 0 2px, transparent 2px 12px); opacity: 0.5; }
.ruler-divider .label { font-family:'Space Mono', monospace; font-size:0.7rem; color: var(--blue);
    border: 1px solid var(--blue); padding: 2px 10px; border-radius: 2px; background: var(--panel-2); }

/* Metric / spec-sheet card with corner brackets */
.spec-card { position: relative; background: var(--panel); border: 1px solid var(--line);
    border-radius: 2px; padding: 16px 18px 14px 18px; margin-bottom: 10px; }
.spec-card:before, .spec-card:after { content: ""; position: absolute; width: 10px; height: 10px;
    border-top: 2px solid var(--orange); border-left: 2px solid var(--orange); top: 4px; left: 4px; }
.spec-card:after { top: auto; left: auto; bottom: 4px; right: 4px;
    border-top: none; border-left: none; border-bottom: 2px solid var(--orange); border-right: 2px solid var(--orange); }
.spec-card .label { color: var(--muted); font-size: 0.72rem; text-transform: uppercase; letter-spacing: 0.08em; font-family:'Space Mono',monospace;}
.spec-card .value { font-family: 'Space Mono', monospace; font-size: 1.65rem; font-weight: 700; color: var(--blue); }
.spec-card .sub   { color: var(--cyan); font-size: 0.8rem; }

/* Tags */
.tag-critical { background: #FF6B35; color: #FFFFFF; border:1px solid #C4491F; padding:3px 14px; border-radius: 2px; font-size:0.76rem; font-weight:700; font-family:'Space Mono',monospace;}
.tag-moderate { background: #E8A628; color: #1A1200; border:1px solid #B37E14; padding:3px 14px; border-radius: 2px; font-size:0.76rem; font-weight:700; font-family:'Space Mono',monospace;}
.tag-good     { background: #00A8C4; color: #FFFFFF; border:1px solid #007286; padding:3px 14px; border-radius: 2px; font-size:0.76rem; font-weight:700; font-family:'Space Mono',monospace;}

/* Insight panel */
.note-card { background: var(--panel); border: 1px solid var(--line); border-left: 6px solid var(--blue);
    border-radius: 2px; padding: 18px 20px; margin-bottom: 14px; }
.note-card h4 { margin-top:0; font-family:'Space Mono', monospace; font-size:1.05rem;}

.identity-chip { background: var(--panel-2); border: 1px solid var(--blue); border-radius: 2px;
    padding: 8px 12px; margin-bottom: 10px; font-size: 0.82rem; }

[data-testid="stDataFrame"] { border: 1px solid var(--line); border-radius: 2px; }

.stButton>button, .stDownloadButton>button {
    background: var(--blue); color: #FFFFFF; border: none; font-weight: 600; border-radius: 2px;
    font-family: 'Space Mono', monospace;
}

div[data-baseweb="select"] > div { border-color: var(--blue) !important; border-radius: 2px !important; }
</style>
""", unsafe_allow_html=True)


def ruler_divider(label_text="§"):
    st.markdown(
        f'<div class="ruler-divider"><div class="track"></div><div class="label">{label_text}</div>'
        f'<div class="track"></div></div>', unsafe_allow_html=True
    )


# --------------------------------------------------------------------------------
# DATA LOADING & FEATURE ENGINEERING
# --------------------------------------------------------------------------------
@st.cache_data
def load_raw_data(path="defects_data.csv"):
    df = pd.read_csv(path)
    df["defect_date"] = pd.to_datetime(df["defect_date"])
    return df


@st.cache_data
def build_product_features(df: pd.DataFrame) -> pd.DataFrame:
    agg = df.groupby("product_id").agg(
        total_defects=("defect_id", "count"),
        avg_repair_cost=("repair_cost", "mean"),
        total_repair_cost=("repair_cost", "sum"),
        max_repair_cost=("repair_cost", "max"),
    ).reset_index()

    sev = pd.crosstab(df["product_id"], df["severity"], normalize="index").add_prefix("pct_sev_")
    typ = pd.crosstab(df["product_id"], df["defect_type"], normalize="index").add_prefix("pct_type_")
    loc = pd.crosstab(df["product_id"], df["defect_location"], normalize="index").add_prefix("pct_loc_")
    insp = pd.crosstab(df["product_id"], df["inspection_method"], normalize="index").add_prefix("pct_insp_")

    features = (
        agg.merge(sev, on="product_id")
           .merge(typ, on="product_id")
           .merge(loc, on="product_id")
           .merge(insp, on="product_id")
    )
    return features


FEATURE_COLS = [
    "total_defects", "avg_repair_cost", "total_repair_cost", "max_repair_cost",
    "pct_sev_Critical", "pct_sev_Moderate", "pct_sev_Minor",
    "pct_type_Structural", "pct_type_Functional", "pct_type_Cosmetic",
    "pct_loc_Component", "pct_loc_Internal", "pct_loc_Surface",
    "pct_insp_Automated Testing", "pct_insp_Manual Testing", "pct_insp_Visual Inspection",
]

# Palet warna klaster — dibuat pekat/kontras agar tiap klaster mudah dibedakan
CLUSTER_COLORS = ["#2563EB", "#FF6B35", "#059669", "#DB2777", "#7C3AED", "#0891B2", "#CA8A04", "#DC2626"]


@st.cache_data
def scan_k(features: pd.DataFrame, k_min=2, k_max=8):
    X = StandardScaler().fit_transform(features[FEATURE_COLS].values)
    rows = []
    for k in range(k_min, k_max + 1):
        km = KMeans(n_clusters=k, random_state=42, n_init=10).fit(X)
        sil = silhouette_score(X, km.labels_)
        rows.append({"k": k, "inertia": km.inertia_, "silhouette": sil})
    return pd.DataFrame(rows)


@st.cache_data
def run_kmeans(features: pd.DataFrame, k: int):
    X = StandardScaler().fit_transform(features[FEATURE_COLS].values)
    km = KMeans(n_clusters=k, random_state=42, n_init=10).fit(X)
    labels = km.labels_
    sil = silhouette_score(X, labels)

    pca = PCA(n_components=2, random_state=42)
    coords = pca.fit_transform(X)

    out = features.copy()
    out["cluster"] = labels
    out["pca_1"] = coords[:, 0]
    out["pca_2"] = coords[:, 1]
    return out, sil, pca.explained_variance_ratio_


def zlabel(z):
    if z >= 0.35:
        return "Tinggi"
    if z <= -0.35:
        return "Rendah"
    return "Sedang"


def generate_cluster_narrative(clustered: pd.DataFrame):
    overall_mean = clustered[FEATURE_COLS].mean()
    overall_std = clustered[FEATURE_COLS].std().replace(0, 1)

    narratives = {}
    for c in sorted(clustered["cluster"].unique()):
        sub = clustered[clustered["cluster"] == c]
        n = len(sub)
        cmean = sub[FEATURE_COLS].mean()
        z = (cmean - overall_mean) / overall_std

        freq_lvl = zlabel(z["total_defects"])
        cost_lvl = zlabel(z["total_repair_cost"])

        dom_sev = max(["Critical", "Moderate", "Minor"], key=lambda s: cmean[f"pct_sev_{s}"])
        dom_type = max(["Structural", "Functional", "Cosmetic"], key=lambda s: cmean[f"pct_type_{s}"])
        dom_loc = max(["Component", "Internal", "Surface"], key=lambda s: cmean[f"pct_loc_{s}"])
        dom_insp = max(["Automated Testing", "Manual Testing", "Visual Inspection"],
                        key=lambda s: cmean[f"pct_insp_{s}"])

        risk = "Tinggi" if (freq_lvl == "Tinggi" or cost_lvl == "Tinggi" or dom_sev == "Critical") else \
               ("Rendah" if (freq_lvl == "Rendah" and cost_lvl == "Rendah") else "Sedang")

        title = f"Frekuensi {freq_lvl} · Biaya {cost_lvl} · dominan cacat {dom_type}"

        rekomendasi = []
        if risk == "Tinggi":
            rekomendasi.append("Tetapkan sebagai prioritas audit mutu & investigasi akar penyebab (root-cause).")
        if dom_insp == "Manual Testing" and dom_sev == "Critical":
            rekomendasi.append("Migrasikan sebagian pemeriksaan ke Automated Testing untuk deteksi dini cacat kritis.")
        if dom_type == "Structural":
            rekomendasi.append("Evaluasi ulang spesifikasi desain dan pemilihan material terkait cacat struktural.")
        if dom_type == "Functional":
            rekomendasi.append("Perketat protokol pengujian fungsional pada tahap akhir jalur produksi.")
        if dom_type == "Cosmetic":
            rekomendasi.append("Tinjau prosedur finishing dan penanganan produk untuk cacat kosmetik.")
        if cost_lvl == "Tinggi":
            rekomendasi.append("Bandingkan opsi perbaikan berulang vs penggantian komponen dari sisi biaya.")
        if not rekomendasi:
            rekomendasi.append("Tetapkan sebagai baseline/benchmark proses produksi untuk klaster lain.")

        narratives[c] = {
            "n": n, "title": title, "risk": risk, "freq_lvl": freq_lvl, "cost_lvl": cost_lvl,
            "dom_sev": dom_sev, "dom_type": dom_type, "dom_loc": dom_loc, "dom_insp": dom_insp,
            "avg_defects": cmean["total_defects"], "avg_cost": cmean["avg_repair_cost"],
            "total_cost": cmean["total_repair_cost"], "rekomendasi": rekomendasi,
        }
    return narratives


def risk_tag(risk):
    cls = {"Tinggi": "tag-critical", "Sedang": "tag-moderate", "Rendah": "tag-good"}[risk]
    return f'<span class="{cls}">Risiko {risk}</span>'


# --------------------------------------------------------------------------------
# SIDEBAR — navigasi dropdown + panel kontrol
# --------------------------------------------------------------------------------
st.sidebar.markdown("## 📐 PANEL REKAYASA")
st.sidebar.caption("Klasterisasi Cacat Produk Manufaktur")
st.sidebar.markdown(
    f'<div class="identity-chip"><b style="color:var(--blue);">{NAMA}</b><br>'
    f'<span style="color:var(--muted);">NIM {NIM}</span></div>',
    unsafe_allow_html=True,
)
st.sidebar.divider()

page = st.sidebar.radio(
    "Navigasi",
    ["📟 Dashboard", "🗂️ Jelajah Data", "⚙️ Proses Klasterisasi", "🧭 Rekomendasi Aksi", "📄 Dokumentasi"],
    label_visibility="collapsed",
)

df = load_raw_data()
features = build_product_features(df)

st.sidebar.divider()
k_choice = st.sidebar.slider("Jumlah Klaster (K)", min_value=2, max_value=8, value=3,
                              help="Jumlah klaster K-Means pada level produk.")
st.sidebar.caption(f"Dataset: {len(df):,} baris cacat • {features.shape[0]} produk")

clustered, sil_score, evr = run_kmeans(features, k_choice)
narratives = generate_cluster_narrative(clustered)

# --------------------------------------------------------------------------------
# IDENTITY STRIP — tampil di setiap halaman
# --------------------------------------------------------------------------------
st.markdown(
    f'<div style="text-align:right; color:var(--muted); font-size:0.8rem; margin-bottom:2px;">'
    f'<b style="color:var(--blue);">{NAMA}</b> &nbsp;|&nbsp; NIM {NIM}</div>',
    unsafe_allow_html=True,
)

# --------------------------------------------------------------------------------
# PAGE: DASHBOARD
# --------------------------------------------------------------------------------
if page == "📟 Dashboard":
    st.markdown("# 📐 Panel Rekayasa — Cacat Produk Manufaktur")
    st.caption("Klasterisasi K-Means · Deployment Streamlit · Pertemuan ke-12")
    ruler_divider("defects_data.csv")

    st.write(
        "Panel ini memetakan **produk manufaktur** ke dalam beberapa klaster berdasarkan spesifikasi "
        "cacat yang tercatat di lini produksi: frekuensi kemunculan, biaya perbaikan, tingkat "
        "keparahan (severity), jenis cacat, titik lokasi cacat, dan metode inspeksi yang dipakai."
    )

    c1, c2, c3, c4 = st.columns(4)
    cards = [
        ("Baris Data Cacat", f"{len(df):,}", "catatan mentah"),
        ("Produk Teridentifikasi", f"{features.shape[0]}", "unit klasterisasi"),
        ("Total Biaya Perbaikan", f"${df['repair_cost'].sum():,.0f}", "akumulasi seluruh cacat"),
        ("Rerata Biaya / Kejadian", f"${df['repair_cost'].mean():,.2f}", "per catatan cacat"),
    ]
    for col, (label, value, sub) in zip([c1, c2, c3, c4], cards):
        col.markdown(
            f'<div class="spec-card"><div class="label">{label}</div>'
            f'<div class="value">{value}</div><div class="sub">{sub}</div></div>',
            unsafe_allow_html=True,
        )

    ruler_divider("spesimen")
    st.markdown("### Spesimen Data")
    st.dataframe(df.head(10), use_container_width=True, hide_index=True)

    st.markdown("### Diagram Proses")
    st.markdown(
        "1. **Rekayasa fitur** — 1.000 catatan cacat dipadatkan menjadi 100 baris (satu entri per produk).\n"
        "2. **Standardisasi** nilai numerik & proporsi kategori (severity, jenis, lokasi, metode inspeksi).\n"
        "3. **Klasterisasi K-Means**, tervalidasi lewat *elbow method* dan *silhouette score*.\n"
        "4. **Rekomendasi otomatis** — setiap klaster diberi label risiko & langkah tindak lanjut untuk tim QC."
    )

# --------------------------------------------------------------------------------
# PAGE: JELAJAH DATA
# --------------------------------------------------------------------------------
elif page == "🗂️ Jelajah Data":
    st.markdown("# 🗂️ Jelajah Data")
    ruler_divider("eksplorasi")

    tab1, tab2 = st.tabs(["Data per Kejadian Cacat", "Data per Produk (Teragregasi)"])

    with tab1:
        colf1, colf2, colf3 = st.columns(3)
        sev_f = colf1.multiselect("Severity", sorted(df["severity"].unique()), default=list(df["severity"].unique()))
        typ_f = colf2.multiselect("Jenis Cacat", sorted(df["defect_type"].unique()), default=list(df["defect_type"].unique()))
        insp_f = colf3.multiselect("Metode Inspeksi", sorted(df["inspection_method"].unique()), default=list(df["inspection_method"].unique()))
        filt = df[df["severity"].isin(sev_f) & df["defect_type"].isin(typ_f) & df["inspection_method"].isin(insp_f)]
        st.dataframe(filt, use_container_width=True, hide_index=True)
        st.caption(f"{len(filt):,} dari {len(df):,} catatan ditampilkan.")

        cA, cB = st.columns(2)
        with cA:
            fig = px.histogram(filt, x="repair_cost", nbins=30, color="severity",
                                title="Sebaran Biaya Perbaikan per Severity",
                                color_discrete_map={"Critical": "#FF6B35", "Moderate": "#E8A628", "Minor": "#0090A8"})
            fig.update_layout(template="plotly_white", plot_bgcolor="#FFFFFF", paper_bgcolor="#FFFFFF", font_color="#14213D")
            st.plotly_chart(fig, use_container_width=True)
        with cB:
            ct = filt.groupby(["defect_type", "defect_location"]).size().reset_index(name="jumlah")
            fig2 = px.bar(ct, x="defect_type", y="jumlah", color="defect_location", barmode="group",
                          title="Jumlah Cacat: Jenis vs Lokasi",
                          color_discrete_sequence=["#0B3D91", "#E8A628", "#0090A8"])
            fig2.update_layout(template="plotly_white", plot_bgcolor="#FFFFFF", paper_bgcolor="#FFFFFF", font_color="#14213D")
            st.plotly_chart(fig2, use_container_width=True)

        df_month = filt.copy()
        df_month["bulan"] = df_month["defect_date"].dt.to_period("M").astype(str)
        trend = df_month.groupby("bulan").agg(jumlah=("defect_id", "count"), biaya=("repair_cost", "sum")).reset_index()
        fig3 = go.Figure()
        fig3.add_bar(x=trend["bulan"], y=trend["jumlah"], name="Jumlah Cacat", marker_color="#0B3D91")
        fig3.add_scatter(x=trend["bulan"], y=trend["biaya"] / trend["biaya"].max() * trend["jumlah"].max(),
                          name="Biaya (skala relatif)", mode="lines+markers", line=dict(color="#FF6B35"))
        fig3.update_layout(title="Tren Bulanan: Jumlah Cacat & Biaya Perbaikan", template="plotly_white",
                            plot_bgcolor="#FFFFFF", paper_bgcolor="#FFFFFF", font_color="#14213D")
        st.plotly_chart(fig3, use_container_width=True)

    with tab2:
        st.write(
            "Setiap baris pada tabel berikut mewakili **satu produk**, dengan fitur hasil agregasi "
            "yang menjadi masukan proses klasterisasi."
        )
        st.dataframe(features.round(3), use_container_width=True, hide_index=True)
        st.download_button("⬇️ Unduh Data Teragregasi (CSV)", features.to_csv(index=False),
                            file_name="product_features_aggregated.csv", mime="text/csv")

# --------------------------------------------------------------------------------
# PAGE: PROSES KLASTERISASI
# --------------------------------------------------------------------------------
elif page == "⚙️ Proses Klasterisasi":
    st.markdown("# ⚙️ Proses Klasterisasi (K-Means)")
    ruler_divider("model")

    st.markdown(
        "**Spesifikasi model:** 16 fitur (frekuensi & biaya cacat, serta proporsi severity, jenis "
        "cacat, lokasi cacat, dan metode inspeksi) distandardisasi (Z-score), lalu dikelompokkan "
        "dengan **K-Means**. Jumlah klaster divalidasi memakai *elbow method* (inertia) dan "
        "*silhouette score*."
    )

    scan = scan_k(features)
    c1, c2 = st.columns(2)
    with c1:
        fig = px.line(scan, x="k", y="inertia", markers=True, title="Elbow Method — Inertia vs K")
        fig.update_traces(line_color="#0B3D91", marker=dict(size=9, color="#FF6B35"))
        fig.update_layout(template="plotly_white", plot_bgcolor="#FFFFFF", paper_bgcolor="#FFFFFF", font_color="#14213D")
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        fig = px.line(scan, x="k", y="silhouette", markers=True, title="Silhouette Score vs K")
        fig.update_traces(line_color="#0090A8", marker=dict(size=9, color="#FF6B35"))
        fig.update_layout(template="plotly_white", plot_bgcolor="#FFFFFF", paper_bgcolor="#FFFFFF", font_color="#14213D")
        st.plotly_chart(fig, use_container_width=True)

    st.info(
        f"Konfigurasi aktif: **K = {k_choice}** · Silhouette score: **{sil_score:.3f}** "
        "(atur ulang lewat slider di sidebar)."
    )

    ruler_divider("visual")
    cL, cR = st.columns([1.3, 1])
    with cL:
        fig = px.scatter(
            clustered, x="pca_1", y="pca_2", color=clustered["cluster"].astype(str),
            hover_data=["product_id", "total_defects", "avg_repair_cost"],
            title=f"Peta Klaster (PCA 2D — variansi terjelaskan {sum(evr)*100:.1f}%)",
            color_discrete_sequence=CLUSTER_COLORS,
        )
        fig.update_traces(marker=dict(size=11, line=dict(width=1, color="#F4F7FB")))
        fig.update_layout(template="plotly_white", plot_bgcolor="#FFFFFF", paper_bgcolor="#FFFFFF",
                           font_color="#14213D", legend_title="Klaster")
        st.plotly_chart(fig, use_container_width=True)
    with cR:
        sizes = clustered["cluster"].value_counts().sort_index()
        fig = px.bar(x=[f"Klaster {i}" for i in sizes.index], y=sizes.values,
                     title="Jumlah Produk per Klaster", color=[f"Klaster {i}" for i in sizes.index],
                     color_discrete_sequence=CLUSTER_COLORS)
        fig.update_layout(template="plotly_white", plot_bgcolor="#FFFFFF", paper_bgcolor="#FFFFFF", font_color="#14213D",
                           showlegend=False, xaxis_title="", yaxis_title="Jumlah Produk")
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("### Spesifikasi Rata-rata Fitur per Klaster")
    profile = clustered.groupby("cluster")[FEATURE_COLS].mean().round(3)
    profile.index = [f"Klaster {i}" for i in profile.index]
    st.dataframe(profile.T, use_container_width=True)

# --------------------------------------------------------------------------------
# PAGE: REKOMENDASI AKSI
# --------------------------------------------------------------------------------
elif page == "🧭 Rekomendasi Aksi":
    st.markdown("# 🧭 Rekomendasi Aksi")
    ruler_divider("insight")
    st.write(
        f"Model dengan **K = {k_choice}** (silhouette **{sil_score:.3f}**) membagi produk ke dalam "
        f"{k_choice} klaster. Berikut spesifikasi tiap klaster beserta rekomendasi tindakan untuk "
        "tim Quality Control:"
    )

    for c in sorted(narratives.keys()):
        nar = narratives[c]
        ccolor = CLUSTER_COLORS[c % len(CLUSTER_COLORS)]
        st.markdown(
            f"""<div class="note-card" style="border-left-color:{ccolor};">
            <h4 style="color:{ccolor};">Klaster {c} — {nar['n']} produk &nbsp; {risk_tag(nar['risk'])}</h4>
            <p><b>{nar['title']}</b></p>
            <p style="color:#5C7091; font-size:0.9rem;">
            Rerata {nar['avg_defects']:.1f} cacat/produk &middot;
            biaya rerata ${nar['avg_cost']:.0f}/kejadian &middot;
            total biaya klaster ${nar['total_cost']:.0f} &middot;
            severity dominan <b>{nar['dom_sev']}</b> &middot;
            jenis cacat dominan <b>{nar['dom_type']}</b> &middot;
            lokasi dominan <b>{nar['dom_loc']}</b> &middot;
            metode inspeksi dominan <b>{nar['dom_insp']}</b>
            </p>
            </div>""",
            unsafe_allow_html=True,
        )
        st.markdown("**Rekomendasi tindakan:**")
        for r in nar["rekomendasi"]:
            st.markdown(f"- {r}")
        st.write("")

    ruler_divider("ringkasan")
    st.markdown("### Ringkasan untuk Manajemen")
    high_risk = [c for c, n in narratives.items() if n["risk"] == "Tinggi"]
    total_cost_all = clustered["total_repair_cost"].sum()
    hr_cost = clustered[clustered["cluster"].isin(high_risk)]["total_repair_cost"].sum() if high_risk else 0
    pct_hr_cost = (hr_cost / total_cost_all * 100) if total_cost_all else 0

    st.markdown(
        f"- **{len(high_risk)} dari {k_choice} klaster** berstatus **risiko tinggi**, mewakili sekitar "
        f"**{pct_hr_cost:.1f}%** dari total biaya perbaikan seluruh produk.\n"
        "- Produk dengan porsi *Manual Testing* besar dan severity *Critical* tinggi sebaiknya jadi "
        "kandidat pertama migrasi ke *Automated Testing*.\n"
        "- Cacat **Structural** cenderung mahal dan berulang — telusuri hingga ke tahap desain atau "
        "pemilihan material, bukan sekadar diperbaiki di ujung jalur produksi.\n"
        "- Klaster berfrekuensi & berbiaya rendah dapat dijadikan **baseline** untuk direplikasi ke "
        "lini produk lain."
    )

# --------------------------------------------------------------------------------
# PAGE: DOKUMENTASI
# --------------------------------------------------------------------------------
else:
    st.markdown("# 📄 Dokumentasi")
    ruler_divider("tentang")
    st.markdown(
        f"""
        **Panel Rekayasa Cacat Produk Manufaktur** disusun sebagai proyek deployment untuk tugas
        mata kuliah (Pertemuan ke-12), mengikuti alur tutorial deployment Streamlit:

        - Kode disusun dalam satu berkas aplikasi beserta `requirements.txt` yang hanya memuat
          pustaka yang benar-benar dipakai (`streamlit`, `pandas`, `numpy`, `scikit-learn`, `plotly`)
          — tanpa pustaka yang tidak diperlukan seperti `pickle`, agar proses build di Streamlit
          Cloud tidak gagal.
        - Repository dihubungkan ke **Streamlit Community Cloud** lewat akun GitHub, memakai
          branch `main`.
        - Setelah proses deploy, server di-*reboot* bila status belum konsisten, hingga akhirnya
          menghasilkan URL publik yang dapat diuji langsung.

        **Spesifikasi metode:** K-Means Clustering pada level produk (bukan level kejadian cacat
        individual), dengan 16 fitur hasil rekayasa dari kolom `defect_type`, `defect_location`,
        `severity`, `inspection_method`, dan `repair_cost`. Jumlah klaster divalidasi memakai
        *elbow method* dan *silhouette score* pada halaman **Proses Klasterisasi**.

        **Sumber data:** `defects_data.csv` — 1.000 catatan cacat dari 100 produk manufaktur.

        ---
        Disusun oleh **{NAMA}** — NIM {NIM}
        """
    )
    st.caption("Dibuat dengan Streamlit • scikit-learn • Plotly")
