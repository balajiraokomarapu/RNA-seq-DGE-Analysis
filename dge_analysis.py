"""
Differential Gene Expression Analysis
Effect of Ethanol Exposure on the Adult Mouse Brain Transcriptome

Dataset  : GSE222445 (NCBI Gene Expression Omnibus)
Organism : Mus musculus (adult mouse brain)
Author   : Komarapu Balaji Rao | 23BT3EP28
Course   : Bioinformatics Laboratory, IIT Kharagpur
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from adjustText import adjust_text

import GEOparse
from pydeseq2.dds import DeseqDataSet
from pydeseq2.ds import DeseqStats
from sklearn.decomposition import PCA
from scipy.spatial.distance import pdist, squareform
import gseapy as gp


# =============================================================
# 1. Load Gene Counts
# =============================================================
def load_counts(filepath: str) -> pd.DataFrame:
    """Load and transpose raw count matrix from GEO CSV."""
    print("Loading count matrix...")
    raw_df = pd.read_csv(filepath, compression="gzip", index_col=0)
    # PyDESeq2 expects samples as rows, genes as columns
    return raw_df.T


# =============================================================
# 2. Fetch Sample Metadata from GEO
# =============================================================
def fetch_metadata(geo_accession: str) -> pd.DataFrame:
    """Download and parse sample metadata from NCBI GEO."""
    print(f"Fetching metadata for {geo_accession}...")
    gse = GEOparse.get_GEO(geo_accession, destdir="./")

    meta_data = []
    for gsm_name, gsm in gse.gsms.items():
        title = gsm.metadata.get("title", [""])[0]
        chars = gsm.metadata.get("characteristics_ch1", [])
        treatment = "Unknown"
        for char in chars:
            if "treatment:" in char:
                treatment = char.split(":")[1].strip()
        meta_data.append({"rownames": title, "condition": treatment})

    sample_info = pd.DataFrame(meta_data).set_index("rownames")
    return sample_info


# =============================================================
# 3. Preprocess & Run DESeq2
# =============================================================
def run_deseq2(counts_df: pd.DataFrame, sample_info: pd.DataFrame):
    """Filter low-count genes and fit DESeq2 model."""
    # Align samples
    common = sample_info.index.intersection(counts_df.index)
    counts_df = counts_df.loc[common]
    sample_info = sample_info.loc[common]

    # Filter low-count genes (row sum < 10)
    counts_df = counts_df.loc[:, counts_df.sum(axis=0) >= 10]
    counts_df = counts_df.round().astype(int)

    print(f"Genes after filtering: {counts_df.shape[1]:,}")

    dds = DeseqDataSet(
        counts=counts_df,
        metadata=sample_info,
        design_factors="condition",
        ref_level=["condition", "Control"],
    )
    print("Running DESeq2 model...")
    dds.deseq2()
    return dds, sample_info


# =============================================================
# 4. PCA Plot
# =============================================================
def plot_pca(dds, sample_info: pd.DataFrame, accession: str) -> None:
    """PCA on log-normalised counts to visualise sample clustering."""
    norm_counts = dds.layers["normed_counts"]
    log_counts = np.log1p(norm_counts)

    pca = PCA(n_components=2)
    pcs = pca.fit_transform(log_counts)
    pca_df = pd.DataFrame(pcs, columns=["PC1", "PC2"], index=log_counts.index)
    pca_df = pca_df.join(sample_info)

    plt.figure(figsize=(8, 6))
    sns.scatterplot(
        data=pca_df,
        x="PC1", y="PC2",
        hue="condition", style="condition",
        s=150,
        palette={"Control": "#2166AC", "Ethanol": "#F4A582"},
    )
    plt.title(f"PCA — Alcohol Use on Mouse ({accession})", fontweight="bold")
    plt.xlabel(f"PC1: {pca.explained_variance_ratio_[0] * 100:.1f}% variance")
    plt.ylabel(f"PC2: {pca.explained_variance_ratio_[1] * 100:.1f}% variance")
    plt.legend(loc="lower center", bbox_to_anchor=(0.5, -0.2), ncol=2)
    plt.tight_layout()
    plt.savefig("pca_plot.png", dpi=150)
    plt.show()
    print("Saved: pca_plot.png")


# =============================================================
# 5. Sample Distance Heatmap
# =============================================================
def plot_distance_heatmap(dds, sample_info: pd.DataFrame) -> None:
    """Euclidean distance heatmap for QC and batch-effect detection."""
    norm_counts = dds.layers["normed_counts"]
    log_counts = np.log1p(norm_counts)

    dist_mat = squareform(pdist(log_counts.values, metric="euclidean"))
    dist_df = pd.DataFrame(dist_mat, index=log_counts.index, columns=log_counts.index)

    cmap_blues = sns.color_palette("Blues_r", as_cmap=True)
    condition_colors = sample_info["condition"].map(
        {"Control": "#2166AC", "Ethanol": "#F4A582"}
    )

    sns.clustermap(
        dist_df,
        cmap=cmap_blues,
        figsize=(9, 9),
        col_colors=condition_colors,
        row_colors=condition_colors,
    )
    plt.suptitle("Sample-to-Sample Distance Heatmap", y=1.02)
    plt.savefig("distance_heatmap.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("Saved: distance_heatmap.png")


# =============================================================
# 6. Extract Results & LFC Shrinkage
# =============================================================
def extract_results(dds) -> pd.DataFrame:
    """Extract DESeq2 results with apeglm-equivalent LFC shrinkage."""
    stat_res = DeseqStats(
        dds,
        contrast=["condition", "Ethanol", "Control"],
        alpha=0.1,
    )
    stat_res.summary()
    stat_res.lfc_shrink(coeff="condition_Ethanol_vs_Control")
    res_df = stat_res.results_df.dropna(subset=["padj"])

    # Classify genes
    res_df["status"] = "NS"
    res_df.loc[
        (res_df["padj"] < 0.1) & (res_df["log2FoldChange"] > 1), "status"
    ] = "Up"
    res_df.loc[
        (res_df["padj"] < 0.1) & (res_df["log2FoldChange"] < -1), "status"
    ] = "Down"

    up = (res_df["status"] == "Up").sum()
    down = (res_df["status"] == "Down").sum()
    print(f"\nSignificant DE genes (|LFC| > 1, padj < 0.1):")
    print(f"  Upregulated  : {up}")
    print(f"  Downregulated: {down}")
    print(f"  Total        : {up + down}")
    return res_df


# =============================================================
# 7. Volcano Plot
# =============================================================
def plot_volcano(res_df: pd.DataFrame) -> None:
    """Volcano plot: log2 fold change vs adjusted p-value significance."""
    plt.figure(figsize=(10, 8))
    sns.scatterplot(
        x=res_df["log2FoldChange"],
        y=-np.log10(res_df["padj"]),
        hue=res_df["status"],
        palette={"Up": "#B2182B", "Down": "#2166AC", "NS": "grey"},
        alpha=0.6, s=30,
    )
    plt.axvline(-1, color="grey", linestyle="--")
    plt.axvline(1, color="grey", linestyle="--")
    plt.axhline(-np.log10(0.1), color="grey", linestyle="--")

    # Label top 20 significant genes
    top_genes = res_df[res_df["status"] != "NS"].sort_values("padj").head(20)
    texts = [
        plt.text(row["log2FoldChange"], -np.log10(row["padj"]), gene, fontsize=9)
        for gene, row in top_genes.iterrows()
    ]
    adjust_text(texts, arrowprops=dict(arrowstyle="-", color="black", lw=0.5))

    plt.title("Volcano Plot — Ethanol vs Control (Shrunken LFC)", fontweight="bold")
    plt.xlabel("Log2 Fold Change")
    plt.ylabel("-Log10(Adjusted P-value)")
    plt.legend(loc="lower center", bbox_to_anchor=(0.5, -0.15), ncol=3)
    plt.tight_layout()
    plt.savefig("volcano_plot.png", dpi=150)
    plt.show()
    print("Saved: volcano_plot.png")


# =============================================================
# 8. Top-30 Gene Heatmap
# =============================================================
def plot_top30_heatmap(dds, res_df: pd.DataFrame, sample_info: pd.DataFrame) -> None:
    """Z-score normalised heatmap of the 30 most significant DE genes."""
    norm_counts = dds.layers["normed_counts"]
    log_counts = np.log1p(norm_counts)

    top30 = res_df.sort_values("padj").head(30).index
    top30_counts = log_counts.loc[:, top30]

    condition_colors = sample_info["condition"].map(
        {"Control": "#2166AC", "Ethanol": "#F4A582"}
    )
    sns.clustermap(
        top30_counts.T,
        z_score=0,
        cmap="vlag",
        figsize=(10, 10),
        col_colors=condition_colors,
        yticklabels=True,
        xticklabels=False,
    )
    plt.suptitle("Top 30 DE Genes : Z-score Normalised Expression", y=1.02)
    plt.savefig("top30_heatmap.png", dpi=150, bbox_inches="tight")
    plt.show()
    print("Saved: top30_heatmap.png")


# =============================================================
# 9. Top Gene Expression Boxplot
# =============================================================
def plot_top_gene(dds, res_df: pd.DataFrame, sample_info: pd.DataFrame) -> None:
    """Boxplot of normalised counts for the single most significant gene."""
    best_gene = res_df.sort_values("padj").index[0]
    print(f"\nMost significant gene: {best_gene}")

    gene_idx = list(dds.var_names).index(best_gene)
    best_gene_df = pd.DataFrame({
        "count": dds.layers["normed_counts"][:, gene_idx],
        "condition": sample_info["condition"],
    })

    plt.figure(figsize=(7, 6))
    sns.boxplot(
        x="condition", y="count", data=best_gene_df,
        palette={"Control": "#2166AC", "Ethanol": "#F4A582"},
        showfliers=False, alpha=0.7,
    )
    sns.stripplot(
        x="condition", y="count", data=best_gene_df,
        color="black", size=5, alpha=0.6, jitter=True,
    )
    plt.yscale("log")
    plt.title(f"Expression of {best_gene} across conditions", fontweight="bold")
    plt.ylabel("Normalised Count (log10 scale)")
    plt.xlabel("Condition")
    plt.tight_layout()
    plt.savefig(f"{best_gene}_expression.png", dpi=150)
    plt.show()
    print(f"Saved: {best_gene}_expression.png")


# =============================================================
# 10. GO & Reactome Pathway Enrichment
# =============================================================
def run_enrichment(res_df: pd.DataFrame) -> None:
    """GO Biological Process and Reactome enrichment via GSEAPy Enrichr."""
    sig_genes = res_df[
        (res_df["padj"] < 0.1) & (abs(res_df["log2FoldChange"]) > 1)
    ].index.tolist()

    print(f"\nRunning enrichment on {len(sig_genes)} significant genes...")
    enr = gp.enrichr(
        gene_list=sig_genes,
        gene_sets=["GO_Biological_Process_2021", "Reactome_2022"],
        organism="Mouse",
        outdir=None,
    )

    enr_results = enr.results
    enr_results = enr_results[enr_results["Adjusted P-value"] < 0.2]

    if not enr_results.empty:
        print("\nTop 10 Enriched Terms:")
        print(
            enr_results.head(10)[["Term", "Overlap", "Adjusted P-value", "Genes"]]
        )
        go_bp = enr_results[
            enr_results["Gene_set"] == "GO_Biological_Process_2021"
        ]
        if not go_bp.empty:
            gp.dotplot(
                enr.results,
                column="Adjusted P-value",
                title="GO Enrichment Analysis — Ethanol vs Control",
                cmap="viridis_r",
                top_term=15,
                figsize=(8, 6),
            )
            plt.savefig("go_enrichment_dotplot.png", dpi=150, bbox_inches="tight")
            plt.show()
            print("Saved: go_enrichment_dotplot.png")
    else:
        print("No enriched terms found with current thresholds (padj < 0.2).")


# =============================================================
# 11. Export Results to CSV
# =============================================================
def export_results(res_df: pd.DataFrame) -> None:
    """Write full and significant-only results to CSV files."""
    f_full = "PyDESeq2_Ethanol_vs_Control_full_results.csv"
    f_sig  = "PyDESeq2_Ethanol_significant_genes_padj01.csv"

    res_df.to_csv(f_full)
    sig_df = res_df[res_df["padj"] < 0.1]
    sig_df.to_csv(f_sig)

    print(f"\n✓ Full results    → {f_full}")
    print(f"✓ Significant     → {f_sig}")
    print(f"  Total significant (padj < 0.1): {len(sig_df)}")


# =============================================================
# MAIN
# =============================================================
if __name__ == "__main__":
    GEO_ACCESSION  = "GSE222445"
    COUNT_FILE     = "GSE222445_manuscript.adults.raw.counts.csv.gz"

    counts_df   = load_counts(COUNT_FILE)
    sample_info = fetch_metadata(GEO_ACCESSION)
    dds, sample_info = run_deseq2(counts_df, sample_info)

    plot_pca(dds, sample_info, GEO_ACCESSION)
    plot_distance_heatmap(dds, sample_info)

    res_df = extract_results(dds)

    plot_volcano(res_df)
    plot_top30_heatmap(dds, res_df, sample_info)
    plot_top_gene(dds, res_df, sample_info)
    run_enrichment(res_df)
    export_results(res_df)

    print("\nAll analysis complete!")
