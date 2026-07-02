# 🧬 Differential Gene Expression Analysis
### Effect of Ethanol Exposure on the Adult Mouse Brain Transcriptome

---

## 📌 Overview
An end-to-end RNA-seq data analysis pipeline built in Python to investigate 
how chronic ethanol exposure alters gene expression in the adult mouse cerebellum.
Raw count data from NCBI GEO was processed through statistical modelling, 
visualisation, and pathway enrichment to identify key biological mechanisms 
underlying alcohol-induced neuroinflammation.

---

## 📊 Dataset
| Field | Details |
|-------|---------|
| GEO Accession | GSE222445 |
| Organism | Mus musculus (adult mouse brain) |
| Samples | 10 total (5 Ethanol, 5 Control) |
| Genes Analysed | ~24,000 |

---

## 🔬 Key Results
- **141 significantly differentially expressed genes** identified (padj < 0.1, |LFC| > 1)
- **86 upregulated** and **55 downregulated** genes in ethanol-exposed mice
- **Top gene: Gpx3** — heavily involved in oxidative stress response
- **Enriched pathways:** Leukocyte migration, antigen presentation, tissue regeneration

---

## 🛠️ Tools & Libraries
| Category | Tools |
|----------|-------|
| Language | Python 3 |
| DEG Analysis | PyDESeq2 |
| Data Handling | pandas, NumPy |
| Visualisation | matplotlib, seaborn |
| Pathway Enrichment | GSEAPy (Enrichr) |
| Data Fetching | GEOparse |

---

## 📁 Repository Structure
├── DGE_Analysis.ipynb                          # Main analysis notebook
├── DGE_Project_Report.pdf                      # Full project report with plots
├── DESeq2_Ethanol_vs_Control_full_results.csv  # Complete DEG results
├── DESeq2_Ethanol_significant_genes_padj01.csv # Significant genes only
└── README.md

---

## 🚀 How to Run
```bash
# Clone the repository
git clone https://github.com/yourusername/dge-analysis.git
cd dge-analysis

# Install dependencies
pip install pydeseq2 gseapy geoparse pandas numpy matplotlib seaborn adjustText

# Open the notebook
jupyter notebook DGE_Analysis.ipynb
```

---

## 📈 Visualisations
The pipeline generates the following plots:
- **PCA Plot** — Sample quality and clustering
- **Sample Distance Heatmap** — Inter-sample similarity
- **Dispersion Plot** — Model fit validation
- **MA Plot** — Fold change vs average expression
- **Volcano Plot** — Statistical significance vs magnitude
- **Top 30 Genes Heatmap** — Z-score normalised expression
- **GO Enrichment Bar & Dot Plots** — Biological pathway analysis
- **Enrichment Map** — GO term network visualisation

---

## 👤 Author
**Komarapu Balaji Rao**
B.Tech, Biotechnology & Biochemical Engineering
IIT Kharagpur | 23BT3EP28
