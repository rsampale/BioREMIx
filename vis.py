import streamlit as st
import pandas as pd
import numpy as np
import json
import requests
import plotly.graph_objects as go
import plotly.express as px
import re
import altair as alt
import matplotlib.pyplot as plt
from db_utils import (
    get_dynamodb_table,
    fetch_gene_list_from_ddb,
    fetch_contribs_from_ddb,
    fetch_enrichr_libs
)

### DDB FETCHING FUNCTIONS USED TO ACCESS HPA DATA IN SOME VISUALIZATIONS ###

# ── CACHE the DynamoDB connection ─────────────────
@st.cache_resource
def get_table():
    return get_dynamodb_table()

# ── CACHE the gene list ───────────
@st.cache_data
def load_genes():
    return fetch_gene_list_from_ddb(get_table())

# ── CACHE each gene’s contribs ─────────────────────
@st.cache_data
def load_contribs(gene: str):
    return fetch_contribs_from_ddb(get_table(), gene)

### END DDB FETCHING FUNCTIONS ###


def plot_residues(df: pd.DataFrame) -> go.Figure:

    st.divider()

    df["Protein_length"] = pd.to_numeric(df["Protein_length"], errors="coerce")
    df.dropna(subset=["Protein_length"], inplace=True)
    df["Protein_length"] = df["Protein_length"].astype(int)
 
    length_median = df["Protein_length"].median()
    length_std = df["Protein_length"].std()
    outlier_threshold = 1
    length_cutoff = length_median + outlier_threshold * length_std
    
    excluded_proteins = df[df["Protein_length"] > length_cutoff] 
    df = df[df["Protein_length"] <= length_cutoff]
    
    residue_columns = [
        "Active_site_residues", "Binding_site_residues", "Transmembrane_domain_residues",
        "Intramembrane_residues", "Glycosylation_residues", "Modified_residues",
        "Signal_peptide_residues", "Disulfide_bond_residues", "DNA_binding", "Propeptide_residues",
        "Lipidation_residues", "Zinc_finger_residues", "Repeat_region_residues", "Coiled_coil_residues", 
        "Transit_peptide_residues", "Cross_linking_residues" 
    ]
 
    df = df[df[residue_columns].notna().any(axis=1)]

    if len(df) > 50:
        df = df.iloc[:50]
 
    def extract_positions(text, max_length):
        if pd.isna(text) or text == "None":
            return []

        positions = []
        for match in re.findall(r'(\d+)\.\.(\d+)', str(text)):
            start, end = int(match[0]), int(match[1])
            positions.extend([p for p in range(start, end + 1) if p <= max_length])
        for match in re.findall(
            r'\b(?:MOD_RES|BINDING|DISULFID|DNA_BIND|SIGNAL|GLYCOSYLATION|INTRAMEMBRANE|TRANSMEMBRANE|ACTIVE_SITE|SITE|PROPEPTIDE|LIPIDATION|ZINC_FINGER|REPEAT|COILED_COIL|TRANSIT|CROSSLINK) +(\d+)',
            str(text)
        ):
            pos = int(match)
            if pos <= max_length:
                positions.append(pos)
        return positions
 
    residue_colors = {
        "Active_site_residues": "red",
        "Binding_site_residues": "blue",
        "Transmembrane_domain_residues": "green",
        "Intramembrane_residues": "purple",
        "Glycosylation_residues": "orange",
        "Modified_residues": "cyan",
        "Signal_peptide_residues": "magenta",
        "Disulfide_bond_residues": "goldenrod",
        "DNA_binding": "brown",
        "Propeptide_residues": "darkgreen",
        "Lipidation_residues": "deeppink",
        "Zinc_finger_residues": "slateblue",
        "Repeat_region_residues": "teal",
        "Coiled_coil_residues": "darkorange",
        "Transit_peptide_residues": "olive",
        "Cross_linking_residues": "indigo"
    }
 
    fig = go.Figure()
    y_offset = 0
    y_ticks = []
    y_labels = []
    added_legends = set()
 
    for _, row in df.iterrows():
        gene_name = row.get("Gene_Name", f"Protein {y_offset}")
        uniprot_id = row.get("Uniprot_ID", "Unknown ID")
        protein_length = row["Protein_length"]
 
        raw_synonyms = row.get("Gene_Name_Synonyms", "")
        gene_name_upper = str(gene_name).strip().upper()
        synonyms_list = [
            s.strip() for s in str(raw_synonyms).split(",")
            if s.strip() and s.strip().upper() != gene_name_upper
        ]
        synonyms_display = ", ".join(synonyms_list) if synonyms_list else "N/A"
        display_name = f"{gene_name} ({uniprot_id})"
 
        hover_text = (
            f"<b>Gene:</b> {gene_name}<br>"
            f"<b>Length:</b> {protein_length}<br>"
            f"<b>UniProt ID:</b> {uniprot_id}<br>"
            f"<b>Synonyms:</b> {synonyms_display}"
        )
 
        y_ticks.append(y_offset)
        y_labels.append(gene_name)
 
        fig.add_trace(go.Scatter(
            x=[0, protein_length],
            y=[y_offset, y_offset],
            mode="lines",
            line=dict(color="black", width=5),
            showlegend=False
        ))
 
        marker_positions = list(range(0, protein_length + 1, 5))
        fig.add_trace(go.Scatter(
            x=marker_positions,
            y=[y_offset] * len(marker_positions),
            mode="markers",
            marker=dict(size=8, color="rgba(0,0,0,0)"),
            hoverinfo="text",
            text=[hover_text] * len(marker_positions),
            showlegend=False
        ))
 
        sub_offset = y_offset - 0.2
        for residue, color in residue_colors.items():
            if pd.notna(row.get(residue)):
                raw_text = str(row[residue])
                positions = extract_positions(raw_text, protein_length)
                if not positions:
                    continue
 
                show_legend = residue not in added_legends
                added_legends.add(residue)
 
                ranges = re.findall(r'(\d+)\.\.(\d+)', raw_text)
                used_positions = set()
 
                for start, end in ranges:
                    start = int(start)
                    end = int(end)
                    range_positions = [p for p in range(start, end + 1) if p <= protein_length]
                    if not range_positions:
                        continue
                    used_positions.update(range_positions)
 
                    fig.add_trace(go.Scatter(
                        x=[start, end],
                        y=[sub_offset, sub_offset],
                        mode="lines",
                        line=dict(color=color, width=2),
                        showlegend=False,
                        hoverinfo="skip"
                    ))
 
                    fig.add_trace(go.Scatter(
                        x=range_positions,
                        y=[sub_offset] * len(range_positions),
                        mode="markers",
                        marker=dict(color=color, size=6),
                        name=residue,
                        hoverinfo="text",
                        text=[f"{display_name} - {residue} at {start}–{end}"] * len(range_positions),
                        showlegend=show_legend
                    ))
 
                    fig.add_trace(go.Scatter(
                        x=[start, start],
                        y=[sub_offset - 0.1, sub_offset + 0.1],
                        mode="lines",
                        line=dict(color="lightgrey", width=1),
                        hoverinfo="skip",
                        showlegend=False
                    ))
 
                    show_legend = False
 
                singles = [p for p in positions if p not in used_positions]
                if singles:
                    fig.add_trace(go.Scatter(
                        x=singles,
                        y=[sub_offset] * len(singles),
                        mode="markers",
                        marker=dict(color=color, size=6),
                        name=residue,
                        hoverinfo="text",
                        text=[f"{display_name} - {residue} at {p}" for p in singles],
                        showlegend=show_legend
                    ))
 
                    for p in singles:
                        fig.add_trace(go.Scatter(
                            x=[p, p],
                            y=[sub_offset - 0.1, sub_offset + 0.1],
                            mode="lines",
                            line=dict(color="lightgrey", width=1),
                            hoverinfo="skip",
                            showlegend=False
                        ))
 
                sub_offset -= 0.2
 
        y_offset -= 1.5
 
    fig.update_layout(
        title="Protein Residue Annotations",
        title_x=0.5,
        xaxis=dict(
            title="Residue Position",
            title_font=dict(size=16, family="Arial", color="black"),
            tickfont=dict(size=12)
        ),
        yaxis=dict(
            title="Gene Name",
            title_font=dict(size=16, family="Arial", color="black"),
            tickmode="array",
            tickvals=y_ticks,
            ticktext=y_labels,
            tickfont=dict(size=14, family="Arial", color="black"),
            showticklabels=True,
            range=[y_offset - 1, 1]
        ),
        height=max(600, len(df) * 100),
        width=1000,
        legend_title="Residue Types",
        legend=dict(
            orientation="h",
            x=0,
            y=1.2
        ),
        margin=dict(t=223),
        hovermode="closest"
    )

    st.markdown(
        "🔍 **Tip:** Highlight a region to zoom in. Double-click the background to zoom out.",
        unsafe_allow_html=True
    )

    st.plotly_chart(fig, use_container_width=True)
 
    # Show warning if proteins were excluded due to being large outliers
 

    if not excluded_proteins.empty:
        excluded_names = ", ".join(excluded_proteins["Gene_Name"].dropna().unique())
        st.warning(
            f"The following proteins were excluded from visualization because their lengths were greater the length cutoff of {int(length_cutoff)} residues: {excluded_names}"
        )
 
    genes = df["Gene_Name"].dropna().unique()
    selected_gene = st.selectbox("🔗 Select a gene to view its UniProt entry:", options=genes)
 
    gene_row = df[df["Gene_Name"] == selected_gene].iloc[0]
    uniprot_id = gene_row["Uniprot_ID"]
    uniprot_url = f"https://www.uniprot.org/uniprotkb/{uniprot_id}"
 
    st.markdown(f"[Open {selected_gene} in UniProt ↗️]({uniprot_url})", unsafe_allow_html=True)

    st.divider()
    
    
    
# Coexpression visualizer:

def plot_coexpressed(merged_df):
    
    # LEGACY VERSION - NON HEATMAP

    # big_df = pd.read_parquet("data/hpa_top100_coexpressed.parquet")

    # st.divider()
    # st.header("Tissue Co-expression Visualizer")

    # use_refined_only = st.checkbox("Search only current/refined genes",value=True)
    # if use_refined_only:
    #     gene = st.selectbox("Pick a gene:", sorted(st.session_state.user_refined_df['Gene_Name'].unique()))
    # else:
    #     gene = st.selectbox("Pick a gene:", sorted(big_df['gene'].unique()))

    # k = st.slider("How many partners (k)?", 1, 15, 10) # default 5 most coexpressed genes

    # df_hpa = (
    #     big_df[big_df["gene"] == gene]
    #     .nlargest(k, "pearson_r")
    #     .reset_index(drop=True)
    #     .loc[:, ["partner", "pearson_r", "top_celltypes"]]
    # )

    # st.dataframe(df_hpa,
    #              use_container_width=True,
    #              column_config={
    #                  "top_celltypes": st.column_config.Column(
    #                      "Tissues with Increased Expression",
    #                      width="large"
    #                  ),
    #                  "partner": st.column_config.Column(
    #                      "Gene Partner"),
    #                  "pearson_r": st.column_config.Column(
    #                      help="Tissue expression profile correlation (pearson)")
    #              })
    # st.write("**Source:** Human Protein Atlast (HPA)")
    
    
    ##### FULL VERSION W/ HEATMAP #####
    st.divider()
    st.header("Tissue Co-expression Visualizer")
    
    use_refined_only2 = st.toggle("Search only current refined genes", value=True)
    if use_refined_only2:
        gene_list = sorted(merged_df['Gene_Name'].unique())
    else:
        gene_list = load_genes()
    
    query2 = st.selectbox("Choose query gene", gene_list)
    k2 = st.slider("How many gene partners (k)?", 1, 15, 10)

    if query2:
        df, pr = load_contribs(query2) # retrieves the tuple of the partenr genes contrib data and the pearson_r series
        
        if df.empty:
            st.warning(f"No data for {query2}")
        else:
            # sort partners by descending pearson_r, then take top k
            top_partners = pr.sort_values(ascending=False).index[:k2]
            df_top = df.loc[top_partners]
            
            # new labels to include and embed correlation
            pr_top = pr.loc[top_partners]
            y_labels = [f"{gene} (r={pr_top[gene]:.2f})" for gene in pr_top.index]
            
            fig = px.imshow(
                df_top,
                labels={"x": "Tissue", "y": "Partner Gene", "color": "Contribution"},
                x=df_top.columns,
                y=y_labels, # use custom labels defined above
                aspect="auto",
            )
            fig.update_layout(
                title=f"Tissue-Gene Coexpression (colored by Correlation Contributions) for {query2} (Top {k2} partner genes)",
                height=600, margin=dict(l=100, r=20, t=50, b=100)
            )
            st.plotly_chart(fig, use_container_width=True)

def plot_gsea(merged_df):

    ENRICHR_ADDLIST_URL = 'https://maayanlab.cloud/Enrichr/addList' # Maayan Lab Enrichr API URL
    ENRICHR_ENRICH_URL = 'https://maayanlab.cloud/Enrichr/enrich'
    query_string = '?userListId=%s&backgroundType=%s'

    # Get list of available libraries to search
    libraries_url = "https://maayanlab.cloud/Enrichr/datasetStatistics"
    libraries = fetch_enrichr_libs(libraries_url)
    if libraries is None:
        st.error("Failed to fetch libraries from Enrichr. Library search is disabled.")
    else:
        libraries = sorted(libraries)
    


    st.divider()
    st.header("Gene Set Enrichment Analysis (GSEA)")

    custom_genes = st.toggle("Analyze a custom set of genes (Default: Current refined genes)")

    # Get genes_str (needed for input to the Analyze Gene Set POST endpoint)
    if custom_genes:
        custom_gene_list = st.text_area("Enter a list of genes (comma-separated):", value="").split(",")
        custom_gene_list = [gene.strip() for gene in custom_gene_list if gene.strip()]
        description = st.text_area("Enter a description for your gene list:", value="Custom Gene List")
    else:
        custom_gene_list = merged_df['Gene_Name'].unique().tolist()
        description = "Custom Gene List"

    genes_str = '\n'.join(custom_gene_list)


    # Search available libraries and use it for the analysis, then run analysis
    if libraries is not None:
        selected_library = st.selectbox("Select a library to search:", libraries, index=69)
    else:
        selected_library = st.text_input("Manually type name of the Enrichr library you want to use in the analysis:","GO_Molecular_Function_2025")
    
    if selected_library: # Button to initialize the analysis

        run_analysis = st.button("**Click To Run Analysis!**",use_container_width=True,type="primary")
        if run_analysis:

            # Retrieve 'userListId' from Enrichr
            payload = {
                'list' : (None, genes_str),
                'description' : (None, description)
            }
            addlist_response = requests.post(ENRICHR_ADDLIST_URL, files=payload)
            if not addlist_response.ok:
                st.error(f"Error: {addlist_response.status_code} - {addlist_response.text}") 
                return

            data = json.loads(addlist_response.text)
            user_list_id = data['userListId']
            st.success(f"Gene list uploaded successfully! User List ID: {user_list_id}")

            # Run the enrichment analysis using the selected library
            enrich_response = requests.get(ENRICHR_ENRICH_URL + query_string % (user_list_id, selected_library))
            if not enrich_response.ok:
                st.error(f"Error: {enrich_response.status_code} - {enrich_response.text}")
                return

            data = json.loads(enrich_response.text)
            if not data:
                st.warning("No enrichment results found.")
                return
            
            # Display the results
            cols = [
                "Rank", "Term", "P_value", "Z_score", "Combined_score",
                "Genes", "Adj_P_value", "Old_P_value", "Old_Adj_P_value"
            ]
            df = pd.DataFrame(data[selected_library], columns=cols)
            # convert numeric columns
            for c in ["P_value", "Z_score", "Combined_score", "Adj_P_value"]:
                df[c] = pd.to_numeric(df[c])
            # Store in session state for persistence
            st.session_state.gsea_df = df
            st.session_state.gsea_library = selected_library

        if 'gsea_df' not in st.session_state:
            st.session_state.gsea_df = None
        df = st.session_state.gsea_df
        if df is None:
            st.info("Click **Run Analysis!** to load enrichment results.")
            return

        # SIDEBAR STUFF - PICK FILTER AND SCORE OPTIONS
        st.sidebar.header("Enrichment Filter")
        mode = st.sidebar.radio("Filter by", ("Top N", "FDR threshold"))

        score_options = {
            "Combined score": "Combined_score",
            "–log₁₀(p-value)": None,  # compute below
            "Z-score": "Z_score"
        }
        score_label = st.sidebar.selectbox("Score to rank by", list(score_options.keys()))
        # if user picks –log10(p), make a new column
        if score_options[score_label] is None:
            df["MinusLog10P"] = -df["P_value"].apply(lambda p: pd.np.log10(p))
            rank_col = "MinusLog10P"
        else:
            rank_col = score_options[score_label]

        # APPLY THE SELECTED FILTERING
        if mode == "Top N":
            top_n = st.sidebar.slider("Show top N terms", 1, min(50, len(df)), 10)
            df_filtered = df.nlargest(top_n, rank_col)

        else:
            fdr_cut = st.sidebar.slider("FDR (Adj P-value) ≤", 0.0, 0.1, 0.05, step=0.005)
            df_fdr = df[df["Adj_P_value"] <= fdr_cut]
            df_filtered = df_fdr.sort_values(rank_col, ascending=False).head(50)
            st.sidebar.markdown(
                f"**{len(df_fdr)}** terms pass FDR ≤ {fdr_cut:.3f}, showing top **{len(df_filtered)}**"
            )

        
        # DISPLAY THE RESULTS AS DF

        st.subheader(f"Top {len(df_filtered)} enriched terms")
        st.write("**Select number of terms and filtering method from sidebar**")

        st.dataframe(
            df_filtered[["Term", rank_col, "Adj_P_value", "Genes"]]
            .rename(columns={rank_col: score_label})
            .reset_index(drop=True)
        )

        # DISPLAY A BAR CHART OF THE SCORES
        st.info("NOTE: You can make the plots larger by clicking the **Expand** button in the top right corner of the plot.")
        if st.checkbox("Show bar chart of scores", value=True):
            # use Matplotlib horizontal bars for readability
            scores = df_filtered.set_index("Term")[rank_col]
            plt.figure(figsize=(8, len(scores)*0.3))
            plt.barh(scores.index, scores.values)
            plt.xlabel(score_label)
            plt.ylabel("GO Term")
            plt.gca().invert_yaxis()  # highest at top
            st.pyplot(plt)

        # DISPLAY AN ALTAIR PLOT
        if st.checkbox("Show bubble plot of scores", value=True):
            chart = (
                alt.Chart(df_filtered)
                .mark_circle(size=100)
                .encode(
                    x=f"{rank_col}:Q",
                    y=alt.Y("Term:N", sort="-x"),
                    color="Z_score:Q",
                    tooltip=["Term", "P_value", "Adj_P_value", "Genes"]
                )
                .properties(height=30*len(df_filtered), width=600)
            )
            st.altair_chart(chart, use_container_width=True)


def create_regional_hits_plot(
    hits_df: pd.DataFrame,
    ld_df: pd.DataFrame,
    chrom: str,
    region_start: int,
    region_end: int,
    gene_symbol: str
):
    """
    –log10(p) vs position. 
    • Non-disease SNPs: circle, colored by r2 (Viridis).  
    • Disease SNPs: red diamond (ignores r2).  
    Embeds a catalog URL in customdata for click handling.
    """
    df = hits_df.copy()
    df["minus_log10_p"] = -np.log10(df["pval"])

    # merge r2 or fill with 0
    if (not ld_df.empty) and ("variant_id" in ld_df.columns):
        df = df.merge(ld_df[["variant_id","r2"]], on="variant_id", how="left")\
               .fillna({"r2": 0.0})
        colorscale = "Viridis"
    else:
        df["r2"] = 0.0
        colorscale = [(0, "lightgray"), (1, "lightgray")]

    ld_available = (not ld_df.empty) and ("variant_id" in ld_df.columns)
    
    # add hover and URL
    if ld_available:
        df["hover_text"] = (
            "SNP: "     + df["variant_id"]
        + "<br>Pos: " + df["pos"].astype(str)
        + "<br>p = "  + df["pval"].apply(lambda x: f"{x:.2e}")
        + "<br>Trait: "+ df["trait"].replace("", "N/A")
        + "<br>r² = " + df["r2"].apply(lambda x: f"{x:.2f}")
        )
    else:
        df["hover_text"] = (
            "SNP: "     + df["variant_id"]
        + "<br>Pos: " + df["pos"].astype(str)
        + "<br>p = "  + df["pval"].apply(lambda x: f"{x:.2e}")
        + "<br>Trait: "+ df["trait"].replace("", "N/A")
        )
    df["catalog_url"] = df["variant_id"].apply(
        lambda vid: f"https://www.ebi.ac.uk/gwas/variants/{vid}"
    )

    # split disease vs non-disease
    df_nd = df[df["is_disease"] == 0]
    df_d  = df[df["is_disease"] == 1]

    fig = go.Figure()

    if ld_available:
        nd_marker = dict(
            size=8,
            color=df_nd["r2"],
            colorscale=colorscale,
            colorbar=dict(title="r² vs lead"),
            cmin=0, cmax=1
        )
    else:
        nd_marker = dict(
            size=8,
            color="lightgray"
        )

    # add the trace
    fig.add_trace(go.Scattergl(
        x=df_nd["pos"],
        y=df_nd["minus_log10_p"],
        mode="markers",
        marker=nd_marker,
        hoverinfo="text",
        hovertext=df_nd["hover_text"],
        customdata=df_nd[["catalog_url"]].values,
        name="non-disease"
    ))

    # Disease trace (solid red diamonds)
    fig.add_trace(go.Scattergl(
        x=df_d["pos"],
        y=df_d["minus_log10_p"],
        mode="markers",
        marker=dict(
            symbol="diamond",
            size=9,
            color="red",
            line=dict(width=1, color="DarkRed")
        ),
        hoverinfo="text",
        hovertext=df_d["hover_text"],
        customdata=df_d[["catalog_url"]].values,
        name="disease"
    ))

    # lead SNP star
    lead_idx = df["pval"].idxmin()
    lead = df.loc[lead_idx]
    fig.add_trace(go.Scatter(
        x=[lead["pos"]],
        y=[lead["minus_log10_p"]],
        mode="markers",
        marker=dict(size=14, symbol="star", color="red"),
        hoverinfo="skip",
        showlegend=False,
    ))

    # GW significance line
    sig = -np.log10(5e-8)
    fig.add_hline(
        y=sig,
        line_dash="dash", line_color="blue",
        annotation_text="p=5×10⁻⁸",
        annotation_position="top right"
    )

    # gene window
    fig.add_shape(
        type="rect",
        x0=region_start, x1=region_end,
        y0=0, y1=sig * 0.05,
        fillcolor="LightSeaGreen", opacity=0.3, line_width=0
    )
    fig.add_annotation(
        x=(region_start+region_end)/2, y=0,
        text=gene_symbol, showarrow=False, yanchor="bottom"
    )

    fig.update_layout(
        title=f"{gene_symbol}: chr{chrom}:{region_start}-{region_end}",
        xaxis_title=f"Chr {chrom} position (bp)",
        yaxis_title="-log₁₀(p-value)",
        margin=dict(t=50, b=50, l=50, r=50),
        clickmode="event+select"
    )

    return fig
