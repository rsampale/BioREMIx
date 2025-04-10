import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import re

def plot_residues(df: pd.DataFrame) -> go.Figure:

    df["Protein_length"] = pd.to_numeric(df["Protein_length"], errors="coerce")
    df.dropna(subset=["Protein_length"], inplace=True)
    df["Protein_length"] = df["Protein_length"].astype(int)
 
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
            y=1.02
        ),
        margin=dict(t=100),
        hovermode="closest"
    )

    st.plotly_chart(fig, use_container_width=True)
 
    genes = df["Gene_Name"].dropna().unique()
    selected_gene = st.selectbox("🔗 Select a gene to view its UniProt entry:", options=genes)
 
    gene_row = df[df["Gene_Name"] == selected_gene].iloc[0]
    uniprot_id = gene_row["Uniprot_ID"]
    uniprot_url = f"https://www.uniprot.org/uniprotkb/{uniprot_id}"
 
    st.markdown(f"[Open {selected_gene} in UniProt ↗️]({uniprot_url})", unsafe_allow_html=True)

    st.divider()