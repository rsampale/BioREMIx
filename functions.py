import streamlit as st
import time
import ast
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import networkx
import numpy as np
import os
import plotly.graph_objects as go
import pandas as pd
import requests
import re
import io
import math
import itertools
import warnings
from openai import OpenAI
from langchain_experimental.agents import create_pandas_dataframe_agent
from langchain_core.prompts import PromptTemplate
from langchain.callbacks.streamlit import StreamlitCallbackHandler # deprecated
from matplotlib_set_diagrams import EulerDiagram
from streamlit_plotly_events import plotly_events
from pyvis.network import Network
from rag import col_retrieval_rag
from vis import plot_residues, plot_coexpressed, plot_gsea, create_regional_hits_plot
from db_utils import get_clinvar_variant_info, get_gwas_variant_info

def authenticate():
    # placeholders variables for UI 
    title_placeholder = st.empty()
    help_placeholder = st.empty()
    password_input_placeholder = st.empty()
    button_placeholder = st.empty()
    success_placeholder = st.empty()
    
    # check if not authenticated 
    if not st.session_state['authenticated']:
        # UI for authentication
        with title_placeholder:
            st.title("Welcome to BioREMIx")
        with help_placeholder:
            with st.expander("**⚠️ Read if You Need Help With Password**"):
                st.write("To request or get an updated password contact developers.")
            
                st.write("**Remi Sampaleanu** remi@wustl.edu")
            # UI and get get user password
            with password_input_placeholder:
                user_password = st.text_input("Enter the application password:", type="password", key="pwd_input")
            check_password = True if user_password == st.secrets["PASSWORD"] else False
            # Check user password and correct password
            with button_placeholder:
                if st.button("Authenticate") or user_password:
                    # If password is correct
                    if check_password:
                        st.session_state['authenticated'] = True
                        password_input_placeholder.empty()
                        button_placeholder.empty()
                        success_placeholder.success("Authentication Successful!")
                        st.balloons()
                        time.sleep(1)
                        success_placeholder.empty()
                        title_placeholder.empty()
                        help_placeholder.empty()
                    else:
                        st.error("❌ Incorrect Password. Please Try Agian.")

                        

def reboot_hypothesizer():
    # Make a copy of the session_state keys
    keys = list(st.session_state.keys())
            
    # Iterate over the keys
    for key in keys:
        # If the key is not 'authenticated', delete it from the session_state
        if key not in ['authenticated','genes_info_df','genes_colmeta_dict','colmeta_dict','colmeta_df']:
            del st.session_state[key]
            
def refineloop_buttonclick(): # SHOULD MAKE THESE JUST TOGGLE MAYBE INSTEAD OF HARDCODING TRUE OR FALSE
    st.session_state['refine_section_visible'] = False
    st.session_state['do_refine_loop'] = True
    st.session_state['show_chat_analyze_buttons'] = True
    st.session_state['show_refine_analyze_buttons'] = False
    st.session_state['show_refine_chat_buttons'] = False
    st.session_state['data_chat'] = False
    st.session_state['analyze_data'] = False
    st.session_state["most_recent_chart_selection"] = None

def chat_buttonclick():
    st.session_state['refine_section_visible'] = False
    st.session_state['do_refine_loop'] = False
    st.session_state['show_chat_analyze_buttons'] = False
    st.session_state['show_refine_analyze_buttons'] = True
    st.session_state['show_refine_chat_buttons'] = False
    st.session_state['data_chat'] = True
    st.session_state['analyze_data'] = False
    st.session_state["most_recent_chart_selection"] = None

def analyze_buttonclick():
    st.session_state['refine_section_visible'] = False
    st.session_state['do_refine_loop'] = False
    st.session_state['show_chat_analyze_buttons'] = False
    st.session_state['show_refine_analyze_buttons'] = False
    st.session_state['show_refine_chat_buttons'] = True
    st.session_state['data_chat'] = False
    st.session_state['analyze_data'] = True

def clear_text(text_element):
    if text_element == 'rq_prompt':
        st.session_state['provide_rq_text'] = ""
    elif text_element == 'ref_prompt':
        st.session_state['provide_refinement_text'] = ""

def submit_text(location):
    if location == 'initial_refinement':
        st.session_state.user_refinement_q = st.session_state.init_refinement_q_widget
        st.session_state.last_refinement_q = st.session_state.init_refinement_q_widget
        st.session_state.init_refinement_q_widget = None
    elif location == 'repeat_refinement':
        if st.session_state.repeat_refinement_q_widget: # Only update the last refinement if the widget is not empty
            st.session_state.last_refinement_q = st.session_state.repeat_refinement_q_widget
        st.session_state.user_refinement_q = st.session_state.repeat_refinement_q_widget
        st.session_state.repeat_refinement_q_widget = None
        
def undo_last_refinement(refinement):
    # st.write("EXECUTED UNDO")
    if refinement == "initial":
        if len(st.session_state.gene_df_history) >= 1:
            st.session_state.gene_df_history.pop()
        st.session_state.user_refinement_q = None
        st.session_state.skipped_initial_refine = False
        st.session_state.last_pandas_code = None
        st.session_state.used_uploaded_goi = False
    elif refinement == "repeat":
        if len(st.session_state.gene_df_history) > 1:
            st.session_state.gene_df_history.pop()
            st.session_state.user_refined_df = st.session_state.gene_df_history[-1][0] # Gets the df part of the most recent tuple in the history
            st.session_state.last_refinement_q = st.session_state.gene_df_history[-1][1]
            st.session_state.last_pandas_code = st.session_state.gene_df_history[-1][2]

def apply_uploaded_goi():
    st.session_state['user_refinement_q'] = (
        f"Filter the dataframe to only include genes in the list: "
        f"{st.session_state['uploaded_goi_list'][:5]}..."
    )
    st.session_state['skipped_initial_refine'] = False

    # actually filter the dataframe
    goi_list = st.session_state["uploaded_goi_list"]
    goi_set  = set(goi_list)  # for O(1) lookups
    base_df = st.session_state["relevant_cols_only_df"]
    mask_name = base_df["Gene_Name"].isin(goi_set)
    mask_synonyms = (
        base_df["Gene_Name_Synonyms"]
          .fillna("")                  # avoid NaN
          .str.split(",")              # make lists
          .apply(lambda lst: any(sym.strip() in goi_set for sym in lst))
    )
    filtered_df = base_df[mask_name | mask_synonyms]
    # st.dataframe(filtered_df) # for testing
    st.session_state['user_refined_df'] = filtered_df
    st.session_state['last_refinement_q'] = st.session_state['user_refinement_q']
    st.session_state['last_pandas_code'] = (
        "relevant_cols_only_df["
          "(relevant_cols_only_df['Gene_Name'].isin(goi_list)) | "
          "(relevant_cols_only_df['Gene_Name_Synonyms']"
            ".fillna('')"
            ".str.split(',')"
            ".apply(lambda lst: any(sym.strip() in goi_list for sym in lst)))"
        "]"
    )
    # mark that we used the uploaded list
    st.session_state['used_uploaded_goi'] = True

@st.cache_data(show_spinner=False)
def fetch_gwas_hits_by_gene(
    symbol: str,
    max_snps: int = 30
) -> pd.DataFrame:
    """
    1) Fetch all SNPs mapped to `symbol` via:
         GET /singleNucleotidePolymorphisms/search/findByGene?geneName=<symbol>&size=500
       (page through `_links.next` if needed). Each SNP object has `rsId`, 
       and `locations[0].chromosomeName` + `chromosomePosition` (GRCh38).
    2) For each rsId, pull its associations via:
         GET /singleNucleotidePolymorphisms/{rsId}/associations?projection=associationBySnp
       That returns the SNP’s association objects; each has:
         - `pvalue`
         - `_links.efoTraits.href` to get trait(s)
    3) From each associationBySnp record, extract:
         - pvalue (float)
         - efoTraits link → call it once → collect embedded `efoTraits[*].trait`
    4) Build a DataFrame of (variant_id, chrom, pos, pval, trait).
    5) Sort by pval ascending, keep only the top `max_snps` rows.
    """

    BASE = "https://www.ebi.ac.uk/gwas/rest/api"

    t_start_total = time.perf_counter()
    print(f"[TIMER] Starting fetch_gwas_hits_by_gene('{symbol}')")

    # 1) Fetch SNPs by gene
    t0 = time.perf_counter()
    snps = []
    url  = f"{BASE}/singleNucleotidePolymorphisms/search/findByGene"
    params = {"geneName": symbol, "size": 500}
    while url:
        resp = requests.get(url, params=params, timeout=10)
        resp.raise_for_status()
        js = resp.json()
        page_snps = js.get("_embedded", {}).get("singleNucleotidePolymorphisms", [])
        snps.extend(page_snps)
        nxt = js.get("_links", {}).get("next", {}).get("href")
        url = nxt
        params = {}
    t1 = time.perf_counter()
    print(f"[TIMER] fetch SNP list took {t1 - t0:.2f}s ({len(snps)} SNPs)")

    if not snps:
        print(f"[TIMER] No SNPs for gene '{symbol}'.")
        return pd.DataFrame(columns=["variant_id", "chrom", "pos", "pval", "trait"])

    records = []
    # 2) For each rsId, pull associationBySnp
    for snp in snps:
        rs = snp.get("rsId")
        loc0 = snp.get("locations", [{}])[0]
        chrom = loc0.get("chromosomeName")
        pos = loc0.get("chromosomePosition")
        if (not rs) or (chrom is None) or (pos is None):
            continue

        # Association fetch timing
        t2 = time.perf_counter()
        aurl = f"{BASE}/singleNucleotidePolymorphisms/{rs}/associations"
        aparams = {"projection": "associationBySnp", "size": 500}
        aresp = requests.get(aurl, params=aparams, timeout=10)
        t3 = time.perf_counter()
        print(f"[TIMER] fetch associations for {rs} took {t3 - t2:.2f}s")

        if aresp.status_code != 200:
            continue

        ajs = aresp.json()
        assoc_list = ajs.get("_embedded", {}).get("associations", [])
        if not assoc_list:
            continue

        # pick the association with smallest pvalue
        best_p = math.inf
        best_trait = ""
        for a in assoc_list:
            pval = a.get("pvalue")
            if pval is None:
                continue
            if pval < best_p:
                best_p = pval
                # extract embedded efoTraits if present
                efo_emb = a.get("_embedded", {}).get("efoTraits", [])
                if efo_emb:
                    traits = [t.get("trait", "") for t in efo_emb if t.get("trait")]
                    best_trait = "; ".join(traits)
                else:
                    # fallback: follow the efoTraits link
                    efo_link = a.get("_links", {}).get("efoTraits", {}).get("href", "")
                    if efo_link:
                        t4 = time.perf_counter()
                        t_resp = requests.get(efo_link, timeout=10)
                        t5 = time.perf_counter()
                        print(f"[TIMER] fetch EFO for {rs} took {t5 - t4:.2f}s")
                        if t_resp.status_code == 200:
                            t_js = t_resp.json()
                            e_list = t_js.get("_embedded", {}).get("efoTraits", [])
                            best_trait = "; ".join([t.get("trait", "") for t in e_list if t.get("trait")])

        if best_p == math.inf:
            continue

        records.append({
            "variant_id": rs,
            "chrom"     : chrom,
            "pos"       : int(pos),
            "pval"      : best_p,
            "trait"     : best_trait
        })

    if not records:
        print(f"[TIMER] No associations found for gene '{symbol}'. Exiting.")
        return pd.DataFrame(columns=["variant_id", "chrom", "pos", "pval", "trait"])

    df_top = pd.DataFrame(records)

    # 5) Sort by pval ascending, keep top max_snps
    # t6 = time.perf_counter()
    # df_top = df.sort_values("pval", ascending=True).head(max_snps).reset_index(drop=True)
    # t7 = time.perf_counter()
    # print(f"[TIMER] Sorting & slicing top {max_snps} took {t7 - t6:.2f}s")

    t_end_total = time.perf_counter()
    print(f"[TIMER] Total fetch_gwas_hits_by_gene time: {t_end_total - t_start_total:.2f}s")

    return df_top[["variant_id", "chrom", "pos", "pval", "trait"]]


@st.cache_data(show_spinner=False)
def fetch_ld_r2(lead_snp: str, population: str, token: str) -> pd.DataFrame:
    """
    Call LDlink LDproxy for lead_snp in 'population'. Requires non-empty token.
    Returns a DataFrame with columns ["variant_id","r2"].
    """
    params = {
        "var"   : lead_snp,
        "pop"   : population,
        "r2_d"  : "r2",
        "token" : token
    }
    r = requests.get("https://ldlink.nci.nih.gov/LDlinkRest/ldproxy", params=params, timeout=10)
    r.raise_for_status()
    txt = r.text
    df = pd.read_csv(
        pd.compat.StringIO(txt), sep="\t",
        dtype={"#RS_Number": str, "R2": float}
    ).rename(columns={"#RS_Number": "variant_id", "R2": "r2"})
    return df[["variant_id","r2"]]

def llm_disease_lookup(inp_traits: set, llm) -> dict:
    """
    Returns a mapping from trait to a binary disease flag: 1 if it's a disease, 0 otherwise.
    """
    traits_as_list = sorted(list(inp_traits))  # Sort to keep consistent order (optional)
    formatted_traits = ", ".join([f"'{trait}'" for trait in traits_as_list])  # Safely quote each one

    prompt = PromptTemplate(
        template="""
        You are a helpful assistant trained on biomedical trait classification.

        - Here is a list of traits from a GWAS or ClinVar study: [{traits}]
        - Some refernce actual diseases/conditions, others do not.

        Instructions:
        - For each trait, determine if it would be considered a **disease**.
        - Output a Python dictionary where each trait is mapped to 1 if it is a disease, 0 if not.
        - Return ONLY the dictionary. Example format:
        {{
            'amyotrophic lateral sclerosis': 1,
            'body height': 0,
            ...
        }}
        """,
        input_variables=["traits"]
    )

    chain = prompt | llm
    try:
        parser_output = chain.invoke({"traits": formatted_traits})
        # st.write("LLM raw output:")
        # st.code(parser_output.content)
        out_dict = ast.literal_eval(parser_output.content)
    except Exception as e:
        st.error(f"Failed to parse LLM output: {e}")
        raise

    return out_dict
    
def make_dbsnp_link(rsid):
    """
    Build an HTML link to the dbSNP record.
    Accepts:
      • plain integers like 5345346
      • strings "5345346" or "rs5345346" (case-insensitive)
    Returns "Unknown" for -1, NaN, empty, or anything non-numeric.
    """
    # return unkwnown if can't be identified as an ID
    if pd.isna(rsid):
        return "Unknown"

    rsid_str = str(rsid).strip()

    if rsid_str in {"-1", "", "nan"}:
        return "Unknown"

    # If it’s already ‘rs…’ keep it; if it’s just digits, prepend ‘rs’
    if rsid_str.lower().startswith("rs"):
        core = rsid_str[2:]
    else:
        core = rsid_str

    # Validate that the remaining part is all digits (is not an rsid otherwise)
    if not re.fullmatch(r"\d+", core):
        return "Unknown"

    rsid_full = f"rs{core}" # put prefix back or make sure its there 
    url = f"https://www.ncbi.nlm.nih.gov/snp/{rsid_full}"
    return f'<a href="{url}" target="_blank">{rsid_full}</a>'

def show_gwas_tool(merged_df: pd.DataFrame, llm): # NEEDS TO LIVE IN SAME FILE AS HELPER FUNCTIONS ABOVE
    """
    All UI and logic for the GWAS‐hit visualizer.
    Called when corresponding button is clicked in analyze_data.
    """

    clinvar_cols = [ # Used later when selecting which columns to keep from S3 retrieved data
        "RS# (dbSNP)",
        "Type",
        "ClinicalSignificance",
        "PhenotypeList",
        "LastEvaluated",
        "Chromosome",
        "Start",
        "Stop"
    ]
    gwas_cols = [
        "SNPS",
        "MAPPED_GENE",
        "MAPPED_TRAIT",
        "DISEASE/TRAIT",
        "P-VALUE",
        "LINK",
        "STUDY",
        "CHR_ID",
        "CHR_POS",
        "CONTEXT",
        "DATE ADDED TO CATALOG"
    ]
    STOPWORDS = {"not provided", "not specified", "see cases"}
    def normalize_trait(trait: str) -> str: # regex helper for cleaning up some traits later and getting better clinvar shared matches
        t = trait.lower()
        # remove stopwords
        for stop in STOPWORDS:
            t = re.sub(rf'\b{re.escape(stop)}\b', '', t)
        # strip digits
        t = re.sub(r'\d+', '', t)
        # drop punctuation (keep letters and spaces)
        t = re.sub(r'[^a-z\s]', ' ', t)
        # collapse whitespace
        t = re.sub(r'\s+', ' ', t).strip()
        return t

    st.write("### GWAS Hit Visualizer")

    # 1) Gene-selection dropdown
    use_custom_genes = st.toggle("Analyze a custom set of genes (Default: Current refined genes)",
                                 help="If you have uploaded Genes of Interest (GOI), this will prefill the field.")
    if use_custom_genes:
        uploaded_goi = st.session_state.get("uploaded_goi_list", [])
        default_text = ", ".join(uploaded_goi) if uploaded_goi else ""
        genes_str = st.text_area(
            "Enter a list of genes (comma-separated):",
            value=default_text
        ).strip()
        inputted_gene_list = [g.strip() for g in genes_str.split(",") if g.strip()]
    else:
        inputted_gene_list = merged_df["Gene_Name"].dropna().unique().tolist()

    defaults = inputted_gene_list
    gene_list = st.multiselect(
        "**Select one or more genes (Max 10):**",
        options=defaults,
        default=defaults[:3] if len(defaults) >= 3 else defaults,
        max_selections=10
    )
    if not gene_list:
        st.warning("Please select at least one gene.")
        return

    # 3) Optional LDlink inputs
    with st.expander("**(Optional) LDlink Settings**",expanded=False):
        st.write("Consider linkage disequilibirium (and color plots accordingly) in the analysis")
        ld_token = st.text_input(
            "LDlink API token (leave blank to skip LD):",
            type="password",
            help="If you supply a valid token, r² coloring will be enabled."
        )
        pop = st.selectbox("LD population", ["EUR","AFR","EAS","AMR","SAS"], index=0)

    # 4) Show shared traits checkbox
    show_shared_traits = st.checkbox("Show shared GWAS and ClinVar traits across selected genes", value = True)

    # 5) “Run” button
    if not st.button("Generate GWAS/ClinVar Results", use_container_width=True, type="primary"):
        st.info("Adjust settings above and click Generate GWAS Results.")
        return

    # 6) Fetch hits for each gene
    MAX_SNPS = 30
    hits_dict  = {}
    clinvar_dict = {}

    for gene in gene_list:
        raw = get_gwas_variant_info(gene)
        raw_cv = get_clinvar_variant_info(gene) # clinvar retrieval
        clinvar_dict[gene] = raw_cv # we can use this later in step 8 where we make the display df
        if raw.empty:
            hits_dict[gene]  = pd.DataFrame(columns=["variant_id","chrom","pos","pval","trait"])
            continue

        df_show = raw.loc[:, gwas_cols].copy()

        hits = ( # reshape because hits and the plotting func that uses it needs a specific format
            raw.assign(
                variant_id = raw["SNPS"].str.split(",").str[0].str.strip(),
                chrom      = raw["CHR_ID"],
                pos        = pd.to_numeric(raw["CHR_POS"], errors="coerce"),
                pval       = pd.to_numeric(raw["P-VALUE"], errors="coerce"),
                trait      = raw["MAPPED_TRAIT"].fillna(raw["DISEASE/TRAIT"])
            )
            .loc[:, ["variant_id", "chrom", "pos", "pval", "trait"]]
            .dropna(subset=["variant_id", "pos", "pval"])
            .sort_values("pval")
            .head(MAX_SNPS)
        )

        hits_dict[gene] = hits

    # 7) Shared traits section
    if show_shared_traits:
        trait_pairs = []
        for gene, hits in hits_dict.items():
            for trait in hits["trait"].dropna().unique():
                if trait:
                    trait_pairs.append({"trait": trait, "gene": gene})
        if trait_pairs:
            trait_df = pd.DataFrame(trait_pairs)
            trait_grouped = (
                trait_df
                .groupby("trait")["gene"]
                .agg(list)
                .reset_index()
                .rename(columns={"gene": "genes"})
            )
            trait_grouped["count"] = trait_grouped["genes"].apply(len)
            shared_t = trait_grouped[trait_grouped["count"] > 1].copy()

            if shared_t.empty:
                st.info("No shared traits across selected genes.")
            else:
                shared_t["genes"] = shared_t["genes"].apply(lambda lst: ", ".join(lst))
                shared_t = shared_t[["trait", "genes", "count"]].sort_values(
                    ["count", "trait"], ascending=[False, True]
                )
                st.write("#### Shared GWAS Traits")
                st.dataframe(shared_t.reset_index(drop=True), use_container_width=True)
        else:
            st.info("No traits to compare across selected genes.")

        # 7b) Shared ClinVar traits
        cl_pairs = []
        for gene, retrieved in clinvar_dict.items():
            filtered_cv = (
                retrieved
                .loc[retrieved["ClinSigSimple"] == '1', clinvar_cols]
                .drop_duplicates()
                .reset_index(drop=True)
            )

            for plist in filtered_cv["PhenotypeList"].dropna():
                for raw_trait in re.split(r"[|;]", plist):
                    raw = raw_trait.strip()
                    if not raw or raw.lower() in STOPWORDS:
                        continue

                    norm = normalize_trait(raw)
                    if not norm:
                        continue

                    cl_pairs.append({
                        "trait_raw": raw,
                        "trait_norm": norm,
                        "gene": gene
                    })

        # build a DataFrame and dedupe trait-gene pairs
        cl_df = pd.DataFrame(cl_pairs).drop_duplicates(subset=["trait_norm", "gene"])

        # group and collect unique genes per trait
        cl_grouped = (
            cl_df
            .groupby("trait_norm")
            .agg({
                "gene": lambda g: sorted(set(g)),
                "trait_raw": lambda raws: sorted(set(raws))[0]
            })
            .reset_index()
        )

        # count how many distinct genes each trait has
        cl_grouped["count"] = cl_grouped["gene"].apply(len)

        # only keep traits seen in more than one gene
        shared_cv = cl_grouped[cl_grouped["count"] > 1].copy()

        if shared_cv.empty:
            st.info("No shared ClinVar phenotypes across selected genes.")
        else:
            # turn the gene-lists into comma separated strings
            shared_cv["genes"] = shared_cv["gene"].apply(lambda lst: ", ".join(lst))
            shared_cv = shared_cv[["trait_raw", "genes", "count"]].rename(
                columns={"trait_raw": "trait"}
            )
            shared_cv = shared_cv.sort_values(["count", "trait"], ascending=[False, True])
            st.write("#### Shared ClinVar Phenotypes")
            st.warning("ClinVar labels are inconsistent. While we have tried to account for some of this variation, some shared traits may be missed. Inspect the individual gene tables for a more detailed view of each gene's ClinVar data.")
            st.dataframe(shared_cv.reset_index(drop=True), use_container_width=True)


    # 8) For each gene, output summary table then plot
    for gene in gene_list:
        hits = hits_dict.get(gene)
        if hits is None or hits.empty:
            with st.expander(f"**{gene}** (click to expand)"):
                st.write(f"No published GWAS hits for {gene}.")
            continue

        # derive window
        chrom = hits["chrom"].iloc[0]
        pos_min, pos_max = int(hits["pos"].min()), int(hits["pos"].max())
        window_start, window_end = max(1, pos_min - 500_000), pos_max + 500_000

        with st.expander(f"**{gene}** (click to expand)"):
            gwas_tab, clinvar_tab = st.tabs(["GWAS", "ClinVar"])
            with gwas_tab:
                # A) Annotate hits with is_disease
                traits_set = set(hits["trait"])
                flags_mapping_dict = llm_disease_lookup(traits_set, llm=llm)
                # add it into hits itself:
                hits = hits.copy()
                hits["is_disease"] = hits["trait"].map(flags_mapping_dict)

                # B) Build a display‐only copy for your table
                df_disp = hits.copy()
                df_disp["pval"] = df_disp["pval"].apply(lambda x: f"{x:.2e}")
                df_disp = df_disp.sort_values("is_disease", ascending=False)
                # turn rsids into Markdown links
                df_links = df_disp.copy()
                df_links["variant_id"] = df_links["variant_id"].apply(
                    lambda vid: f'<a href="https://www.ebi.ac.uk/gwas/variants/{vid}" target="_blank">{vid}</a>'
                )

                # convert to HTML (escape=False preserves our <a> tags)
                html_table = df_links.to_html(index=False, escape=False)

                # wrap it in a fixed‐height, scrollable div
                scrollable = f"""
                <div style="max-height:400px; overflow-y:auto; border:1px solid #ddd; padding:5px">
                {html_table}
                </div>
                """
                st.write("**Summary of GWAS hits for this region:**")
                # st.dataframe(df_disp)
                st.markdown(
                    scrollable,
                    unsafe_allow_html=True
                )

                # C) Fetch LD
                lead = hits.loc[hits["pval"].idxmin(), "variant_id"]
                if ld_token:
                    try:
                        ld_df = fetch_ld_r2(lead, pop, ld_token)
                    except Exception as e:
                        st.warning(f"LDlink error for {lead}: {e}\nSkipping LD for {gene}.")
                        ld_df = pd.DataFrame(columns=["variant_id", "r2"])
                else:
                    ld_df = pd.DataFrame(columns=["variant_id", "r2"])

                # D) Plot 
                fig = create_regional_hits_plot(
                    hits,
                    ld_df,
                    chrom,
                    window_start,
                    window_end,
                    gene.upper().strip()
                )
                st.plotly_chart(fig, use_container_width=True)
            with clinvar_tab:
                retrieved_clinvar = clinvar_dict[gene]
                filtered_clinvar = (
                    retrieved_clinvar
                        # keep only pathogenic / likely-pathogenic entries
                        .loc[retrieved_clinvar["ClinSigSimple"] == '1', clinvar_cols]
                        # Remove exact duplicates that sometimes crop up
                        .drop_duplicates()
                        # consistent row ordering:
                        .reset_index(drop=True)
                )
                
                clinvar_links = filtered_clinvar.copy()
                clinvar_links["RS# (dbSNP)"] = clinvar_links["RS# (dbSNP)"].apply(make_dbsnp_link)

                # convert to HTML (escape=False preserves our <a> tags)
                html_table = clinvar_links.to_html(index=False, escape=False)

                # wrap it in a fixed‐height, scrollable div
                scrollable = f"""
                <div style="max-height:400px; overflow-y:auto; border:1px solid #ddd; padding:5px">
                {html_table}
                </div>
                """
                st.write(f"**Summary of ClinVar data for {gene}**")
                st.markdown(
                    scrollable,
                    unsafe_allow_html=True
                )
                # st.dataframe(filtered_clinvar)

# builds a euler diagram for disease association
def build_visual_1(llm):

    all_columns = list(st.session_state.merged_df.columns)
    prompt = PromptTemplate(
    template = """
    - Here is a list of columns in a dataframe: {all_cols}
    - The columns hold information relating to gene names, disease associations, biological processes, and much more.
    - Some names contain acronyms. Try to decode these remembering that this is a biological/genetic dataset.
    Instructions:
                - Using the list of column names, select any column names you think might be relevant to diseases associated with a gene
                - Try to find columns related to neurodegenerative diseases such as SMA, SCA, ALS, juvenile ALS, parkinson, alzeimer, HSP, CMT, dHMN, and possibly others if they are present
                - Do not include any super ambiguous column name like curated diseases, any disease designation, NDD count, or other diseases. Only include column names that seem related to very specific diseases
                - Do not include cancer or diabetes related columns
                - Never include the column for a gene name. Only include columns for disease names
                - Return two lists in a tuple. The first should be the real column names, the second should be the plot labels for these names (thus should be better formatted without underscores, etc.)
                - E.g. Return: (['colname_1','colname_2'],['col_label1','col_label2'])
                - Return ONLY the tuple. Do not add the word python or any quotations. 
    """
    )
    chain = prompt | llm
    parser_output = chain.invoke({"all_cols": all_columns})
    
    # recasts parser_output_content as a tuple
    parser_output_content = ast.literal_eval(parser_output.content)
    
    # separates lists in the tuple
    colnames_list=parser_output_content[0]
    colnames_labels=parser_output_content[1]

    # creates new df with only disease columns and adds to session state
    relevant_cols_only_df = st.session_state.merged_df[colnames_list]
    st.session_state['relevant_cols_only_df'] = relevant_cols_only_df

    # counts disease associations
    string_counts = relevant_cols_only_df.apply(lambda col: (col == 1).sum())

    # remove diseases that have 0 count
    string_counts = string_counts[string_counts > 0]
    
    # makes sure there is data to display
    nonzero_indices = string_counts > 0
    filtered_counts = string_counts[nonzero_indices]
    filtered_labels = [label for label, keep in zip(colnames_labels, nonzero_indices) if keep]
    if len(filtered_counts) == 0:
        st.write("You do not have any data to plot. Try to redo your refinement.")
        return

    # builds disease sets 
    disease_sets = {
        disease: set(st.session_state['relevant_cols_only_df'].index[st.session_state['relevant_cols_only_df'][disease] == 1]) for disease in st.session_state['relevant_cols_only_df'].columns
    }
    disease_sets = {disease: genes for disease, genes in disease_sets.items() if genes}

    # create labels for euler diagram
    disease_list = list(disease_sets.keys())
    disease_list_labels = [disease.replace("_", " ") for disease in disease_list]

    # Assigns unique integer indices to diseases
    disease_index = {disease: i for i, disease in enumerate(disease_list)}

    # creates list of binary tuples for each gene
    disease_tuples = []

    # Iterates over genes and initializes binary tuple for each gene adding to disease_list
    for gene in set.union(*disease_sets.values()):
        binary_tuple = [0] * len(disease_list)

        for disease, genes in disease_sets.items():
            if gene in genes:
                binary_tuple[disease_index[disease]] = 1
        disease_tuples.append(tuple(binary_tuple))
    
    # Iteratively counts number of each disease tuple occurrence
    tuple_counts = {}
    for tup in disease_tuples:
        tuple_counts[tup] = tuple_counts.get(tup, 0) + 1
   
    tuple_counts = {t: c for t, c in tuple_counts.items() if any(t)}

    
    fig1, ax = plt.subplots(figsize=(8, 6))
    if len(disease_list_labels) < 2:
        st.session_state["most_recent_chart_selection"] = "no_euler"
        return
    diagram = EulerDiagram(tuple_counts, set_labels = disease_list_labels, ax = ax)

    # get orgins, radii, width, height to help place labels
    origins = diagram.origins
    radii = diagram.radii
    highest_set = np.max(origins[1])
    lowest_set = np.min(origins[1])
    height_middle = (highest_set + lowest_set)/2
    rightmost_set = np.max(origins[0])
    leftmost_set = np.min(origins[0])

    # places set labels
    label_count = 0
    set_label_artists = diagram.set_label_artists
    for label in set_label_artists:
        setx = origins[label_count][0]
        sety = origins[label_count][1]
        if sety >= height_middle:
            label.set_x(setx)
            label.set_y(sety + radii[label_count])
        else:
            label.set_x(setx)
            label.set_y(sety - radii[label_count])

        label_count += 1
        label.set_fontweight("bold")
        label.set_horizontalalignment("center")

    #color the patches and edgecolors
    subset_artists = diagram.subset_artists
    polygon_count = 0
    for polygon in subset_artists:
        # subset_artists[polygon].set_color(colors[polygon_count])
        subset_artists[polygon].set_edgecolor("black")
        polygon_count += 1

    plt.title("Disease Associations")
    plt.tight_layout()


    # Save figure to BytesIO
    img_bytes = io.BytesIO()
    fig1.savefig(img_bytes, format="png", bbox_inches="tight")
    img_bytes.seek(0)  # Move to the beginning
    # Store in session state
    st.session_state["most_recent_chart_selection"] = img_bytes


#builds a bar chart
def build_visual_2(llm):

    all_columns = list(st.session_state.merged_df.columns)
    prompt = PromptTemplate(
    template = """
    - Here is a list of columns in a dataframe: {all_cols}
    - The columns hold information relating to gene names, disease associations, biological processes, and much more.
    - Some names contain acronyms. Try to decode these remembering that this is a biological/genetic dataset.
    Instructions:
                - Using the list of column names, select one column the you think is most relevant to subcellular location. It may be called something similar to subcellular location. 
                - Return the of the columns you found for subcellular location exactly as they are titled in the dataframe. Do not add any special characters, underscores, parantheses, or quotations. 
                - E.g. Return: ('colname_1')
                - Return ONLY the column name with that formatting. Do not add any special characters or quotations
    """
    )
    chain = prompt | llm
    parser_output = chain.invoke({"all_cols": all_columns})

    
    parser_output_content = parser_output.content.strip('"')
    parser_output_content = parser_output_content.strip('(')
    parser_output_content = parser_output_content.strip(')')

    # creates list of subcellular locations and gets counts for each location
    all_locations = st.session_state.merged_df[parser_output_content].dropna().str.split(";")
    flat_locations = [loc.strip() for sublist in all_locations for loc in sublist]
    location_counts = pd.Series(flat_locations).value_counts()

    # places location counts in ascending order
    ordered_location_counts = location_counts.sort_values(ascending = False)

    # makes sure there is actually data to plot
    nonzero_indices = ordered_location_counts > 0
    filtered_counts = ordered_location_counts[nonzero_indices]
    if len(filtered_counts) == 0:
        st.write("You do not have any data to plot. Try to redo your refinement.")
        return
    
    # grabs top 20 subcellular locations
    top20_counts = filtered_counts.head(20).copy()
    
    # confiures bar chart
    fig1 = go.Figure(
        data=[go.Bar(x=top20_counts.index, y=top20_counts.values, marker_color="palegreen")]
    )
    fig1.update_layout(
        title = dict(
            text = "Distribution of Genes Across Top 20 Subcellular Locations", 
        ),
        xaxis = dict(
            title = dict(
                text = "Subcellular Location",
            ),
            tickangle = 45
        ),
        yaxis = dict(
            title = dict(
                text = "Number of Genes",
            ),
        ),
        template="plotly_white",
        autosize = False,
        height = 600
    )

    # Save the figure to a BytesIO object
    img_bytes = io.BytesIO()
    fig1.write_image(img_bytes, format = "png", scale=2)
    img_bytes.seek(0)  # Move to start

    # Store in session state
    st.session_state["most_recent_chart_selection"] = img_bytes

def build_visual_3(llm): # NETWORK DIAGRAM FOR PROTEINS

    def extract_protein_name(protein):
        match = re.search(r"\[([A-Za-z0-9_]+)\]", protein)
        return match.group(1) if match else protein

    all_columns = list(st.session_state.merged_df.columns)
    prompt = PromptTemplate(
    template = """
    - Here is a list of columns in a dataframe: {all_cols}
    - The columns hold information relating to gene names, disease associations, biological processes, and much more.
    - Some names contain acronyms. Try to decode these remembering that this is a biological/genetic dataset.
    Instructions:
                - Using the list of column names, select 4 columns.
                - First, select the one you think is most relevant to protein interactions. It may be called something similar to interacts with. 
                - Second, select the column that refers to gene names
                - Third, select the column that refers to the id of the genes. It will likely be the first column
                - Fourth, select the column that refers to nicknames or synonyms of the genes.
                - Return the name of the columns you found for subcellular location exactly as they are titled in the dataframe. Do not add any special characters, underscores, parantheses, or quotations. 
                - E.g. Return the column names in a tuple: (colname_1, colname_2, colname_3, colname_4)
                - Return ONLY the column name with that formatting. Do not add any special characters or quotations. Make sure there are no space characters in any of the column names
    """
    )

    chain = prompt | llm
    parser_output = chain.invoke({"all_cols": all_columns})

    parser_output_content = parser_output.content.strip("() ")
    parser_output_content = [col.strip() for col in parser_output_content.split(",")]
    protein_interaction_col, gene_name_col, id_col, synonyms_col = parser_output_content

    protein_interacts = st.session_state.merged_df[protein_interaction_col]
    name_synonyms = (
        st.session_state.genes_info_df[synonyms_col] # Needs to be genes_info_df to get all of them, not just the ones involved in refined proteins
        .str.strip()
    ) 
    all_possible_proteins = st.session_state.genes_info_df[id_col]
    gene_names = st.session_state.genes_info_df[gene_name_col]
    proteins = st.session_state.merged_df[id_col]
    label_map_tuples = list(zip(all_possible_proteins, gene_names, name_synonyms))
    all_proteins = set(proteins)

    # creates a list of interacting proteins to use for nodes
    interactions = []
    for index, neighbors in protein_interacts.items():
        if pd.notna(neighbors) and isinstance(neighbors, str):
            protein = proteins[index] if index < len(proteins) else None
            if protein: 
                neighbor_list = neighbors.split(";")
                for neighbor in neighbor_list:
                    clean_neighbor = extract_protein_name(neighbor.strip())
                    if clean_neighbor and pd.notna(clean_neighbor):
                        interactions.append(clean_neighbor) 
    
    # generates edge pairs in a list of tuples
    edge_pairs = []
    for index, neighbors in protein_interacts.items():
        if pd.notna(neighbors) and isinstance(neighbors, str):
            protein = proteins[index] if index < len(proteins) else None
            if protein: 
                neighbor_list = neighbors.split(";")
                for neighbor in neighbor_list:
                    clean_neighbor = extract_protein_name(neighbor.strip())
                    if clean_neighbor and pd.notna(clean_neighbor):
                        edge_pairs.append((protein, clean_neighbor))
    edge_pairs = list({tuple(sorted(edge)) for edge in edge_pairs})

    protein_interacts.tolist()
    all_proteins = list(all_proteins)
    html_proteins_list = list(itertools.chain(interactions, all_proteins))
    
    file_path = os.path.join("networkdiagram", "protein_network.html")
    

    # creating node/edge properties
    node_ids = []
    for protein in html_proteins_list:
        if protein not in node_ids:
            node_ids.append(protein)
    refined_proteins = all_proteins
    other_proteins = list(set([protein for protein in html_proteins_list if protein not in all_proteins]))

    # creates labels for proteins in a dict, then cleans labels so only one name shows
    labels_dict = {}
    for map in label_map_tuples:
        if map[0] in html_proteins_list:
            labels_dict[map[0]] = [map[1], map[2]]
    for key, value in labels_dict.items():
        if isinstance(value[0], str):
            value[0] = value[0].split(';', 1)[0]
    
    network_label_map = {protein: labels_dict[protein][0] if protein in labels_dict else f"ID: {protein}" for protein in html_proteins_list}

    selection_label_map = {protein: labels_dict[protein] if protein in labels_dict else f"ID: {protein}" for protein in html_proteins_list}
    cleaned_selection_map = {
        protein: str(value).strip("['\"']'").strip(' nan ').replace("'", "") for protein, value in selection_label_map.items()
    }

    # adds brackets to gene nicknames for selection menu
    for key, value in cleaned_selection_map.items():
        if value.startswith('ID:'):
            cleaned_selection_map[key] = value
        else:
            parts = value.split(',', 1)
            if len(parts) == 2:
                cleaned_selection_map[key] = f'{parts[0]} [{parts[1]} ]'
            else:
                cleaned_selection_map[key] = f'{value}'

    # generates node to write into html file
    def generate_node(protein):
        color = "#97c2fc" if protein in other_proteins else "#FF0000"
        label = network_label_map[protein]
        return f'    {{"color": "{color}", "id": "{protein}", "label": "{label}", "shape": "dot", "size": 10}}'
    js_nodes = ",\n".join([generate_node(p) for p in refined_proteins + other_proteins])

    # generates edges to write into the html file
    def generate_edge(protein1, protein2):
        color = "#FF0000"
        return f'   {{"from": "{protein1}", "to": "{protein2}", "width": 1}}'
    js_edges = ",\n".join([generate_edge(p1, p2) for p1, p2 in edge_pairs])

    # updates html file to add nodes, edges, and update the selection menu
    if os.path.exists(file_path):
        with open(file_path, "r", encoding="utf-8") as f:
            html_content = f.read()

        nodes_pattern = re.compile(r"nodes\s*=\s*new\s*vis\.DataSet\(\[(.*?)\]\);", re.DOTALL)
        updated_html = nodes_pattern.sub(f"nodes = new vis.DataSet([\n{js_nodes}\n]);", html_content)

        edges_pattern = re.compile(r"edges\s*=\s*new\s*vis\.DataSet\(\[(.*?)\]\);", re.DOTALL)
        updated_html = edges_pattern.sub(f"edges = new vis.DataSet([\n{js_edges}\n]);", updated_html)

        select_pattern = r'(<select[^>]*id="select-node"[^>]*>)(.*?)(</select>)'
        match = re.search(select_pattern, updated_html, re.DOTALL)
        if match:
            before_options = match.group(1)  
            after_options = match.group(3)  

            new_options = "\n".join([f'    <option value="{node}">{cleaned_selection_map[node]}</option>' for node in other_proteins + refined_proteins])
            new_select_html = before_options + "\n" + new_options + "\n" + after_options
            updated_html = re.sub(select_pattern, new_select_html, updated_html, flags=re.DOTALL)

        with open(file_path, "w", encoding="utf-8") as f:
            f.write(updated_html)
        st.info("You can zoom and scroll through the network diagram. Click on a node to see its name and highlight connected nodes/proteins.")
        st.components.v1.html(updated_html, height=800, scrolling=True)
    
    else:
        st.write("Could not update the HTML file necessary to create the Network Diagram")



def repeat_refinement(llm):
    if 'repeat_refinement_q_widget' not in st.session_state:
        st.session_state['repeat_refinement_q_widget'] = None

    st.subheader("Enter your refining statement:")
    repeat_refine_box = st.container(height=150)
    with repeat_refine_box:
        st.text_input("E.g. 'Only keep genes involved in ALS'",max_chars=501,key='repeat_refinement_q_widget',on_change=submit_text(location="repeat_refinement")) # if maxchars = 500 it thinks its the same text_input as before
        st.write("**Your Most Recent Data Refinement Query:** ",st.session_state.last_refinement_q)
    
    ## repeat-refining agent:
    if st.session_state['user_refinement_q']: # only bother re-creating the dataframe if the user has given new input
        pd_df_agent = create_pandas_dataframe_agent(
            llm=llm,
            df=st.session_state['user_refined_df'],
            agent_type="tool-calling", # can also be others like 'openai-tools' or 'openai-functions'
            verbose=True,
            allow_dangerous_code=True,
            # prefix=additional_prefix,
            # suffix=additional_suffix, # AS SOON AS YOU ADD A SUFFIX IT GETS CONFUSED ABOUT THE ACTUAL COL NAMES. DOES NOT SEEM TO BE IN THE SAME CONTEXT. 
            include_df_in_prompt=True,
            number_of_head_rows=10
        )
        pd_df_agent.handle_parsing_errors = "Check your output and make sure it conforms, use the Action Input/Final Answer syntax"
        full_prompt = f"""
                User refining statement: {st.session_state['user_refinement_q']}
                Instructions: 
                - Given the user refinement statement above and the dataframe you were given, return the pandas expression required to achieve this.
                - Keep in mind some column values may be comma or otherwise delimited and contain multiple values.
                - Return only the code in your reply. The final df should be called df.
                - Do not include any additional formatting, such as markdown code blocks
                - For formatting do not allow any lines of code to exceed 80 columns
                - Example: E.g. you might return df_y = dfx[dfx['blah'] == 'foo']
                """
        response = pd_df_agent.run(full_prompt)
        # st.write(response)
        pandas_code_only = response.split('=', 1)[1] # keep only the pandas expression not the variable assignment
        pandas_code_only = pandas_code_only.replace("df", "st.session_state['user_refined_df']")
        pandas_code_only.replace("```", "").strip() # remove code backticks left over
        # st.write(f"Code to be evaluated:{pandas_code_only}")
        user_refined_df = eval(pandas_code_only)
        st.session_state['user_refined_df'] = user_refined_df
        st.session_state['last_pandas_code'] = pandas_code_only

        # Add to history
        st.session_state.gene_df_history.append((st.session_state['user_refined_df'],st.session_state['last_refinement_q'],st.session_state['last_pandas_code']))
        st.session_state['last_refinement_q'] = st.session_state.gene_df_history[-1][1]
    
    st.subheader("Instructions:")
    st.write("**Press enter to submit a refinement. Repeat as many times as needed.**")
    st.header("Current refined data:")
    with st.expander("**Click** to see most recent filter code",expanded=False):
        st.write(st.session_state['last_pandas_code'])
    # st.write(f"len of gene_df_history: {len(st.session_state.gene_df_history)}")
    # st.write(f"Query at top of history: {st.session_state.gene_df_history[-1][1]}")
    st.dataframe(st.session_state.user_refined_df) # maybe change to point to most recent df on history?
    st.button("**Undo** the last refinement",use_container_width=True,icon=":material/undo:",type="primary",on_click=undo_last_refinement,args=("repeat",))

    st.divider()

def chat_with_data(llm, rag_llm):
    
    PERPLEXITY_API_KEY = st.secrets["PERPLEXITY_API_KEY"]
    if not PERPLEXITY_API_KEY:
        raise ValueError("Perplexity API key is not set. Please set the PERPLEXITY_API_KEY environment variable.")
    perplex_client = OpenAI(api_key=PERPLEXITY_API_KEY, base_url="https://api.perplexity.ai")

    begeneral_prefix = """If the user is asking a specific question that can be answered from the df given to you, do so. Keep in mind some column values may actually be
    delimited lists or contain multiple values. If the user seems to be asking 
    a more general question about a gene, possible associations, biological process, etc. just use your internal knowledge. You can also mix the two sources of info,
    but then be clear where you are getting your information from. Try to keep responses relatively short unless asked for more information or told otherwise.

    Do not simulate data or use only the preview. You are an agent that can code has access to the real dataframe and can simply access it as the variable 'df'.
    """

    alternate_prefix = """You have been provided with a pandas dataframe (named 'df'). Use that and any other knowledge you have internally to answer the following user query:

    Do not simulate data or use only the preview. You are an agent that can code has access to the real dataframe and can simply access it as the variable 'df'.
    """
    alternate_prefix2 = """You have been provided with two pandas dataframes, df1 and df2. df1 contains rows of genes and columns of associated metadata/annotations. df2 contains expression data like logfc, padj, and the disease in the comparison.
    There may or may not be some overlap in the genes between the two.

    Use those dataframes and any other knowledge you have internally to answer the user query that will follow:

    Do not simulate data or use only the preview. You are an agent that can code has access to the real dataframes with the names provided above.
    """
    # Note that you have two dataframes you have access to. One contains genes and annotated info about those genes, and the other contains a user-uploaded gene expression
    # table with genes, logFC, padj, disease, and cell_type. Only use this expression dataframe if the user asks a question that requires it for an answer. When using it, consider only the genes
    # also present in the first dataframe.

    st.header("Chat with your data")
    st.markdown("""Ask questions about the genes/proteins you have narrowed down, general questions about biology, and more. If you have uploaded expression data, you may also use that as part of your queries.\n
**For more complex questions where sources or internet search are desired**, try using Perplexity mode (**Note** that in this mode the agent does not have direct access to your data table. If you are referencing specific genes/proteins, make sure they are in your chat history).""")

    if "messages" not in st.session_state or st.sidebar.button("Clear chat history",use_container_width=True):
        st.session_state["messages"] = [{"role": "system", "content": "Run code or pandas expressions on the dataframes ('df1' and 'df2') given to you to answer the user queries. Assume the user is talking about 'their' genes in df1 unless they are referencing expression."}]
        # NOTE: Can add an extra 'assistant' message here that says Hi/welcome, but that breaks perplexity.
        
    online_search = st.sidebar.toggle("Toggle Perplexity Online Search",help="When ON, uses Perplexity instead of ChatGPT as the base model LLM. Perplexity has realtime access to the internet and can provide real links and sources.")

    
    for msg in st.session_state.messages:
        if msg["role"] != "system":
            st.chat_message(msg["role"]).write(msg["content"])
            # writes the user's progress -S

    if prompt := st.chat_input(placeholder="Ask a question here"):
        # Tack on instructions to the beginning of prompt HERE
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.chat_message("user").write(prompt)

    #     # RUN AN OPENAIEMBEDDINGS RAG CALL AGAINST COLUMN METADATA TO DETERMINE COLUMNS TO USE AND FORMAT CONSIDERATIONS:
    #     column_helper_text = col_retrieval_rag(prompt, rag_llm)
    #     suffix = "Consider using the following information to inform your dataframe operations, if you need to make any: \n" + column_helper_text
    # else:
    #     suffix = "" # agent needs a valid suffix when created even when no user prompt yet

    # Create the agent used for chat retrieval with df access
    non_toolcalling_agent = create_pandas_dataframe_agent(
        llm=llm,
        df=st.session_state['user_refined_df'],
        prefix=begeneral_prefix,
        allow_dangerous_code=True,
        include_df_in_prompt=True,
        number_of_head_rows=15
    )
    # non_toolcalling_agent.handle_parsing_errors = "Check your output and make sure it answers the user query and is a valid JSON object wrapped in triple backticks, use the Action Input/Final Answer syntax"

    temp_df = st.session_state['user_refined_df'].copy()
    pd_df_agent = create_pandas_dataframe_agent( # 'SIMULATES' the data instead of really using the df unless made very clear it has access to df in the prefix
        llm=llm,
        df=[st.session_state['user_refined_df'],st.session_state['expression_df']],
        prefix=alternate_prefix2,
        agent_type="tool-calling",
        allow_dangerous_code=True,
        include_df_in_prompt=True,
        number_of_head_rows=5
        # suffix=suffix
    )
    # pd_df_agent.handle_parsing_errors = True

    # Chat loop/logic
    with st.chat_message("assistant"):
        
        # with st.expander("session_state.messages:",expanded=False):
        #         st.write(st.session_state.messages)
        # st.write(len(st.session_state.messages)) # is 1 before user provides anything

        if len(st.session_state.messages) > 1:
            # st.write(suffix)
            # USE INTERNET/PERPLEXITY IF TOGGLE IS ON
            if not online_search:
                if st.session_state.messages[-1]["role"] == "user": 
                    st_cb = StreamlitCallbackHandler(st.container(), expand_new_thoughts=False,max_thought_containers=5)
                    # try:
                    #     response = non_toolcalling_agent.run(st.session_state.messages, callbacks=[st_cb]) # Has more error loops with certain queries
                    # except:
                    response = pd_df_agent.run(st.session_state.messages, callbacks=[st_cb]) # Still can't access the internet to provide specifics on studies etc.
                    st.session_state.messages.append({"role": "assistant", "content": response})
                    st.write(response)
            else:
                # use perplexity for the response instead
                if st.session_state.messages[-1]["role"] == "user": # Needs user-system alternating, only get response if last message was a user one
                    response = perplex_client.chat.completions.create(model="sonar",messages=st.session_state.messages)
                    response_content = response.choices[0].message.content
                    # Add on the links when actually displaying the response:
                    response_links = response.citations # A list of strings (links)
                    numbered_links = "\n".join(f"{i+1}. {link}" for i, link in enumerate(response_links))
                    final_response = f"{response_content}\n\n{numbered_links}"
                    st.session_state.messages.append({"role": "assistant", "content": final_response})
                    st.write(final_response) # Maybe just change this to 'response' after the April 2025 API changes

    # Put expander with the data at the bottom:
    with st.expander("**Click to view data being referenced**"):
        st.dataframe(st.session_state['user_refined_df'])

    st.divider()

def send_genesdata():
    gene_list = list(st.session_state['user_refined_df']['Gene_Name']) # HARDCODED SO WONT WORK IF USER DF DOESNT HAVE THIS NAME FOR GENE COLUMN
    
    DDB_GENESLIST_API_URL = st.secrets["DDB_GENESLIST_API_URL"]
    
    payload = {
        "values": gene_list  # Only include the values list here
    }
    headers = {"Content-Type": "application/json"}
    
    response = requests.post(f"{DDB_GENESLIST_API_URL}/store_data", json=payload, headers=headers)
    response_json = response.json()
    # st.write(response_json)
    # st.write(response)
    if response.status_code == 200:
        session_id = response_json["session_id"]
        shiny_url = f"https://biominers.net/?session_id={session_id}"
        st.session_state["neurokinex_url"] = shiny_url
        # st.markdown(f"[Go to Shiny App]({shiny_url})")
        st.markdown(f"WORKED: {session_id}")
    else:
        # st.write(response_json)
        st.error("Failed to store data.")

def analyze_data(llm):
    if 'merged_df' not in st.session_state:
         st.session_state['merged_df'] = None

    #create a merged data frame and add to session state
    common_columns = list(set(st.session_state.user_refined_df.columns) & set(st.session_state.genes_info_df.columns))
    defined_merged_df = st.session_state.user_refined_df.merge(st.session_state.genes_info_df, on=common_columns, how="inner")
    defined_merged_df = defined_merged_df.loc[:, ~defined_merged_df.columns.duplicated()]

    st.session_state['merged_df'] = defined_merged_df
    st.session_state['merged_df']['Gene_Name'] = st.session_state['merged_df']['Gene_Name'].astype(str) # Hardcoded to only work for Gene_Name column, but makes sure no floats are left which will later break some visualizations

    st.title("Data Visualization")
    st.subheader("Your genes at a glance:")

    col1, col2 = st.columns(2)

    if 'most_recent_chart_selection' not in st.session_state:
        st.session_state.most_recent_chart_selection = None

    if 'interactive_visualization' not in st.session_state:
        st.session_state.interactive_visualization = None

    with col1:
        if st.button("Neurodegenerative Disease Associations", use_container_width=True):
            st.session_state.interactive_visualization = None
            build_visual_1(llm=llm)
        if st.button("Protein Interactions", use_container_width=True, help="Not recommended for more than 500 genes"):
            st.session_state.most_recent_chart_selection = None
            st.session_state.interactive_visualization = "network"
        if st.button("Transcript Tissue Co-expression",use_container_width=True):
            st.session_state.most_recent_chart_selection = None
            st.session_state.interactive_visualization = "coexpress"

    with col2:
        if st.button("Top 20 Subcellular Locations",use_container_width=True):
            st.session_state.interactive_visualization = None
            build_visual_2(llm=llm)
        if st.button("Primary Structure Overview", use_container_width=True, help="Only displays first 50 proteins in your refined data"):
            st.session_state.most_recent_chart_selection = None
            st.session_state.interactive_visualization = "residues"
        if st.button("Gene Set Enrichment Analysis", use_container_width=True):
            st.session_state.most_recent_chart_selection = None
            st.session_state.interactive_visualization = "gsea"
    if st.button("GWAS + ClinVar Analysis", use_container_width=True, help="Not recommended for more than 50 genes"):
        st.session_state.most_recent_chart_selection = None
        st.session_state.interactive_visualization = "gwas"
    
    # Necessary to put the interactive visualizations in the main panel:
    if st.session_state.interactive_visualization == "network":
        build_visual_3(llm=llm)
    elif st.session_state.interactive_visualization == "residues":
        plot_residues(df=st.session_state.merged_df)
    elif st.session_state.interactive_visualization == "coexpress":
        plot_coexpressed(merged_df = st.session_state.merged_df)
    elif st.session_state.interactive_visualization == "gsea":
        plot_gsea(merged_df = st.session_state.merged_df)
    elif st.session_state.interactive_visualization == "gwas":
        show_gwas_tool(merged_df = st.session_state.merged_df, llm = llm)
    
    # Print most recent saved chart to the screen:
    if st.session_state.most_recent_chart_selection:
        if st.session_state.most_recent_chart_selection == "no_euler": # Occurs when the euler viz func fails because < 2 associated diseases
            st.info("Less than 2 known neurodegenerative disease associations across selected genes.")
        else: 
            st.image(st.session_state.most_recent_chart_selection) # SHOULD MAKE IT SO THAT THIS GETS DELETED IF NEW REFINEMENTS ARE MADE (as it would no longer be accurate)
    
    st.divider()
    
    with st.expander("**Click to view your current gene data**"):
         st.dataframe(st.session_state['merged_df'])
    # clear most recent chart selection when button 3 clicked
    
    send_genes_placeholder = st.empty()
    with send_genes_placeholder:
        if st.button("Send your genes to the BioMiners Tool Suite",use_container_width=True):
            send_genesdata()
            st.link_button(label="View Genes in the Biominers Tool Suite",url=st.session_state["neurokinex_url"],type="primary",use_container_width=True)

    st.divider()





    