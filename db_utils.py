import streamlit as st
import requests
import os
import boto3
import duckdb
from boto3.dynamodb.conditions import Key
from decimal import Decimal
import pandas as pd

# CONNECT TO DYNAMODB FOR HPA DATA USED IN SOME VISUALIZATIONS:
AWS = st.secrets["aws"]

# AWS S3 CONNECTION FACTORY
def _build_duckdb_connection() -> duckdb.DuckDBPyConnection:
    """
    One-time factory.  Sets up a DuckDB connection that can read
    from S3 using credentials in st.secrets
    """

    con = duckdb.connect()
    con.execute(f"SET s3_access_key_id='{AWS['access_key_id']}';")
    con.execute(f"SET s3_secret_access_key='{AWS['secret_key']}';")
    con.execute("INSTALL httpfs; LOAD httpfs;")
    con.execute(f"SET s3_region='{AWS['region_name']}';")
    return con
# Use cache_resrouce to save ~150ms and not re-connect multiple times per user and session
@st.cache_resource(show_spinner=False)
def get_con() -> duckdb.DuckDBPyConnection:
    return _build_duckdb_connection()

def get_clinvar_variant_info(gene_symbol: str, *, assembly: str | None = "GRCh38"):
    """
    Return **all** columns from the variant Parquet for a given gene.
    Optional `assembly` lets you keep only GRCh38 or GRCh37 rows.
    """
    gene_symbol = gene_symbol.upper().strip()
    BUCKET = AWS.get("clinvar_bucket_name", "clinvar-bucket")
    VARIANT_PATH = f"s3://{BUCKET}/clinvar/variant_summary.slim.parquet"
    GENE_PATH    = f"s3://{BUCKET}/clinvar/gene_specific_summary.slim.parquet"
    
    con = get_con()

    query = f"""
        SELECT *
        FROM read_parquet(?)
        WHERE GeneSymbol = ?
        { 'AND Assembly = ?' if assembly else '' }
    """

    params = [VARIANT_PATH, gene_symbol]
    if assembly:
        params.append(assembly)

    return con.execute(query, params).df()

def get_gwas_variant_info(gene_symbol: str):
    gene_symbol = gene_symbol.upper().strip()
    if not gene_symbol:
        return pd.DataFrame()

    VARIANT_PATH = "s3://gwas-data/gwas/gwas_associations.parquet"
    con = get_con()

    query = r"""
        SELECT *
        FROM read_parquet(?)
        WHERE regexp_matches(
              "MAPPED_GENE",
              '(?i)\b' || ? || '\b'          -- one backslash on each side
        )
    """
    return con.execute(query, [VARIANT_PATH, gene_symbol]).df()

def get_dynamodb_table():
    session = boto3.Session(
        aws_access_key_id     = AWS["access_key_id"],
        aws_secret_access_key = AWS["secret_key"],
        region_name           = AWS["region_name"]
    )
    dynamodb = session.resource("dynamodb")
    table    = dynamodb.Table(AWS["table_name"])
    
    return table

def fetch_gene_list_from_ddb(table) -> list[str]:
    """
    Scan only the partition keys (query_gene) and return a sorted list.
    """
    genes = []
    scan_kwargs = {"ProjectionExpression": "query_gene"}
    while True:
        resp = table.scan(**scan_kwargs)
        genes += [item["query_gene"] for item in resp.get("Items", [])]
        if "LastEvaluatedKey" in resp:
            scan_kwargs["ExclusiveStartKey"] = resp["LastEvaluatedKey"]
        else:
            break
        
    return sorted(genes)


def fetch_contribs_from_ddb_legacy(table, query_gene: str) -> pd.DataFrame:
    """
    Get the single item for query_gene and unpack the 'partners' list into
    a DataFrame (index=tissue, columns=partner, values=contrib_ti).
    (a single item contains a partner list and all tissue contributions)
    
    In the output tuple, also return a series of the pearson_r values, indexed by partner gene
    """
    resp = table.get_item(Key={"query_gene": query_gene})
    partners = resp.get("Item", {}).get("partners", [])
    if not partners:
        return pd.DataFrame(), pd.Series(dtype=float)  # no data available

    # build a wide DF: rows=tissues, cols=partners
    df = pd.DataFrame({
        p["partner"]: p["contrib_map"]
        for p in partners
    }).T
    df.index.name = "tissue"
    
    # build the pearson_r series
    pearson_r = pd.Series(
        { p["partner"]: float(p["pearson_r"]) for p in partners },
        name="pearson_r"
    )
    
    return df, pearson_r

@st.cache_data
def get_coexpression_data(
    gene_symbol: str,
    k: int = 10,
    *,
    bucket: str = "hpa-tool-data",
    refined_genes: list[str] | None = None,
) -> pd.DataFrame:
    """
    Returns a DataFrame with rows = [ query gene + its top-k partners ],
    columns = ['gene','pearson_r', tissue_1 … tissue_50], ordered:
      • query gene first (pearson_r NULL)
      • then partners sorted by descending pearson_r
    If `refined_genes` is provided, only those partners will be considered.
    """
    gene = gene_symbol.upper().strip()
    con = get_con()

    corr_path = f"s3://{bucket}/hpa/correlations/gene={gene}/*.parquet"
    expr_path = f"s3://{bucket}/hpa/expression/expression.parquet"

    # build an optional AND-clause (this is if we are using the refined only filtering mode)
    if refined_genes:
        # escape and quote each gene
        quoted = ", ".join(f"'{g.upper().strip()}'" for g in refined_genes)
        filter_clause = f"AND partner IN ({quoted})"
    else:
        filter_clause = ""

    sql = f"""
    WITH top_corr AS (
      SELECT partner, pearson_r
      FROM read_parquet('{corr_path}')
      WHERE gene = '{gene}'
        {filter_clause}
      ORDER BY pearson_r DESC
      LIMIT {k}
    ),
    expr AS (
      SELECT *
      FROM read_parquet('{expr_path}')
      WHERE gene = '{gene}'
         OR gene IN (SELECT partner FROM top_corr)
    )
    SELECT
      expr.gene                           AS gene,
      CASE WHEN expr.gene = '{gene}' THEN NULL
           ELSE top_corr.pearson_r
      END                                 AS pearson_r,
      expr.*                             -- 50 tissue cols
    FROM expr
    LEFT JOIN top_corr
      ON expr.gene = top_corr.partner
    ORDER BY
      CASE WHEN expr.gene = '{gene}' THEN 0 ELSE 1 END,
      pearson_r DESC
    ;
    """

    return con.execute(sql).df()

@st.cache_data(show_spinner=False)
def fetch_enrichr_libs(lib_url : str) -> list[str]:
    try:
        resp = requests.get(lib_url, timeout=30)
        resp.raise_for_status()                    # raises HTTPError on 4xx / 5xx
        stats = resp.json()
        return [d["libraryName"] for d in stats["statistics"]]

    # --- handle specific failure modes ---
    except requests.exceptions.HTTPError as e:      # bad status code
        st.error(f"Server returned {e.response.status_code} – {e.response.reason}")
        return None
    except requests.exceptions.Timeout:
        st.error("Enrichr API timed out.  Try again in a moment.")
        return None
    except requests.exceptions.RequestException as e:  # catch‑all
        st.error(f"Network error: {e}")
        return None