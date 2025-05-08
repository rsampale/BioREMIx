import streamlit as st
import requests
import os
import boto3
from boto3.dynamodb.conditions import Key
from decimal import Decimal
import pandas as pd

# CONNECT TO DYNAMODB FOR HPA DATA USED IN SOME VISUALIZATIONS:
AWS = st.secrets["aws"]

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


def fetch_contribs_from_ddb(table, query_gene: str) -> pd.DataFrame:
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