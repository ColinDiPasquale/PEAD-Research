# Step 1 - Getting data from Compustat

import wrds
import pandas as pd
import numpy as np
import os
from config import DATA_DIR

def runStep1():
    print("Connecting...")
    database = wrds.Connection() # database object
    print("Connected.\n")

    # --- Data pull ---

    # Var descriptions:
    # gvkey       - company ID
    # cusip       - financial ID
    # tic         - ticker
    # conm        - company name
    # datedate    - date of last day of quarter
    # rdq         - earnings announcement date
    # fyearq      - report year
    # fqtr        - report quarter (1–4)

    # epspxq      - basic EPS excluding extraordinary items
    # epspiq      - diluted EPS excluding extraordinary items

    # revtq       - total revenue
    # niq         - quarterly net income
    # ibq         - income before extraordinary items
    # saleq       - net sales

    # atq         - total assets
    # ceqq        - book value
    # ltq         - total liabilities
    # actq        - current assets
    # lctq        - current liabilities
    # cheq        - cash

    # prccq       - price at end of quarter
    # cshoq       - outstanding shares
    # dvpsxq      - dividends per share
    # exchg       - exchange

    # Filters:
    # indfmt='INDL'  - industrial format
    # datafmt='STD'  - standardized
    # popsrc='D'     - domestic
    # consol='C'     - consolidated statements
    # datedate BETWEEN '2000-01-01' AND '2023-12-31'  - reports between those dates
    # epspxq IS NOT NULL  - ensure there is an EPS
    # AND rdq IS NOT NULL - ensure there is a date

    query = """
        SELECT
            gvkey,
            cusip,
            tic,
            conm,
            datadate,
            rdq,
            fyearq,
            fqtr,

            epspxq,
            epspiq,

            revtq,
            niq,
            ibq,
            saleq,

            atq,
            ceqq,
            ltq,
            actq,
            lctq,
            cheq,

            prccq,
            cshoq,
            dvpsxq,

            exchg

        FROM comp.fundq

        WHERE indfmt  = 'INDL'
        AND datafmt = 'STD'
        AND popsrc  = 'D'
        AND consol  = 'C'
        AND datadate BETWEEN '2000-01-01' AND '2023-12-31'
        AND epspxq IS NOT NULL
        AND rdq    IS NOT NULL
    """

    print("Pulling Compustat data...")
    rawData = database.raw_sql(query, date_cols=["datadate", "rdq"])
    print(f"Pull complete.")
    print(rawData.head())

    # --- Cleaning ---

    df = rawData.copy()
    df = df.sort_values(["gvkey", "datadate"]).reset_index(drop=True) # sort for time series
    df = df.drop_duplicates(subset=["gvkey", "datadate"], keep="last") # remove any duplicates
    df = df[(df["atq"] > 0) & (df["prccq"] > 0) & (df["cshoq"] > 0)] # require positive assets and non-missing price
    df["mktcap"] = df["prccq"] * df["cshoq"] # get market cap (in millions)
    df["btm"] = df["ceqq"] / df["mktcap"] # get book to market ratio
    df["leverage"] = df["ltq"] / df["atq"] # get leverage
    df["logAssets"] = np.log(df["atq"]) # get log of assets

    print(f"\nDone cleaning.")

    # --- Finding SUE via seasonal random walk ---

    df = df.sort_values(["gvkey", "datadate"]).reset_index(drop=True)
    df["epsDiff"] = df.groupby(["gvkey", "fqtr"])["epspxq"].diff(1) # current EPS - same quarter last year
    df["stdEpsDiff"] = (df.groupby(["gvkey", "fqtr"])["epsDiff"].transform(lambda x: x.rolling(8, min_periods=4).std())) # Rolling 8 quarter stddev of those differences
    df["stdEpsDiffFloor"] = df["stdEpsDiff"].clip(lower=0.01) # Floor denom at 0.01 to avoid div by 0
    df["sue"] = df["epsDiff"] / df["stdEpsDiffFloor"] # Find SUE
    dfSue = df.dropna(subset=["sue"]).copy() # Drop anywhere that SUE didn't work (mainly the first 3 years)

    print(f"\nCalculated SUE.")

    # Winsorizing SUE at 1st / 99th percentile
    p01 = dfSue["sue"].quantile(0.01)
    p99 = dfSue["sue"].quantile(0.99)
    dfSue["winsorizedSUE"] = dfSue["sue"].clip(lower=p01, upper=p99)

    # --- Validation Checks ---
    print("\nSummary\n")

    print(f"Total firm-quarters     : {len(dfSue):,}")
    print(f"Unique firms (gvkey)      : {dfSue['gvkey'].nunique():,}")
    print(f"Date range                : {dfSue['datadate'].min().date()} → {dfSue['datadate'].max().date()}")
    print(f"Quarters with rdq present : {dfSue['rdq'].notna().sum():,}")

    print(f"\nSUE distribution: {dfSue['winsorizedSUE'].describe().round(4)}")

    print("\nMissing values") # TODO remove?
    keyCols = ["rdq", "epspxq", "atq", "ceqq", "prccq", "cshoq", "sue", "mktcap", "btm"]
    print(dfSue[keyCols].isnull().sum())

    print("\nAnnual observation count")
    print(dfSue.groupby(dfSue['datadate'].dt.year).size().to_string())

    # SUE quintile distribution
    print("\nSUE quintile distribution")
    dfSue["sueQuintile"] = pd.qcut(dfSue["winsorizedSUE"], q=5, labels=[1, 2, 3, 4, 5])
    print(dfSue["sueQuintile"].value_counts().sort_index())
    # TODO: try without winsorizing

    # --- Saving output ---
    outputPath1 = os.path.join(DATA_DIR, "compustatQuarterlyData.parquet")
    dfSue.to_parquet(outputPath1, index=False)
    print(f"\nSaved: {outputPath1}")

    # Create/save short preview
    dfSue.head(500).to_csv(os.path.join(DATA_DIR, "compustatQuarterlyPreview.csv"), index=False)
    print("Saved short preview: data/compustatQuarterlyPreview.csv")

    print("\nStep 1 complete")