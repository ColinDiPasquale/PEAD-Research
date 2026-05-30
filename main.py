# Post earnings announcement drift prediction using machine learning
# By Colin DiPasquale

import wrds
import pandas as pd
import numpy as np
import os
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.impute import SimpleImputer
import lightgbm as lgb
import matplotlib
matplotlib.use("Agg")  # For saving plots
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

outputDirectory = "./data"
os.makedirs(outputDirectory, exist_ok=True)

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

print("\nMissing values (key columns") # TODO remove?
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
outputPath1 = os.path.join(outputDirectory, "compustatQuarterlyData.parquet")
dfSue.to_parquet(outputPath1, index=False)
print(f"\nSaved: {outputPath1}")

# Create/save short preview
dfSue.head(500).to_csv(os.path.join(outputDirectory, "compustatQuarterlyPreview.csv"), index=False)
print("Saved short preview: data/compustatQuarterlyPreview.csv")

print("\nStep 1 complete") # Just giving an update (step 1 was Computsat data pull/clean)

compustatData = dfSue.copy()
compustatData = compustatData[compustatData["datadate"].dt.year >= 2004].copy() # Dropping 2003 since it was pretty sparse

# --- Daily Data Pull ---

# Var descriptions:
# permno    - CRSP ID
# date      - the date
# ret       - daily return
# retx      - daily return excluding dividends
# prc       - price
# shrout    - shares outstanding (in thousands)
# hexcd     - exchange code header (1 is NYSE, 2 is AMEX, 3 is NASDAQ)
# shrcd     - share code (10 is common stock, 11 is US incorporated)

# Filters:
# INNER JOIN    - keep only the shared rows
# AND d.date BETWEEN...     - Still gets correct data for a company even if it's name changed
# WHERE d.date BETWEEN...   - Makes sure the data is in the range we used previously


print("Pulling CRSP daily returns...")

crspQuery = """
    SELECT
        d.permno,
        d.date,
        d.ret,
        d.retx,
        d.prc,
        d.shrout,
        d.hexcd
    FROM crsp.dsf d
    -- Join to dsenames to filter common stocks (shrcd 10/11 = ordinary common shares)
    INNER JOIN crsp.dsenames n
        ON d.permno = n.permno
        AND d.date BETWEEN n.namedt AND COALESCE(n.nameendt, '2099-12-31')
        AND n.shrcd IN (10, 11)
    WHERE d.date BETWEEN '2003-01-01' AND '2024-06-30'
"""

# --- Cleaning this data ---

crsp = database.raw_sql(crspQuery, date_cols=["date"])
print(f"Pull complete.")

crsp = crsp.dropna(subset=["ret"]) # Drop missing entries
crsp["prc"] = crsp["prc"].abs()  # negative = midpoint quote, still valid, so convert to abs val
crsp = crsp.sort_values(["permno", "date"]).reset_index(drop=True) # Resort

# --- Fama-French Factors Pull ---

# Var descriptions:
# date      - the date
# mktrf     - market excess return
# smb       - return diff between small and large companies
# hml       - return diff between value and growth stocks
# rf        - risk-free rate

# Functions:
# WHERE date BETWEEN...     - getting the data in the timeframe we've been using

print("\nPulling Fama-French daily factors...")
ff_query = """
    SELECT
        date,
        mktrf,
        smb,
        hml,
        rf
    FROM ff.factors_daily
    WHERE date BETWEEN '2003-01-01' AND '2024-06-30'
"""

ff = database.raw_sql(ff_query, date_cols=["date"])
ff = ff.sort_values("date").reset_index(drop=True)
ff[["mktrf", "smb", "hml", "rf"]] = ff[["mktrf", "smb", "hml", "rf"]] / 100 # Converting from percent to decimal
print(f"Pull complete.")

# --- Linking Compustat to CSRP via CUSIP ---
# Clemson doesn't have a subscription for the CCM link table so I have to use this as a workaround

# Var descriptions:
# permno    - CRSP ID
# ncusip    - CUSIP from CRSP
# namedt    - date the name became active
# nameednt  - date the name became inactive
# shrcd     - share code (10 is common stock, 11 is US incorporated)
# hexcd     - exchange code header (1 is NYSE, 2 is AMEX, 3 is NASDAQ)

# Functions:
# WHERE shrcd IN (10, 11)   - ensuring it's a common stock or US incorporated
# ncusip IS NOT NULL        - making sure it has a CUSIP
# ncusip != ''              - making sure the CUSIP isn't blank

print("\nPulling CRSP names for CUSIP link...")
namesQuery = """
    SELECT
        permno,
        ncusip,
        namedt,
        nameendt,
        shrcd,
        exchcd
    FROM crsp.dsenames
    WHERE shrcd IN (10, 11)
      AND ncusip IS NOT NULL
      AND ncusip != ''
"""
names = database.raw_sql(namesQuery, date_cols=["namedt", "nameendt"])
print(f"Pull complete.")

database.close()

names["nameendt"] = names["nameendt"].fillna(pd.Timestamp("2099-12-31")) # Filling in missing nameendt with future date

compustatData["cusip8"] = compustatData["cusip"].str[:8] # Trimming Compustat cusip to 8 digits to match CRSP ncusip
compustatLink = compustatData.merge(names[["permno", "ncusip", "namedt", "nameendt", "shrcd", "exchcd"]],left_on="cusip8",right_on="ncusip",how="left") # Merge on CUSIP

compustatLink = compustatLink[(compustatLink["rdq"] >= compustatLink["namedt"]) &(compustatLink["rdq"] <= compustatLink["nameendt"])].copy() # Only keep rows where rdw is within CRSP name window
compustatLink = compustatLink.drop(columns=["namedt", "nameendt", "ncusip", "cusip8"]) # Drop link columns
compustatLink = compustatLink.drop_duplicates(subset=["gvkey", "datadate"]) # If there's multiple matches just keep the first one

print(f"\nCompustat entries after linking: {len(compustatLink):,}")
print(f"Entries lost: {len(compustatData) - len(compustatLink):,}")


# --- Creating event windows ---
# Day 0 is announcement date
# Days -1 to 0 are announcement window
# Days +1 to +21 are short drift window (~1 month, since market is closed on weekends)
# Days +21 to +63 are medium drift window

print("\nCalculating event windows...")

tradingDays = pd.Series(crsp["date"].unique()) # Get list of all trading days
tradingDays = tradingDays.sort_values().reset_index(drop=True) # Sort the list
tradingDaysIndex = {d: i for i, d in enumerate(tradingDays)}  # Converting trading days to an int

tradingDaysSeries = pd.Series(sorted(crsp["date"].unique()))

allRdq = compustatLink["rdq"].values 
rdqSeries = pd.Series(pd.to_datetime(allRdq)) # Find day 0 for each event

# Converts all the announcement dates into a series of timestamps
uniqueRdq = pd.Series(rdqSeries.unique()).dropna()
rdqToDay0 = {}
for rdq in uniqueRdq:
    rdq = pd.Timestamp(rdq)
    candidates = tradingDaysSeries[tradingDaysSeries >= rdq]
    rdqToDay0[rdq] = candidates.iloc[0] if len(candidates) > 0 else pd.NaT

# For each date, find the nearest trading date on or after (since some companies report on weekends)
compustatLink["day0"] = rdqSeries.map(rdqToDay0).values

def offsetFromDay0(day0Series, offset):
    """Finds the date that the given offset corresponds to"""
    results = []
    for d in day0Series:
        d = pd.Timestamp(d)
        if pd.isna(d) or d not in tradingDaysIndex:
            results.append(pd.NaT)
            continue
        idx = tradingDaysIndex[d] + offset
        results.append(tradingDaysSeries.iloc[idx] if 0 <= idx < len(tradingDaysSeries) else pd.NaT)
    return pd.Series(results, index=day0Series.index)

compustatLink = compustatLink.reset_index(drop=True)
day0Column = pd.Series(compustatLink["day0"])

compustatLink["dayM1"]  = offsetFromDay0(day0Column, -1).values
compustatLink["dayP1"]  = offsetFromDay0(day0Column, +1).values
compustatLink["dayP21"] = offsetFromDay0(day0Column, +21).values
compustatLink["dayP63"] = offsetFromDay0(day0Column, +63).values

print("Event windows calculated.")

# --- Compute CARs ---
# TODO use beta-adjusted returns for longer windows

# Merge FF market return into CRSP
crsp = crsp.merge(ff[["date", "mktrf", "rf"]], on="date", how="left") # Merging FF market returns into CRSP
crsp["marketReturn"] = crsp["mktrf"] + crsp["rf"] # total return
crsp["abnormalReturn"] = crsp["ret"] - crsp["marketReturn"] # abnormal return

crspIndexed = crsp.set_index(["permno", "date"])[["ret", "abnormalReturn"]] # Indexing CRSP just to make it faster

def computeCumulativeAbnormalReturn(permno, startDate, endDate, crspIdx):
    """Computes cumulative abnormal return for the given permno between the start and end date."""
    if pd.isna(permno) or pd.isna(startDate) or pd.isna(endDate):
        return np.nan
    try:
        mask = crspIdx.loc[permno]
        window = mask[(mask.index >= startDate) & (mask.index <= endDate)]
        if len(window) == 0:
            return np.nan
        return (1 + window["abnormalReturn"]).prod() - 1
    except KeyError:
        return np.nan

print("\nComputing CARs...")

# Converting to lists for iteration
permnos   = compustatLink["permno"].values
dayM1    = compustatLink["dayM1"].values
day0      = compustatLink["day0"].values
dayP1    = compustatLink["dayP1"].values
dayP21   = compustatLink["dayP21"].values
dayP63   = compustatLink["dayP63"].values

n = len(compustatLink)
carPostAnnouncement  = np.full(n, np.nan)   # announcement window
carPost21   = np.full(n, np.nan)   # short drift
carPost63   = np.full(n, np.nan)   # medium drift

for i in range(n):
    if i % 50000 == 0:
        print(f"  {i:,} / {n:,}")
    p = permnos[i]
    if pd.isna(p):
        continue
    p = int(p)
    carPostAnnouncement[i] = computeCumulativeAbnormalReturn(p, dayM1[i],  day0[i],   crspIndexed)
    carPost21[i]  = computeCumulativeAbnormalReturn(p, dayP1[i],  dayP21[i], crspIndexed)
    carPost63[i]  = computeCumulativeAbnormalReturn(p, dayP1[i],  dayP63[i], crspIndexed)

compustatLink["carPostAnnouncement"] = carPostAnnouncement
compustatLink["carPost21"]  = carPost21
compustatLink["carPost63"]  = carPost63

print("CARs computed.")

# --- Cleaning ---
eventPanel = compustatLink.copy()

eventPanel = eventPanel.dropna(subset=["carPost21"]) # Remove all missing CAR entries

# Winsorize at 1/99th percentile
# TODO try without winsorizing
for col in ["carPostAnnouncement", "carPost21", "carPost63"]:
    p01 = eventPanel[col].quantile(0.01)
    p99 = eventPanel[col].quantile(0.99)
    eventPanel[col] = eventPanel[col].clip(lower=p01, upper=p99)

print("\nSummary")
print(eventPanel[["carPostAnnouncement", "carPost21", "carPost63"]].describe().round(4))
print(eventPanel.groupby("sueQuintile")[["carPost21", "carPost63"]].mean().round(4)) # Q5 should be higher than Q1

# --- Save output ---
outputPath2 = os.path.join(outputDirectory, "eventPanel.parquet")
eventPanel.to_parquet(outputPath2, index=False)
print(f"\nSaved: {outputPath2}")
print("\nStep 2 complete")

# --- Organizing 6 month momentum data ---

crspMomentum = crsp[["permno", "date", "ret"]].dropna(subset=["ret"]).copy() # Copying the old crsp dataframe is way faster than requerying
crspMomentum = crspMomentum.sort_values(["permno", "date"]).reset_index(drop=True) # Just resorting the new, smaller dataframe

# --- Build the trading day calendar ---
tradingDays = pd.Series(sorted(crspMomentum["date"].unique()))
tradingDaysIndex = {d: i for i, d in enumerate(tradingDays)}

def getOffsetDate(baseDate, offset):
    """Gets the trading date offset days from the base date."""
    baseDate = pd.Timestamp(baseDate)
    if baseDate not in tradingDaysIndex:
        candidates = tradingDays[tradingDays >= baseDate]
        if len(candidates) == 0:
            return pd.NaT
        baseDate = candidates.iloc[0]
    idx = tradingDaysIndex[baseDate] + offset
    if 0 <= idx < len(tradingDays):
        return tradingDays.iloc[idx]
    return pd.NaT

print("\nComputing momentum windows...")

crspMomentumIdx = crspMomentum.set_index(["permno", "date"])["ret"] # Reindexing to make it faster

def computeCumulativeRawReturn(permno, startDate, endDate):
    """Computes cumulative raw return between two dates for a given permno."""
    if pd.isna(permno) or pd.isna(startDate) or pd.isna(endDate):
        return np.nan
    try:
        p = int(permno)
        window = crspMomentumIdx.loc[p]
        window = window[(window.index >= startDate) & (window.index <= endDate)]
        if len(window) < 5:  # require at least 5 trading days
            return np.nan
        return (1 + window).prod() - 1
    except KeyError:
        return np.nan

permnos = eventPanel["permno"].values # Grabbing all the CRSP IDs from the eventPanel dataframe
rdqs = pd.to_datetime(eventPanel["rdq"].values) # Converts all the announcement dates into a clean array

length = len(eventPanel)
momentum6Months = np.full(length, np.nan)
momentum1Month = np.full(length, np.nan)

print(f"Computing momentum for {length:,} events...")
for i in range(length):
    if i % 50000 == 0:
        print(f"{i:,} / {length:,}")
    rdq = rdqs[i]
    p   = permnos[i]
    if pd.isna(rdq) or pd.isna(p):
        continue
    # momentum6Months: -126 to -21 trading days before announcement date (rdq)
    month6Start = getOffsetDate(rdq, -126)
    month6End   = getOffsetDate(rdq, -21)
    # momentum1Month: -21, -2 trading days before announcement date (rdq)
    month1Start = getOffsetDate(rdq, -21)
    month1End   = getOffsetDate(rdq, -2)

    momentum6Months[i] = computeCumulativeRawReturn(p, month6Start, month6End)
    momentum1Month[i] = computeCumulativeRawReturn(p, month1Start, month1End)

eventPanel["momentum6Months"] = momentum6Months
eventPanel["momentum1Month"] = momentum1Month
print("Momentum computed.")

# --- Finding accruals ---

compustatFeatures = compustatData[["gvkey", "datadate", "ibq", "atq", "actq","lctq", "cheq", "epspxq", "fqtr"]].copy()
compustatFeatures = compustatFeatures.sort_values(["gvkey", "datadate"])

compustatFeatures["nwc"] = (compustatFeatures["actq"] - compustatFeatures["cheq"]) - compustatFeatures["lctq"] # Net working capital
compustatFeatures["nwcChange"] = compustatFeatures.groupby("gvkey")["nwc"].diff(1) # Net working capital change
compustatFeatures["avgAssets"] = (compustatFeatures.groupby("gvkey")["atq"].transform(lambda x: (x + x.shift(1)) / 2)) # Average assets
compustatFeatures["accruals"] = compustatFeatures["nwcChange"] / compustatFeatures["avgAssets"] # Accruals

# Getting a rolling 8 quarter average earnings volatility
compustatFeatures["earningsVolatility"] = (compustatFeatures.groupby(["gvkey", "fqtr"])["epspxq"].transform(lambda x: x.rolling(8, min_periods=4).std()))

# Beat streak (consecutive quarters of positive seasonal EPS difference)
compustatFeatures["epsDiff"] = compustatFeatures.groupby(["gvkey", "fqtr"])["epspxq"].diff(1)
compustatFeatures["beat"] = (compustatFeatures["epsDiff"] > 0).astype("Int64")

def rollingStreak(s):
    """Counts consecutive 1s ending at each position. N/A is 0"""
    streak = []
    count = 0
    for v in s:
        if pd.isna(v) or v != 1:
            count = 0
        else:
            count += 1
        streak.append(count)
    return streak

compustatFeatures["beatStreak"] = (compustatFeatures.groupby("gvkey")["beat"].transform(lambda x: pd.Series(rollingStreak(x.values), index=x.index)))

# Keeping only whats necessary (helped me stay organized)
accrualColumns = ["gvkey", "datadate", "accruals", "earningsVolatility", "beatStreak"]
compustatAccruals = compustatFeatures[accrualColumns].drop_duplicates(subset=["gvkey", "datadate"])

# --- Merging every feature into 1 dataframe ---
print("\nAssembling feature matrix...")

completeDF = eventPanel.copy() # Start with main event panel
completeDF = completeDF.merge(compustatAccruals, on=["gvkey", "datadate"], how="left") # Add in accrual features
completeDF["ep"] = completeDF["ibq"] / completeDF["mktcap"] # Get earnings to price
completeDF["logMarketCap"] = np.log(completeDF["mktcap"].clip(lower=0.01)) # Get log of market cap
completeDF["absSue"] = completeDF["winsorizedSUE"].abs() # Get abs SUE

# Clean up infinite values from ratio features
for col in ["btm", "ep", "leverage", "accruals"]:
    completeDF[col] = completeDF[col].replace([np.inf, -np.inf], np.nan)

# --- Making a dataframe that has just what's needed for the ML section ---

featureColumns = [
    # IDs/Dates
    "gvkey", "permno", "datadate", "rdq", "fyearq", "fqtr",

    # Post announcement cumulative abnormal returns (CAR)
    "carPostAnnouncement", "carPost21", "carPost63",

    # Earnings surprise
    "winsorizedSUE", "absSue",

    # Size & liquidity
    "logMarketCap", "logAssets", "exchcd",

    # Valuation
    "btm", "ep", "leverage",

    # Earnings quality
    "accruals", "earningsVolatility", "beatStreak",

    # Momentum
    "momentum6Months", "momentum1Month",

    # SUE quintile
    "sueQuintile",
]

ml = completeDF[featureColumns].copy()
ml = ml.dropna(subset=["carPost21", "winsorizedSUE", "logMarketCap", "btm"]) # Double check the super important fields are always present
# ml = ml.dropna() # Commenting this out for now, not sure if this would drop too much. TODO go back and change/remove based on the print statements

print("ML dataframe created.")

# --- Output checking and saving ---

print("\nMissing values by feature")
print(ml[featureColumns[6:]].isnull().sum().sort_values(ascending=False))

# Just printing everything to double check it looks good (besides the ids obviously)
print("\nFeature summary statistics")
featuresOnly = [c for c in featureColumns if c not in
             ["gvkey", "permno", "datadate", "rdq", "fyearq", "fqtr", "sueQuintile"]]
print(ml[featuresOnly].describe().round(4).T[["mean", "std", "min", "max"]])

print("\nMean carPost21 by SUE quintile")
print(ml.groupby("sueQuintile")["carPost21"].mean().round(4))

outputPath3 = os.path.join(outputDirectory, "MLFeatures.parquet")
ml.to_parquet(outputPath3, index=False)
print(f"\nSaved: {outputPath3}")
print("Step 3 complete")

# --- Winsorizing outliers ---

ml = ml.sort_values("datadate").reset_index(drop=True) # Sorting chronologically for walk forward
winsorizedColumns = ["btm", "ep", "leverage", "accruals", "earningsVolatility", "momentum6Months", "momentum1Month", "winsorizedSUE", "absSue"]

print("\nWinsorizing outliers...")
for col in winsorizedColumns:
    if col not in ml.columns:
        continue
    p01 = ml[col].quantile(0.01)
    p99 = ml[col].quantile(0.99)
    ml[col] = ml[col].clip(lower=p01, upper=p99)
    print(f"  {col:20s}: [{p01:.4f}, {p99:.4f}]")

ml["exchcd"] = ml["exchcd"].where(ml["exchcd"].isin([1, 2, 3]), other=np.nan) # Exchange was broken so I just made anything that wasn't 1, 2, or 3 as NaN

# --- Setting up the ML features ---

FEATURES = [
    "winsorizedSUE",        # standardized unexpected earnings (SUE) winsorized
    "absSue",               # surprise magnitude
    "logMarketCap",         # firm size (log market cap)
    "logAssets",            # firm size (log total assets)
    "exchcd",               # exchange listing
    "btm",                  # book to market ratio
    "ep",                   # earnings to price ratio
    "leverage",             # financial leverage
    "accruals",             # earnings quality (Accruals)
    "earningsVolatility",   # earnings stability
    "beatStreak",           # consecutive beat history
    "momentum6Months",      # 6 month pre announcement momentum
    "momentum1Month",       # 1 month pre announcement momentum
]
TARGET = "carPost21"           # 1 month post earnings cumulative abnormal return

xAll = ml[FEATURES].copy() # Input matrix
yAll = ml[TARGET].copy() # Output matrix
dates = ml["datadate"].copy() # Dates so we can keep track during walk-forward

# --- Walk forward validation ---

years = sorted(ml["datadate"].dt.year.unique()) # Sorts by year
minTrainYears = 5 # The # of years the model will have to train off of before making predictions
testYears = [y for y in years if y >= years[0] + minTrainYears] # The matrix of test years

print(f"\nWalk-forward validation: {len(testYears)} test years ({testYears[0]}–{testYears[-1]})")

# Ongoing loop results (for each of the 15 runs)
olsPreds   = []
lgbmPreds  = []
actuals     = []
testDates  = []
testIndices = []

imputer = SimpleImputer(strategy="median") # Initialized here but will fill the values in between columns
scaler  = StandardScaler() # Initialized here but will standardize each feature to have a mean of 0

# --- Setting up LightGBM parameters ---

lgbmParameters = {
    "objective"       : "regression",
    "metric"          : "rmse",           # root mean squared error
    "n_estimators"    : 500,
    "learning_rate"   : 0.02,
    "num_leaves"      : 31,               # controls tree complexity
    "min_child_samples": 50,              # minimum observations per leaf
    "subsample"       : 0.8,              # row subsampling (reduces overfitting)
    "colsample_bytree": 0.8,             # feature subsampling
    "reg_alpha"       : 0.1,             # L1 regularization
    "reg_lambda"      : 0.1,             # L2 regularization
    "random_state"    : 42,
    "verbose"         : -1,              # suppress LightGBM output
    "n_jobs"          : -1,
}

featureImportances = np.zeros(len(FEATURES))

# --- Creating model ---

for i, testYear in enumerate(testYears):
    trainMask = dates.dt.year < testYear # Selecting all rows from before the given year
    testMask  = dates.dt.year == testYear # Selects all rows from after that year

    # Splits that data into x and y
    xTrain = xAll[trainMask].copy()
    yTrain = yAll[trainMask].copy()
    xTest  = xAll[testMask].copy()
    yTest  = yAll[testMask].copy()

    if len(xTest) == 0:
        continue

    # Now using the imputer/scalar from before
    xTrainImpute = imputer.fit_transform(xTrain)
    xtestImpute  = imputer.transform(xTest)
    xTrain_sc = scaler.fit_transform(xTrainImpute)
    xTest_sc  = scaler.transform(xtestImpute)

    # OLS (Ridge regression which is just linear regression that tries to avoid overfitting)
    ols = Ridge(alpha=1.0)
    ols.fit(xTrain_sc, yTrain)
    olsPred = ols.predict(xTest_sc) # Making our predictions

    # LGBM (the callbacks just speeds things up a bit by not creating the whole tree if it can't improve more)
    lgbm = lgb.LGBMRegressor(**lgbmParameters)
    lgbm.fit(xTrainImpute, yTrain, eval_set=[(xtestImpute, yTest)], callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(period=-1)])
    lgbmPredictions = lgbm.predict(xtestImpute) # Making the actual predictions

    # Adding feature importances to the running total
    featureImportances += lgbm.feature_importances_

    # Appending the ongoing predictions
    olsPreds.extend(olsPred)
    lgbmPreds.extend(lgbmPredictions)
    actuals.extend(yTest.values)
    testDates.extend(dates[testMask].values)
    testIndices.extend(xAll[testMask].index.tolist())

    if (i + 1) % 5 == 0 or i == 0: # Just so I know it's still actually making progress
        print(f"Completed test year {testYear} (train n={trainMask.sum():,}, test n={testMask.sum():,})")

# Average feature importances across loops
featureImportances /= len(testYears)

# --- Evaluating the models ---

# Putting everything in arrays
actuals = np.array(actuals)
olsPreds = np.array(olsPreds)
lgbmPreds = np.array(lgbmPreds)

olsR2 = r2_score(actuals, olsPreds) # R^2 scores (generally they will be super low, like .01 is still considered very good)
lgbmR2 = r2_score(actuals, lgbmPreds)
olsRMSE = np.sqrt(mean_squared_error(actuals, olsPreds)) # Getting RMSE
lgbmRMSE= np.sqrt(mean_squared_error(actuals, lgbmPreds))

print("\nModel Performance:")
print(f"\n{'Model':<20} {'R²':>10} {'RMSE':>10}")
print(f"{'Ridge OLS':<20} {olsR2:>10.4f} {olsRMSE:>10.4f}")
print(f"{'LightGBM':<20} {lgbmR2:>10.4f} {lgbmRMSE:>10.4f}")

print("\nNote: R^2 is normally super low (.01 to .03).")

# --- Seeing if the model actually predict drift direction ---

resultsDF = pd.DataFrame({
    "actual"            : actuals,
    "olsPred"           : olsPreds,
    "lgbmPredictions"   : lgbmPreds,
    "date"              : testDates,
})

# Sorts both the lgbm and ols results by quintile
resultsDF["lgbmQuintile"] = pd.qcut(resultsDF["lgbmPredictions"], q=5, labels=[1, 2, 3, 4, 5])
resultsDF["olsQuintile"] = pd.qcut(resultsDF["olsPred"], q=5, labels=[1, 2, 3, 4, 5])

print("\nActual car21 by LightGBM predicted quintile")
print("(Q5 = model predicts strongest positive drift)")
lgbmSorted = resultsDF.groupby("lgbmQuintile")["actual"].mean().round(4) 
print(lgbmSorted)

print("\nActual car21 by OLS predicted quintile")
olsSorted = resultsDF.groupby("olsQuintile")["actual"].mean().round(4)
print(olsSorted)

lgbmSpread = lgbmSorted.iloc[-1] - lgbmSorted.iloc[0] # Just getting the spreads
olsSpread  = olsSorted.iloc[-1] - olsSorted.iloc[0]
print(f"\nLightGBM Q5-Q1 spread: {lgbmSpread:.4f} ({lgbmSpread*100:.2f}%)")
print(f"OLS Q5-Q1 spread: {olsSpread:.4f} ({olsSpread*100:.2f}%)")

# --- Feature Importances ---

importanceDF = pd.DataFrame({
    "feature"   : FEATURES,
    "importance": featureImportances,
}).sort_values("importance", ascending=False).reset_index(drop=True)

print("\nLightGBM Feature Importances")
print(importanceDF.to_string(index=False))

# --- Plots ---

plotDirectory = "./plots"
os.makedirs(plotDirectory, exist_ok=True)


# Plot 1: Feature Importances
fig, ax = plt.subplots(figsize=(9, 6))
colors = ["#2196F3" if i < 5 else "#90CAF9" for i in range(len(importanceDF))] # Just typed in a random hex code then used the little popup to pick something I liked
ax.barh(importanceDF["feature"][::-1], importanceDF["importance"][::-1], color=colors[::-1])
ax.set_xlabel("Average Gain (Feature Importance)")
ax.set_title("LightGBM Feature Importances\nPredicting 1 Month Post-Earnings Drift (car21)")
ax.set_ylabel("")
plt.tight_layout()
plt.savefig(os.path.join(plotDirectory, "featureImportance.png"), dpi=150)
plt.close()
print(f"\nSaved: plots/featureImportance.png")

# Plot 2: Portfolio sort (actual car21 by predicted quintile)
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
for ax, sort, title in zip(axes, [lgbmSorted, olsSorted], ["LightGBM Predicted Quintile", "OLS Predicted Quintile"]):
    bars = ax.bar(sort.index.astype(str), sort.values * 100, color=["#EF5350", "#EF9A9A", "#B0BEC5", "#A5D6A7", "#4CAF50"]) # Did the same thing with the hex codes here
    ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax.set_xlabel("Predicted Drift Quintile (1 = lowest, 5 = highest)")
    ax.set_ylabel("Actual car21 (%)")
    ax.set_title(f"Actual 1 Month Drift by\n{title}")
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f%%"))
plt.tight_layout()
plt.savefig(os.path.join(plotDirectory, "portfolioSort.png"), dpi=150)
plt.close()
print("Saved: plots/portfolioSort.png")

# Plot 3: Annual R^2 over time
resultsDF["year"] = pd.to_datetime(resultsDF["date"]).dt.year
annualR2 = resultsDF.groupby("year").apply(lambda g: r2_score(g["actual"], g["lgbmPredictions"]) if len(g) > 10 else np.nan).dropna()

fig, ax = plt.subplots(figsize=(10, 4))
ax.bar(annualR2.index, annualR2.values, color="#42A5F5", alpha=0.8) # TODO pick better colors
ax.axhline(0, color="black", linewidth=0.8)
ax.axhline(lgbmR2, color="red", linewidth=1.2, linestyle="--", label=f"Overall R²={lgbmR2:.4f}")
ax.set_xlabel("Year")
ax.set_ylabel("Out of Sample R²")
ax.set_title("LightGBM Annual Out-of-Sample R²\n(Predicting 1 Month Post-Earnings Drift)")
ax.legend()
plt.tight_layout()
plt.savefig(os.path.join(plotDirectory, "annualR2.png"), dpi=150)
plt.close()
print("Saved: plots/annualR2.png")

# --- Saving results ---
resultsDF.to_parquet(os.path.join(outputDirectory, "modelPredictions.parquet"), index=False)
importanceDF.to_csv(os.path.join(outputDirectory, "featureImportances.csv"), index=False)

print("\n--- Summary ---\n")
print(f"Out of sample observations: {len(actuals):,}")
print(f"Test period: {pd.to_datetime(testDates).min().date()} → {pd.to_datetime(testDates).max().date()}")
print(f"LightGBM R²: {lgbmR2:.4f}")
print(f"LightGBM Q5-Q1 spread: {lgbmSpread*100:.2f}%")
print(f"Top feature: {importanceDF.iloc[0]['feature']}")
print("\nStep 4 complete")