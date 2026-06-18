# Step 2 - Getting data from CRSP and building CARs

import wrds
import pandas as pd
import numpy as np
import os
from config import DATA_DIR

def runStep2():
    print("Connecting...")
    database = wrds.Connection() # database object
    print("Connected.\n")

    compustatData = pd.read_parquet(os.path.join(DATA_DIR, "compustatQuarterlyData.parquet"))
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
    outputPath2 = os.path.join(DATA_DIR, "eventPanel.parquet")
    eventPanel.to_parquet(outputPath2, index=False)
    outputPath3 = os.path.join(DATA_DIR, "crsp.parquet")
    crsp.to_parquet(outputPath3, index=False)
    print(f"\nSaved: {outputPath2}")
    print(f"\nSaved: {outputPath3}")
    print("\nStep 2 complete")