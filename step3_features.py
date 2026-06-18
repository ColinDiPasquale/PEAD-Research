# Step 3 - Creating features and getting ready for the ML process

import pandas as pd
import numpy as np
import os
from config import DATA_DIR

def runStep3():
    eventPanel = pd.read_parquet(os.path.join(DATA_DIR, "eventPanel.parquet"))
    crsp = pd.read_parquet(os.path.join(DATA_DIR, "crsp.parquet"))
    compustatData = pd.read_parquet(os.path.join(DATA_DIR, "compustatQuarterlyData.parquet"))

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

    outputPath3 = os.path.join(DATA_DIR, "MLFeatures.parquet")
    ml.to_parquet(outputPath3, index=False)
    print(f"\nSaved: {outputPath3}")
    print("Step 3 complete")