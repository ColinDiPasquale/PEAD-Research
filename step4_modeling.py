# Step 4 - Running LGBM/OLS and creating graphs

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
from config import DATA_DIR, PLOT_DIR

def runStep4():
    ml = pd.read_parquet(os.path.join(DATA_DIR, "MLFeatures.parquet"))

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

    # Plot 1: Feature Importances
    fig, ax = plt.subplots(figsize=(9, 6))
    colors = ["#2196F3" if i < 5 else "#90CAF9" for i in range(len(importanceDF))] # Just typed in a random hex code then used the little popup to pick something I liked
    ax.barh(importanceDF["feature"][::-1], importanceDF["importance"][::-1], color=colors[::-1])
    ax.set_xlabel("Average Gain (Feature Importance)")
    ax.set_title("LightGBM Feature Importances\nPredicting 1 Month Post-Earnings Drift (car21)")
    ax.set_ylabel("")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, "featureImportance.png"), dpi=150)
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
    plt.savefig(os.path.join(PLOT_DIR, "portfolioSort.png"), dpi=150)
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
    plt.savefig(os.path.join(PLOT_DIR, "annualR2.png"), dpi=150)
    plt.close()
    print("Saved: plots/annualR2.png")

    # --- Saving results ---
    resultsDF.to_parquet(os.path.join(DATA_DIR, "modelPredictions.parquet"), index=False)
    importanceDF.to_csv(os.path.join(DATA_DIR, "featureImportances.csv"), index=False)

    print("\n--- Summary ---\n")
    print(f"Out of sample observations: {len(actuals):,}")
    print(f"Test period: {pd.to_datetime(testDates).min().date()} → {pd.to_datetime(testDates).max().date()}")
    print(f"LightGBM R²: {lgbmR2:.4f}")
    print(f"LightGBM Q5-Q1 spread: {lgbmSpread*100:.2f}%")
    print(f"Top feature: {importanceDF.iloc[0]['feature']}")
    print("\nStep 4 complete")