"""Generate synthetic time series data resembling Kaggle Store Sales.

Creates:
- train.csv: daily sales for N stores × M items with seasonality, trend, promo effects
- stores.csv: store metadata (city, state, type, cluster)
- items.csv: item metadata (family, class, perishable)
- oil.csv: simulated oil prices
- holidays_events.csv: simulated holiday events
- test.csv: test period data
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config import RAW_DATA_DIR, RANDOM_STATE

rng = np.random.default_rng(RANDOM_STATE)


def generate_oil_prices(dates: pd.DatetimeIndex) -> pd.DataFrame:
    """Generate synthetic oil prices with trend and noise."""
    n = len(dates)
    trend = np.linspace(100, 50, n)
    seasonal = 10 * np.sin(2 * np.pi * np.arange(n) / 365)
    noise = rng.normal(0, 3, n)
    prices = trend + seasonal + noise
    prices = np.clip(prices, 20, 120)
    return pd.DataFrame({"date": dates, "dcoilwtico": prices})


def generate_holidays(dates: pd.DatetimeIndex) -> pd.DataFrame:
    """Generate synthetic holiday events."""
    holidays = []
    for year in range(dates.min().year, dates.max().year + 1):
        # Fixed holidays
        holidays.extend([
            (f"{year}-01-01", "New Year", "National"),
            (f"{year}-12-25", "Christmas", "National"),
        ])
        # Random holidays
        n_random = rng.integers(3, 8)
        for _ in range(n_random):
            random_date = pd.Timestamp(f"{year}-01-01") + pd.Timedelta(days=int(rng.integers(0, 365)))
            holidays.append((random_date.strftime("%Y-%m-%d"), f"Holiday_{rng.integers(1, 20)}", "Local"))

    df = pd.DataFrame(holidays, columns=["date", "type", "locale"])
    df["date"] = pd.to_datetime(df["date"])
    df["transferred"] = False
    df = df[df["date"].isin(dates)].drop_duplicates("date")
    return df


def generate_stores(n_stores: int = 20) -> pd.DataFrame:
    """Generate store metadata."""
    cities = ["Quito", "Guayaquil", "Cuenca", "Ambato", "Manta"]
    states = ["Pichincha", "Guayas", "Azuay", "Tungurahua", "Manabi"]
    types = ["A", "B", "C", "D"]

    df = pd.DataFrame({
        "store_nbr": range(1, n_stores + 1),
        "city": rng.choice(cities, n_stores),
        "state": rng.choice(states, n_stores),
        "type": rng.choice(types, n_stores, p=[0.4, 0.3, 0.2, 0.1]),
        "cluster": rng.integers(1, 10, n_stores),
    })
    return df


def generate_items(n_items: int = 50) -> pd.DataFrame:
    """Generate item metadata."""
    families = [
        "GROCERY", "BEVERAGES", "PRODUCE", "DAIRY", "FROZEN",
        "CLEANING", "BREAD", "MEAT", "PERSONAL CARE", "DELI",
    ]

    df = pd.DataFrame({
        "item_nbr": range(1, n_items + 1),
        "family": rng.choice(families, n_items),
        "class": rng.integers(1000, 9999, n_items),
        "perishable": rng.choice([0, 1], n_items, p=[0.75, 0.25]),
    })
    return df


def generate_sales(
    dates: pd.DatetimeIndex,
    stores: pd.DataFrame,
    items: pd.DataFrame,
    oil: pd.DataFrame,
    holidays: pd.DataFrame,
) -> pd.DataFrame:
    """Generate synthetic daily sales with seasonality, trend, and promo effects."""
    records = []
    oil_map = oil.set_index("date")["dcoilwtico"].to_dict()
    holiday_dates = set(holidays["date"])

    for store_nbr in stores["store_nbr"]:
        store_type = stores[stores["store_nbr"] == store_nbr]["type"].values[0]
        store_multiplier = {"A": 1.5, "B": 1.0, "C": 0.7, "D": 0.4}[store_type]

        for item_nbr in items["item_nbr"]:
            family = items[items["item_nbr"] == item_nbr]["family"].values[0]
            family_multiplier = {
                "GROCERY": 1.2, "BEVERAGES": 1.0, "PRODUCE": 1.3, "DAIRY": 0.9,
                "FROZEN": 0.7, "CLEANING": 0.8, "BREAD": 1.1, "MEAT": 0.9,
                "PERSONAL CARE": 0.6, "DELI": 0.8,
            }.get(family, 1.0)

            for date in dates:
                dayofweek = date.dayofweek
                month = date.month
                is_holiday = date in holiday_dates

                # Base sales
                base = 50 * store_multiplier * family_multiplier

                # Weekly seasonality (weekends higher)
                weekly = 1.3 if dayofweek >= 5 else 1.0

                # Monthly seasonality
                monthly = 1.0 + 0.2 * np.sin(2 * np.pi * month / 12)

                # Holiday boost
                holiday_boost = 1.5 if is_holiday else 1.0

                # Oil price effect (inverse relationship)
                oil_price = oil_map.get(date, 70)
                oil_effect = 1.0 + (100 - oil_price) / 200

                # Trend
                days_since_start = (date - dates.min()).days
                trend = 1.0 + days_since_start / 1000

                # Promotion (random 10% of days)
                onpromo = rng.random() < 0.1
                promo_effect = 1.3 if onpromo else 1.0

                # Noise
                noise = rng.lognormal(0, 0.3)

                sales = base * weekly * monthly * holiday_boost * oil_effect * trend * promo_effect * noise
                sales = max(0, int(sales))

                records.append({
                    "date": date.strftime("%Y-%m-%d"),
                    "store_nbr": store_nbr,
                    "item_nbr": item_nbr,
                    "sales": sales,
                    "onpromotion": int(onpromo),
                })

    return pd.DataFrame(records)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_stores", type=int, default=20)
    parser.add_argument("--n_items", type=int, default=50)
    parser.add_argument("--start_date", default="2013-01-01")
    parser.add_argument("--end_date", default="2017-08-15")
    parser.add_argument("--test_start", default="2017-08-16")
    parser.add_argument("--test_end", default="2017-08-31")
    args = parser.parse_args()

    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)

    dates = pd.date_range(args.start_date, args.end_date, freq="D")
    test_dates = pd.date_range(args.test_start, args.test_end, freq="D")
    all_dates = pd.date_range(args.start_date, args.test_end, freq="D")

    print(f"Generating data for {args.n_stores} stores, {args.n_items} items ...")
    print(f"Train period: {dates.min().date()} ~ {dates.max().date()}")
    print(f"Test period:  {test_dates.min().date()} ~ {test_dates.max().date()}")

    # Metadata
    stores = generate_stores(args.n_stores)
    items = generate_items(args.n_items)
    stores.to_csv(RAW_DATA_DIR / "stores.csv", index=False)
    items.to_csv(RAW_DATA_DIR / "items.csv", index=False)
    print(f"Stores: {len(stores)} | Items: {len(items)}")

    # Oil
    oil = generate_oil_prices(all_dates)
    oil.to_csv(RAW_DATA_DIR / "oil.csv", index=False)
    print(f"Oil prices: {len(oil)} days")

    # Holidays
    holidays = generate_holidays(all_dates)
    holidays.to_csv(RAW_DATA_DIR / "holidays_events.csv", index=False)
    print(f"Holidays: {len(holidays)} events")

    # Sales
    print("Generating sales data (this may take a moment) ...")
    train_sales = generate_sales(dates, stores, items, oil[oil["date"].isin(dates)], holidays)
    train_sales.to_csv(RAW_DATA_DIR / "train.csv", index=False)
    print(f"Train sales: {len(train_sales):,} rows")

    # Test data (same structure, sales=0 or NaN)
    test_sales = generate_sales(test_dates, stores, items, oil[oil["date"].isin(test_dates)], holidays)
    test_sales["sales"] = np.nan
    test_sales.to_csv(RAW_DATA_DIR / "test.csv", index=False)
    print(f"Test sales: {len(test_sales):,} rows")

    print(f"\nAll files saved to {RAW_DATA_DIR}")


if __name__ == "__main__":
    main()
