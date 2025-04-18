import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List

# --------------------------------------------------------------
#  Local project imports (provided by the IMC boiler‑plate)
# --------------------------------------------------------------
from datamodel import (
    Listing,
    OrderDepth,
    TradingState,
    Order,
    ConversionObservation,
    Observation,
)
from macarons_only_v2 import Trader, Product  # the agent you built earlier

# --------------------------------------------------------------
#  Helper constructors
# --------------------------------------------------------------

def build_order_depth(row: pd.Series) -> OrderDepth:
    """Create an OrderDepth snapshot from one row of prices_round CSV."""
    depth = OrderDepth()

    # Bid levels (volumes are + in datamodel)
    for level in range(1, 4):
        p_key = f"bid_price_{level}"
        v_key = f"bid_volume_{level}"
        price = row[p_key]
        vol = row[v_key]
        if not np.isnan(price) and not np.isnan(vol):
            depth.buy_orders[int(price)] = int(vol)

    # Ask levels (convert to –ve volume as per datamodel)
    for level in range(1, 4):
        p_key = f"ask_price_{level}"
        v_key = f"ask_volume_{level}"
        price = row[p_key]
        vol = row[v_key]
        if not np.isnan(price) and not np.isnan(vol):
            depth.sell_orders[int(price)] = -int(vol)

    return depth


def build_conversion_obs(row: pd.Series) -> ConversionObservation:
    return ConversionObservation(
        bidPrice=row["bidPrice"],
        askPrice=row["askPrice"],
        transportFees=row["transportFees"],
        exportTariff=row["exportTariff"],
        importTariff=row["importTariff"],
        sugarPrice=row["sugarPrice"],
        sunlightIndex=row["sunlightIndex"],
    )


# --------------------------------------------------------------
#  Back‑test runner
# --------------------------------------------------------------

def backtest_day(
    prices_csv: Path,
    observations_csv: Path,
    start_cash: float = 0.0,
) -> float:
    """Run a simple mark‑to‑mid back‑test for the Magnificent Macarons trader."""

    # ---------------- Load data ---------------
    prices = pd.read_csv(prices_csv, sep=";")
    obs_df = pd.read_csv(observations_csv)

    # Keep only the macaron rows to speed things up
    prices = prices[prices["product"] == "MAGNIFICENT_MACARONS"]

    # Groups by timestamp for quick lookup
    price_groups = prices.groupby("timestamp")
    obs_groups = obs_df.groupby("timestamp")
    timestamps = sorted(set(price_groups.indices) & set(obs_groups.indices))

    # ---------------- Objects & state ----------
    trader = Trader()
    trader_data: str = ""

    cash: float = start_cash
    position: Dict[str, int] = {"MAGNIFICENT_MACARONS": 0}

    # A single listing is enough for the Trader – denomination hard‑coded
    listings = {"MAGNIFICENT_MACARONS": Listing("MAGNIFICENT_MACARONS", "MAGNIFICENT_MACARONS", "SEASHELLS")}

    for ts in timestamps:
        # ------- Build OrderBook & observations for this tick -------
        book_row = price_groups.get_group(ts).iloc[0]
        depth = {"MAGNIFICENT_MACARONS": build_order_depth(book_row)}

        conv_row = obs_groups.get_group(ts).iloc[0]
        conv_obs = {"MAGNIFICENT_MACARONS": build_conversion_obs(conv_row)}
        observation = Observation(plainValueObservations={}, conversionObservations=conv_obs)

        state = TradingState(
            traderData=trader_data,
            timestamp=ts,
            listings=listings,
            order_depths=depth,
            own_trades={},
            market_trades={},
            position=position.copy(),
            observations=observation,
        )

        orders_dict, conversion_req, trader_data = trader.run(state)

        # -------- Simulate conversion at mid‑price (up to ±10) --------
        if conversion_req:
            conv_mid = (conv_obs["MAGNIFICENT_MACARONS"].bidPrice + conv_obs["MAGNIFICENT_MACARONS"].askPrice) / 2
            cash -= conversion_req * conv_mid
            position["MAGNIFICENT_MACARONS"] += conversion_req

        # -------- Naïve fills against top of book --------------------
        ob = depth["MAGNIFICENT_MACARONS"]
        best_bid = max(ob.buy_orders) if ob.buy_orders else None
        best_ask = min(ob.sell_orders) if ob.sell_orders else None

        for order in orders_dict.get("MAGNIFICENT_MACARONS", []):
            filled = False
            if order.quantity > 0 and best_ask is not None and order.price >= best_ask:
                filled = True
                fill_price = best_ask
            elif order.quantity < 0 and best_bid is not None and order.price <= best_bid:
                filled = True
                fill_price = best_bid

            if filled:
                qty = order.quantity
                cash -= qty * fill_price  # buy: negative cash; sell: positive cash (qty is signed)
                position["MAGNIFICENT_MACARONS"] += qty

        # ---- Periodic PnL snapshot (optional) ----
        mid_price = book_row["mid_price"]
        pnl = cash + position["MAGNIFICENT_MACARONS"] * mid_price

    print("==============================")
    print(f"Final inventory : {position["MAGNIFICENT_MACARONS"]} units")
    print(f"End‑of‑day PnL  : {pnl:.2f} shells")
    return pnl


if __name__ == "__main__":
    import sys

    ROOT = Path("./round-4-island-data-bottle")  # adjust to match your folder structure

    day = sys.argv[1]

    backtest_day(
        prices_csv=ROOT / f"prices_round_4_day_{day}.csv",
        observations_csv=ROOT / f"observations_round_4_day_{day}.csv",
    )
