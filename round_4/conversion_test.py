from datamodel import OrderDepth, TradingState, Order, ConversionObservation
from typing import Dict, List, Tuple
import jsonpickle
import numpy as np
import random 


class Product:
    MACARONS = "MAGNIFICENT_MACARONS"


# === PARAMETERS ===
PARAMS = {
    Product.MACARONS: {
        # MM + MR params
        "init_make_edge": 2.0,
        "step_size": 0.5,
        "min_edge": 0.5,
        "volume_bar": 75,
        "dec_edge_discount": 0.80,
        "volume_avg_window": 5,
        "take_edge": 0.75,
        
        # fixed MR stats (historical)
        "mr_gap_mean": 6.6,
        "mr_gap_std": 0.6,
        "mr_band_k": 1.0,
        "mr_trade_qty": 25,

        # ARB params
        # "shipping_cost": 1.0,     # per-unit shipping
        "p_exec": 0.3,            # estimated execution probability
        "arb_trade_qty": 20,      # units per arbitrage signal
    }
}

POSITION_LIMITS = {Product.MACARONS: 75}
CONVERSION_LIMIT = 10  # max ±10 conversion per tick


def clamp(value: int, low: int, high: int) -> int:
    return max(low, min(high, value))


def implied_bid_ask(obs: ConversionObservation) -> Tuple[float, float]:
    bid = obs.bidPrice - obs.exportTariff - obs.transportFees - 0.1
    ask = obs.askPrice + obs.importTariff + obs.transportFees
    return bid, ask


class Trader:
    def __init__(self, params: Dict = None):
        self.params = params or PARAMS

    def run(self, state: TradingState):
        tdata: Dict = jsonpickle.decode(state.traderData) if state.traderData else {}
        results: Dict[str, List[Order]] = {}
        conversions: int = 0
        pos = 0

        if not tdata.get("bought"):
            depth = state.order_depths[Product.MACARONS]
            best_ask = min(depth.sell_orders)
            best_bid = min(depth.buy_orders)
            if random.random() < 0.5:
                # random buy
                qty = 1
                results[Product.MACARONS] = [Order(Product.MACARONS, best_ask, qty)]
            else:
                # random sell
                qty = -1
                results[Product.MACARONS] = [Order(Product.MACARONS, best_bid, qty)]

            pos += qty

        else:

            pos = state.position.get(Product.MACARONS, 0)
            conversions = clamp(-pos, -CONVERSION_LIMIT, CONVERSION_LIMIT)

        return results, conversions, jsonpickle.encode(tdata)