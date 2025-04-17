from datamodel import OrderDepth, TradingState, Order, ConversionObservation
from typing import Dict, List, Tuple
import jsonpickle
import numpy as np

class Product:
    MACARONS = "MAGNIFICENT_MACARONS"

# === TUNE THESE PARAMETERS ONLY ===
PARAMS = {
    Product.MACARONS: {
        "init_make_edge": 2.0,          # opening quote edge ( S )
        "step_size": 0.5,               # edge adjustment quantum ( ΔS )
        "min_edge": 0.5,                # hard floor for the edge
        "volume_bar": 75,               # full‑size fill threshold
        "dec_edge_discount": 0.80,      # edge‑reduction profitability factor
        "volume_avg_window": 5,         # how many stamps to average over
        "make_probability": 0.566,      # aggression factor for quoting inside edge
        "take_edge": 0.75,              # width for aggressive taking
    }
}

POSITION_LIMITS = {Product.MACARONS: 75}
CONVERSION_LIMIT = 10  # absolute value, per tick

# -----------------------------------------------------------------------------
# Utility helpers
# -----------------------------------------------------------------------------

def clamp(value: int, low: int, high: int) -> int:
    return max(low, min(high, value))

def implied_bid_ask(obs: ConversionObservation) -> Tuple[float, float]:
    """Fair bid/ask from the foreign market after fees & tariffs."""
    bid = obs.bidPrice - obs.exportTariff - obs.transportFees - 0.1  # small buffer
    ask = obs.askPrice + obs.importTariff + obs.transportFees
    return bid, ask

# -----------------------------------------------------------------------------
# Core trader class
# -----------------------------------------------------------------------------

class Trader:
    def __init__(self, params: Dict = None):
        self.params = params or PARAMS

    # ---------------------------------------------------------
    # Adaptive edge logic (simple inventory‑/fill‑based agent)
    # ---------------------------------------------------------
    def _adapt_edge(self, timestamp: int, tdata: Dict, pos: int) -> float:
        conf = self.params[Product.MACARONS]
        cache = tdata.setdefault(Product.MACARONS, {
            "edge": conf["init_make_edge"],
            "fills": [] ,
            "optimised": False,
        })

        # collect realised fills (proxy: |position change|)
        cache["fills"].append(abs(pos))
        if len(cache["fills"]) > conf["volume_avg_window"]:
            cache["fills"].pop(0)

        # need enough data before tuning
        if len(cache["fills"]) < conf["volume_avg_window"] or cache["optimised"]:
            return cache["edge"]

        avg_fill = np.mean(cache["fills"])   # average abs inventory‑move
        curr_edge = cache["edge"]

        # --- raise edge if we keep filling full size ---
        if avg_fill >= conf["volume_bar"]:
            curr_edge += conf["step_size"]
            cache["fills"].clear()
        # --- lower edge if we could earn more with tighter spread ---
        elif (
            conf["dec_edge_discount"] * conf["volume_bar"] * (curr_edge - conf["step_size"])
            > avg_fill * curr_edge
        ):
            curr_edge = max(conf["min_edge"], curr_edge - conf["step_size"])
            cache["fills"].clear()
            cache["optimised"] = True

        cache["edge"] = curr_edge
        return curr_edge

    # ---------------------------------------------------------
    # Aggressive take / market‑making logic
    # ---------------------------------------------------------
    def _quote_macarons(
        self,
        depth: OrderDepth,
        obs: ConversionObservation,
        pos: int,
        edge: float,
    ) -> List[Order]:
        conf = self.params[Product.MACARONS]
        orders: List[Order] = []
        limit = POSITION_LIMITS[Product.MACARONS]

        implied_bid, implied_ask = implied_bid_ask(obs)
        take_width = conf["take_edge"]

        # 1) Aggressive taking if book crosses implied fair value by > take_width
        # -------------------------------------------------------------------
        if depth.sell_orders:
            best_ask = min(depth.sell_orders)
            ask_vol = -depth.sell_orders[best_ask]
            if best_ask < implied_bid - take_width and pos < limit:
                qty = clamp(ask_vol, 0, limit - pos)
                if qty:
                    orders.append(Order(Product.MACARONS, best_ask, qty))
                    pos += qty
        if depth.buy_orders:
            best_bid = max(depth.buy_orders)
            bid_vol = depth.buy_orders[best_bid]
            if best_bid > implied_ask + take_width and pos > -limit:
                qty = clamp(bid_vol, 0, limit + pos)
                if qty:
                    orders.append(Order(Product.MACARONS, best_bid, -qty))
                    pos -= qty

        # 2) Passive market‑making quotes around implied fair
        # -------------------------------------------------------------------
        passive_bid = round(implied_bid - edge)
        passive_ask = round(implied_ask + edge)

        # Clip passive quote size so that combined with existing inventory we stay inside limits
        bid_qty = clamp(limit - pos, 0, conf["volume_bar"])
        ask_qty = clamp(limit + pos, 0, conf["volume_bar"])

        if bid_qty:
            orders.append(Order(Product.MACARONS, passive_bid, bid_qty))
        if ask_qty:
            orders.append(Order(Product.MACARONS, passive_ask, -ask_qty))

        return orders

    # ---------------------------------------------------------
    # Main entry
    # ---------------------------------------------------------
    def run(self, state: TradingState):
        tdata: Dict = {}
        if state.traderData:
            tdata = jsonpickle.decode(state.traderData)

        results: Dict[str, List[Order]] = {}
        conversions: int = 0

        if (
            Product.MACARONS in state.order_depths and
            Product.MACARONS in state.observations.conversionObservations
        ):
            
            depth = state.order_depths[Product.MACARONS]
            obs = state.observations.conversionObservations[Product.MACARONS]
            pos = state.position.get(Product.MACARONS, 0)

            edge = self._adapt_edge(state.timestamp, tdata, pos)
            orders = self._quote_macarons(depth, obs, pos, edge)
            results[Product.MACARONS] = orders

            print(orders)

            # Simple inventory‑flattening conversion request (respect ±10 limit)
            conversions = clamp(-pos, -CONVERSION_LIMIT, CONVERSION_LIMIT)

        return results, conversions, jsonpickle.encode(tdata)
