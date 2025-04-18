from datamodel import OrderDepth, TradingState, Order, ConversionObservation
from typing import Dict, List, Tuple
import jsonpickle
import numpy as np

import time

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
        "take_edge": 0.25,               # width for aggressive taking


    }
}

POSITION_LIMITS = {Product.MACARONS: 70}
CONVERSION_LIMIT = 10  # absolute value, per tick

# Utility helpers
def clamp(value: int, low: int, high: int) -> int:
    return max(low, min(high, value))

def implied_bid_ask(obs: ConversionObservation) -> Tuple[float, float]:
    """Fair bid/ask from the foreign market after fees & tariffs."""
    bid = obs.bidPrice - obs.exportTariff - obs.transportFees - 0.1
    ask = obs.askPrice + obs.importTariff + obs.transportFees
    return bid, ask

# Core trader class
class Trader:
    def __init__(self, params: Dict = None):
        self.params = params or PARAMS

    # ---------------------------------------------------------
    # Adaptive edge logic (simple inventory‑/fill‑based agent)
    # ---------------------------------------------------------
    def _adapt_edge(self, tdata: Dict, pos: int) -> float:
        conf = self.params[Product.MACARONS]
        cache = tdata.setdefault(Product.MACARONS, {
            "edge": conf["init_make_edge"],
            "fills": [],
            "optimised": False,
        })

        # collect realised fills (proxy: |position change|)
        cache["fills"].append(abs(pos))
        if len(cache["fills"]) > conf["volume_avg_window"]:
            cache["fills"].pop(0)



        # need enough data before tuning
        if len(cache["fills"]) < conf["volume_avg_window"] or cache["optimised"]:
            return cache["edge"]

        avg_fill = np.mean(cache["fills"])
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

    # ---------------------------------------------------------
    #  Arbitrage‑first quoting logic
    # ---------------------------------------------------------
    def _quote_macarons(
        self,
        depth: OrderDepth,
        obs:   ConversionObservation,
        pos:   int,
        edge:  float,
    ) -> List[Order]:

        conf   = self.params[Product.MACARONS]
        limit  = POSITION_LIMITS[Product.MACARONS]
        orders: List[Order] = []

        # ---------- fair values ----------
        implied_bid, implied_ask = implied_bid_ask(obs)
        implied_mid              = 0.5 * (implied_bid + implied_ask)

        book_best_bid = max(depth.buy_orders)  if depth.buy_orders else None
        book_best_ask = min(depth.sell_orders) if depth.sell_orders else None
        if book_best_bid is None or book_best_ask is None:
            return orders                                       # empty book → nothing to do
        
        book_mid = 0.5 * (book_best_bid + book_best_ask)

        # ---------- conversion‑arbitrage thresholds ----------
        arb_th     = conf.get("arb_threshold", 1.75)            # ≃ the 1.75‑peak you saw
        take_size  = conf.get("arb_take_qty", 25)               # how much to lift / hit each tick

        diff = implied_mid - book_mid                           # >0 ⇒ local cheap vs foreign

        # Optionally save deviation history (for debugging or offline analysis)
        # if not hasattr(self, 'save_list'):
        #     self.save_list = []
                
        # self.save_list.append(diff)
        
        # np.savetxt('diff.csv', self.save_list, delimiter=',', fmt='%f')

        
        
        # --- Case ①  local market is under‑priced → BUY locally, convert/export, then flatten
        if diff > arb_th and pos < limit:
            ask_vol = -depth.sell_orders[book_best_ask]
            qty     = clamp(min(take_size, ask_vol), 0, limit - pos)
            if qty:
                orders.append(Order(Product.MACARONS, book_best_ask,  qty))
                pos += qty                                      # optimistic fill

        # --- Case ②  local market is over‑priced → SELL locally, convert/import, then flatten
        elif diff < -arb_th and pos > -limit:
            bid_vol =  depth.buy_orders[book_best_bid]
            qty     = clamp(min(take_size, bid_vol), 0, limit + pos)
            if qty:
                orders.append(Order(Product.MACARONS, book_best_bid, -qty))
                pos -= qty

        # ---------- passive maker of last resort ----------
        # If no arb trade executed above, post a tight two‑sided quote for flow capture
        if not orders:
            passive_bid = round(implied_bid - edge)
            passive_ask = round(implied_ask + edge)

            bid_qty = clamp(limit - pos, 0, conf["volume_bar"])
            ask_qty = clamp(limit + pos, 0, conf["volume_bar"])

            if bid_qty:
                orders.append(Order(Product.MACARONS, passive_bid,  bid_qty))
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



            edge = self._adapt_edge(tdata, pos)
            orders = self._quote_macarons(depth, obs, pos, edge)
            results[Product.MACARONS] = orders

            # Simple inventory‑flattening conversion request (respect ±10 limit)
            conversions = clamp(-pos, -CONVERSION_LIMIT, CONVERSION_LIMIT)

        return results, conversions, jsonpickle.encode(tdata)
