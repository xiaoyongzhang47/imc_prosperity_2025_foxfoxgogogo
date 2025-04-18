from datamodel import OrderDepth, TradingState, Order, ConversionObservation
from typing import Dict, List, Tuple
import jsonpickle
import numpy as np


class Product:
    MACARONS = "MAGNIFICENT_MACARONS"


# === YOUR ORIGINAL PARAMS PLUS FIXED‐μ/σ FOR MR ===
PARAMS = {
    Product.MACARONS: {
        "init_make_edge": 2.0,
        "step_size": 0.5,
        "min_edge": 0.5,
        "volume_bar": 15,
        "dec_edge_discount": 0.80,
        "volume_avg_window": 10,
        "take_edge": 0.75,
        # fixed historical gap stats:
        "mr_gap_mean": 6.36117,
        "mr_gap_std": 0.6,
        "mr_band_k": 1.0,
        "mr_trade_qty": 10,
    }
}

POSITION_LIMITS = {Product.MACARONS: 75}
CONVERSION_LIMIT = 10  # unchanged, max convert ±10 each tick


def clamp(value: int, low: int, high: int) -> int:
    return max(low, min(high, value))


def implied_bid_ask(obs: ConversionObservation) -> Tuple[float, float]:
    bid = obs.bidPrice - obs.exportTariff - obs.transportFees - 0.1
    ask = obs.askPrice + obs.importTariff + obs.transportFees
    return bid, ask


class Trader:
    def __init__(self, params: Dict = None):
        self.params = params or PARAMS

    def _adapt_edge(self, tdata: Dict, pos: int) -> float:
        conf = self.params[Product.MACARONS]
        
        cache = tdata.setdefault(
            Product.MACARONS,
            {"edge": conf["init_make_edge"], "fills": [], "optimised": False},
        )
        cache["fills"].append(abs(pos))

       
        if len(cache["fills"]) > conf["volume_avg_window"]:
            cache["fills"].pop(0)
        if len(cache["fills"]) < conf["volume_avg_window"] or cache["optimised"]:
            return cache["edge"]
        
        print(cache["fills"])
        
        avg_fill = np.mean(cache["fills"])
        edge = cache["edge"]
        if avg_fill >= conf["volume_bar"]:
            edge += conf["step_size"]
            cache["fills"].clear()
        elif (
            conf["dec_edge_discount"]
            * conf["volume_bar"]
            * (edge - conf["step_size"])
            > avg_fill * edge
        ):
            edge = max(conf["min_edge"], edge - conf["step_size"])
            cache["fills"].clear()
            cache["optimised"] = True

        cache["edge"] = edge
        return edge

    def _quote_mm(
        self,
        depth: OrderDepth,
        obs: ConversionObservation,
        pos: int,
        edge: float,
    ) -> List[Order]:
        conf = self.params[Product.MACARONS]
        limit = POSITION_LIMITS[Product.MACARONS]
        orders: List[Order] = []

        implied_bid, implied_ask = implied_bid_ask(obs)
        take_width = conf["take_edge"]

        # aggressive take
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

        # passive quotes
        passive_bid = round(implied_bid - edge)
        passive_ask = round(implied_ask + edge)
        bid_qty = clamp(limit - pos, 0, conf["volume_bar"])
        ask_qty = clamp(limit + pos, 0, conf["volume_bar"])
        if bid_qty:
            orders.append(Order(Product.MACARONS, passive_bid, bid_qty))
        if ask_qty:
            orders.append(Order(Product.MACARONS, passive_ask, -ask_qty))

        return orders

    def _quote_mean_reversion(
        self,
        depth: OrderDepth,
        implied_mid: float,
        pos: int,
    ) -> List[Order]:
        conf = self.params[Product.MACARONS]
        orders: List[Order] = []
        limit = POSITION_LIMITS[Product.MACARONS]

        if not depth.buy_orders or not depth.sell_orders:
            return orders

        market_mid = (max(depth.buy_orders) + min(depth.sell_orders)) / 2
        gap = market_mid - implied_mid

        mean_gap = conf["mr_gap_mean"]
        std_gap = conf["mr_gap_std"] or 1.0
        upper = mean_gap + conf["mr_band_k"] * std_gap
        lower = mean_gap - conf["mr_band_k"] * std_gap
        qty = conf["mr_trade_qty"]

        if gap > upper and pos > -limit:
            trade_qty = clamp(qty, 0, limit + pos)
            if trade_qty:
                orders.append(
                    Order(Product.MACARONS, max(depth.buy_orders), -trade_qty)
                )
                pos -= trade_qty
        elif gap < lower and pos < limit:
            trade_qty = clamp(qty, 0, limit - pos)
            if trade_qty:
                orders.append(
                    Order(Product.MACARONS, min(depth.sell_orders), trade_qty)
                )
                pos += trade_qty

        return orders

    def run(self, state: TradingState):
        tdata: Dict = jsonpickle.decode(state.traderData) if state.traderData else {}
        results: Dict[str, List[Order]] = {}
        conversions: int = 0

        if (
            Product.MACARONS in state.order_depths
            and Product.MACARONS in state.observations.conversionObservations
        ):
            depth = state.order_depths[Product.MACARONS]
            obs = state.observations.conversionObservations[Product.MACARONS]
            pos = state.position.get(Product.MACARONS, 0)

            

            edge = self._adapt_edge(tdata, pos)

            
            implied_bid, implied_ask = implied_bid_ask(obs)
            implied_mid = (implied_bid + implied_ask) / 2

            orders: Order =[]

            orders += self._quote_mm(depth, obs, pos, edge)

            orders += self._quote_mean_reversion(depth, implied_mid, pos)



            results[Product.MACARONS] = orders
            # **unchanged**: flatten remaining position via conversion ≤ ±10
            conversions = clamp(abs(pos), 0, CONVERSION_LIMIT)

        return results, conversions, jsonpickle.encode(tdata)