from datamodel import OrderDepth, TradingState, Order, ConversionObservation
from typing import Dict, List, Tuple
import jsonpickle
import numpy as np

import time 


class Product:
    MACARONS = "MAGNIFICENT_MACARONS"


# === PARAMETERS ===
PARAMS = {
    Product.MACARONS: {
        # ARB params
        "p_exec": 0.2,            # estimated execution probability
        "arb_trade_qty": 25,      # units per arbitrage signal
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

    def _quote_arbitrage(
        self,
        depth: OrderDepth,
        obs: ConversionObservation,
        pos: int,
    ) -> List[Order]:
        conf = self.params[Product.MACARONS]
        orders: List[Order] = []
        limit = POSITION_LIMITS[Product.MACARONS]

        # local best prices
        best_ask_local = min(depth.sell_orders) if depth.sell_orders else None
        best_bid_local = max(depth.buy_orders) if depth.buy_orders else None

        f_bid = obs.bidPrice
        f_ask = obs.askPrice

        
        if best_ask_local is not None:
            profit = (f_bid - best_ask_local - obs.transportFees - obs.exportTariff)
    
            if profit > 0 and pos < limit:

                qty = clamp(conf["arb_trade_qty"], 0, limit - pos)

                if qty:
                    orders.append(
                        Order(Product.MACARONS, best_ask_local, qty)
                    )

                    pos += qty


        if best_bid_local is not None:

            profit = (best_bid_local - f_ask - obs.transportFees - obs.importTariff)

            if profit > 0 and pos > -limit:

               
                qty = clamp(conf["arb_trade_qty"], 0, limit + pos)
                if qty:
                    orders.append(
                        Order(Product.MACARONS, best_bid_local, -qty)
                    )
                    pos -= qty
        
        
        return orders

    # Main entry: arbitrage → MM → MR → conversion
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

            conversions = clamp(-pos, -CONVERSION_LIMIT, CONVERSION_LIMIT)

            # arbitrage
            orders = self._quote_arbitrage(depth, obs, pos)

            # print(pos)

            
            
            results[Product.MACARONS] = orders
            # leave the rest to conversion (±10 per tick)
            

        return results, conversions, jsonpickle.encode(tdata)