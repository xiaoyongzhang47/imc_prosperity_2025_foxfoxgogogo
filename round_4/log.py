import json
from typing import Any

from datamodel import Listing, Observation, Order, OrderDepth, ProsperityEncoder, Symbol, Trade, TradingState



class Logger:
    def __init__(self) -> None:
        self.logs = ""
        self.max_log_length = 3750

    def print(self, *objects: Any, sep: str = " ", end: str = "\n") -> None:
        self.logs += sep.join(map(str, objects)) + end

    def flush(self, state: TradingState, orders: dict[Symbol, list[Order]], conversions: int, trader_data: str) -> None:
        base_length = len(
            self.to_json(
                [
                    self.compress_state(state, ""),
                    self.compress_orders(orders),
                    conversions,
                    "",
                    "",
                ]
            )
        )

        # We truncate state.traderData, trader_data, and self.logs to the same max. length to fit the log limit
        max_item_length = (self.max_log_length - base_length) // 3

        print(
            self.to_json(
                [
                    self.compress_state(state, self.truncate(state.traderData, max_item_length)),
                    self.compress_orders(orders),
                    conversions,
                    self.truncate(trader_data, max_item_length),
                    self.truncate(self.logs, max_item_length),
                ]
            )
        )

        self.logs = ""

    def compress_state(self, state: TradingState, trader_data: str) -> list[Any]:
        return [
            state.timestamp,
            trader_data,
            self.compress_listings(state.listings),
            self.compress_order_depths(state.order_depths),
            self.compress_trades(state.own_trades),
            self.compress_trades(state.market_trades),
            state.position,
            self.compress_observations(state.observations),
        ]

    def compress_listings(self, listings: dict[Symbol, Listing]) -> list[list[Any]]:
        compressed = []
        for listing in listings.values():
            compressed.append([listing.symbol, listing.product, listing.denomination])

        return compressed

    def compress_order_depths(self, order_depths: dict[Symbol, OrderDepth]) -> dict[Symbol, list[Any]]:
        compressed = {}
        for symbol, order_depth in order_depths.items():
            compressed[symbol] = [order_depth.buy_orders, order_depth.sell_orders]

        return compressed

    def compress_trades(self, trades: dict[Symbol, list[Trade]]) -> list[list[Any]]:
        compressed = []
        for arr in trades.values():
            for trade in arr:
                compressed.append(
                    [
                        trade.symbol,
                        trade.price,
                        trade.quantity,
                        trade.buyer,
                        trade.seller,
                        trade.timestamp,
                    ]
                )

        return compressed

    def compress_observations(self, observations: Observation) -> list[Any]:
        conversion_observations = {}
        for product, observation in observations.conversionObservations.items():
            conversion_observations[product] = [
                observation.bidPrice,
                observation.askPrice,
                observation.transportFees,
                observation.exportTariff,
                observation.importTariff,
                observation.sugarPrice,
                observation.sunlightIndex,
            ]

        return [observations.plainValueObservations, conversion_observations]

    def compress_orders(self, orders: dict[Symbol, list[Order]]) -> list[list[Any]]:
        compressed = []
        for arr in orders.values():
            for order in arr:
                compressed.append([order.symbol, order.price, order.quantity])

        return compressed

    def to_json(self, value: Any) -> str:
        return json.dumps(value, cls=ProsperityEncoder, separators=(",", ":"))

    def truncate(self, value: str, max_length: int) -> str:
        lo, hi = 0, min(len(value), max_length)
        out = ""

        while lo <= hi:
            mid = (lo + hi) // 2

            candidate = value[:mid]
            if len(candidate) < len(value):
                candidate += "..."

            encoded_candidate = json.dumps(candidate)

            if len(encoded_candidate) <= max_length:
                out = candidate
                lo = mid + 1
            else:
                hi = mid - 1

        return out


logger = Logger()




from datamodel import OrderDepth, TradingState, Order, ConversionObservation
from typing import Dict, List, Tuple
import jsonpickle
import numpy as np



class Product:
    MACARONS = "MAGNIFICENT_MACARONS"


# === PARAMETERS ===
PARAMS = {
    Product.MACARONS: {
        "mr_gap_mean": 6.6,
        "mr_band_width": 1.4,
        "mr_trade_qty": 75,

    }
}

POSITION_LIMITS = {Product.MACARONS: 75}
CONVERSION_LIMIT = 10  # max ±10 conversion per tick

        # if not hasattr(self, 'save_list'):
        #     self.save_list = []
                
        # self.save_list.append(diff)
        
        # np.savetxt('diff.csv', self.save_list, delimiter=',', fmt='%f')



def clamp(value: int, low: int, high: int) -> int:
    return max(low, min(high, value))


def implied_bid_ask(obs: ConversionObservation) -> Tuple[float, float]:
    bid = obs.bidPrice - obs.exportTariff - obs.transportFees - 0.1
    ask = obs.askPrice + obs.importTariff + obs.transportFees
    return bid, ask


class Trader:
    def __init__(self, params: Dict = None):
        self.params = params or PARAMS
        self.save_list = []

    def debug_saver(self, var, csv_name: str):
        self.save_list.append(var)  
        np.savetxt(csv_name, self.save_list, delimiter=',', fmt='%f')

    # Mean‑reversion layer (unchanged) …
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

        upper = mean_gap + conf["mr_band_width"] 
        lower = mean_gap - conf["mr_band_width"] 

        # self.debug_saver(gap, 'gap3.csv')

        if gap > upper and pos > -limit:
            trade_qty = clamp(conf["mr_trade_qty"], 0, limit + pos)
           
            if trade_qty:
                orders.append(
                    Order(Product.MACARONS, max(depth.buy_orders), -trade_qty)
                )
                pos -= trade_qty


        elif gap < lower and pos < limit:
            trade_qty = clamp(conf["mr_trade_qty"], 0, limit - pos)
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
            
            orders: Order = []
            depth = state.order_depths[Product.MACARONS]
            obs = state.observations.conversionObservations[Product.MACARONS]
            pos = state.position.get(Product.MACARONS, 0)

            # conversions = clamp(-pos, -CONVERSION_LIMIT, CONVERSION_LIMIT)
            pos += conversions

            implied_bid, implied_ask = implied_bid_ask(obs)
            implied_mid = (implied_bid + implied_ask) / 2

            # 3) mean‑reversion
            mr = self._quote_mean_reversion(depth, implied_mid, pos)
            orders += mr

            # self.debug_saver(pos, "pos3.csv")

            results[Product.MACARONS] = orders

            # leave the rest to conversion (±10 per tick)
            

        # return results, conversions, jsonpickle.encode(tdata)
        
        logger.flush(state, results, conversions, jsonpickle.encode(tdata))
        return results, conversions, jsonpickle.encode(tdata)