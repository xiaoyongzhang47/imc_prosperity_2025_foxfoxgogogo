from datamodel import OrderDepth, TradingState, Order, ConversionObservation
from typing import Dict, List, Tuple
import jsonpickle
import numpy as np
import collections


class Product:
    MACARONS = "MAGNIFICENT_MACARONS"


# === PARAMETERS ===
PARAMS = {
    Product.MACARONS: {
        "mr_gap_mean": 8.61,
        "mr_band_width": 1,
        "mr_trade_qty": 75,
        "mr_gap_win_sz": 50
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


        # # on first call, init the history list
        # if not hasattr(self, 'pos_list'):
        #     self.pos_list = []

        # self.pos_list.append(pos)
        # if len(self.pos_list) > conf["mr_pos_avg_window"]:
        #     self.pos_list.pop(0)

        # mean_pos = np.mean(self.pos_list)

        # # 4) inventory‐bias adjustment
        # pos_th    = 50
        # delta_gap = 0.25

        # if mean_pos < -pos_th:
        #     conf["mr_gap_mean"] += delta_gap

        # elif mean_pos > pos_th:
        #     conf["mr_gap_mean"] -= delta_gap

        # mean_gap = conf["mr_gap_mean"]


        market_mid = (max(depth.buy_orders) + min(depth.sell_orders)) / 2
        gap = market_mid - implied_mid
                
        if not hasattr(self, 'mr_win'):
            self.mr_win = collections.deque()
            
        if len(self.mr_win) == self.params[Product.MACARONS]["mr_gap_win_sz"]:
            self.mr_win.popleft()
        self.mr_win.append(gap)

        mean_gap = sum(self.mr_win)/len(self.mr_win)

        
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

        # print(gap)
        # self.debug_saver(gap, 'gap3.csv')
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

            if state.timestamp < 50000:
                PARAMS[Product.MACARONS]["mr_gap_mean"] = 7.29
            else:
                PARAMS[Product.MACARONS]["mr_gap_mean"] = 8.47


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
            

        return results, conversions, jsonpickle.encode(tdata)