from datamodel import OrderDepth, TradingState, Order
from typing import List, Dict, Any, Tuple, Optional
import jsonpickle  # type: ignore
import numpy as np
from math import log, sqrt
from statistics import NormalDist
import random


class Product:
    VOLCANICROCK = "VOLCANIC_ROCK"
    VR_VOUCHER_0950 = "VOLCANIC_ROCK_VOUCHER_9500"
    VR_VOUCHER_0975 = "VOLCANIC_ROCK_VOUCHER_9750"
    VR_VOUCHER_1000 = "VOLCANIC_ROCK_VOUCHER_10000"
    VR_VOUCHER_1025 = "VOLCANIC_ROCK_VOUCHER_10250"
    VR_VOUCHER_1050 = "VOLCANIC_ROCK_VOUCHER_10500"

# Lower z-score threshold to actually trigger voucher trades
PARAMS = {
    Product.VR_VOUCHER_0950: {"strike": 9500,  "std_window": 10, "zscore_threshold": 2.0},
    Product.VR_VOUCHER_0975: {"strike": 9750,  "std_window": 10, "zscore_threshold": 2.0},
    Product.VR_VOUCHER_1000: {"strike":10000, "std_window": 10, "zscore_threshold": 2.0},
    Product.VR_VOUCHER_1025: {"strike":10250, "std_window": 10, "zscore_threshold": 2.0},
    Product.VR_VOUCHER_1050: {"strike":10500, "std_window": 10, "zscore_threshold": 2.0},
}

class BlackScholes:

    @staticmethod
    def black_scholes_call(spot: float, strike: float, time_to_expiry: float, volatility: float) -> float:
        norm = NormalDist()
        d1 = (log(spot/strike) + 0.5 * volatility**2 * time_to_expiry) / (volatility * sqrt(time_to_expiry))
        d2 = d1 - volatility * sqrt(time_to_expiry)
        return spot * norm.cdf(d1) - strike * norm.cdf(d2)

    @staticmethod
    def black_scholes_put(spot: float, strike: float, time_to_expiry: float, volatility: float) -> float:
        norm = NormalDist()
        d1 = (log(spot/strike) + 0.5 * volatility**2 * time_to_expiry) / (volatility * sqrt(time_to_expiry))
        d2 = d1 - volatility * sqrt(time_to_expiry)
        return strike * norm.cdf(-d2) - spot * norm.cdf(-d1)

    @staticmethod
    def delta(spot: float, strike: float, time_to_expiry: float, volatility: float) -> float:
        norm = NormalDist()
        d1 = (log(spot/strike) + 0.5 * volatility**2 * time_to_expiry) / (volatility * sqrt(time_to_expiry))
        return norm.cdf(d1)

    @staticmethod
    def gamma(spot: float, strike: float, time_to_expiry: float, volatility: float) -> float:
        norm = NormalDist()
        d1 = (log(spot/strike) + 0.5 * volatility**2 * time_to_expiry) / (volatility * sqrt(time_to_expiry))
        return norm.pdf(d1) / (spot * volatility * sqrt(time_to_expiry))

    @staticmethod
    def vega(spot: float, strike: float, time_to_expiry: float, volatility: float) -> float:
        norm = NormalDist()
        d1 = (log(spot/strike) + 0.5 * volatility**2 * time_to_expiry) / (volatility * sqrt(time_to_expiry))
        return norm.pdf(d1) * (spot * sqrt(time_to_expiry)) / 100

    @staticmethod
    def implied_volatility(
        call_price: float,
        spot: float,
        strike: float,
        time_to_expiry: float,
        max_iterations: int = 200,
        tolerance: float = 1e-10
    ) -> float:
        low_vol = 0.01
        high_vol = 1.0
        volatility = 0.5 * (low_vol + high_vol)
        for _ in range(max_iterations):
            estimated = BlackScholes.black_scholes_call(spot, strike, time_to_expiry, volatility)
            diff = estimated - call_price
            if abs(diff) < tolerance:
                break
            if diff > 0:
                high_vol = volatility
            else:
                low_vol = volatility
            volatility = 0.5 * (low_vol + high_vol)
        return volatility

    @staticmethod
    def expected_volatility(strike: float, volcanic_rock_mid_price: float, tte: float, a: float = 0.645, b: float = 0.00686) -> float:
        m_t = log(strike / volcanic_rock_mid_price) / sqrt(tte)
        return a * (m_t ** 2) + b

class Trader:
    def __init__(self, params: Optional[Dict[str, Any]] = None, hedge_ratio: float = 0.25):
        self.params = params if params is not None else PARAMS
        self.hedge_ratio = hedge_ratio
        self.ROUND = 3
        self.LIMIT = {
            Product.VOLCANICROCK:   400,
            Product.VR_VOUCHER_0950: 200,
            Product.VR_VOUCHER_0975: 200,
            Product.VR_VOUCHER_1000: 200,
            Product.VR_VOUCHER_1025: 200,
            Product.VR_VOUCHER_1050: 200,
        }
        self.save_list: List[Any] = []

    def is_good_to_trade(self, product: str, state: TradingState) -> bool:
        return product in self.params and product in state.order_depths

    def get_volcanic_rock_voucher_mid_price(
        self,
        voucher_order_depth: OrderDepth,
        traderData: Dict[str, Any]
    ) -> float:
        if voucher_order_depth.buy_orders and voucher_order_depth.sell_orders:
            best_bid = max(voucher_order_depth.buy_orders.keys())
            best_ask = min(voucher_order_depth.sell_orders.keys())
            mid = 0.5 * (best_bid + best_ask)
            traderData["prev_voucher_price"] = mid
            return mid
        return traderData.get("prev_voucher_price", 0.0)

    def voucher_orders(
        self,
        product: str,
        voucher_order_depth: OrderDepth,
        voucher_position: int,
        traderData: Dict[str, Any],
        volatility: float,
        expected_volatility: float,
    ) -> Tuple[Optional[List[Order]], Optional[List[Order]]]:
        past = traderData.setdefault("past_voucher_vol", [])
        past.append(volatility)
        window = self.params[product]["std_window"]
        if len(past) > window:
            past.pop(0)
        if len(past) < window:
            return None, None

        vol_std = float(np.std(past)) or 1e-8
        z = (volatility - expected_volatility) / vol_std
        thr = self.params[product]["zscore_threshold"]

        # Extreme vol -> take directional voucher positions
        if z >= thr and voucher_position != -self.LIMIT[product] and voucher_order_depth.buy_orders:
            bid = max(voucher_order_depth.buy_orders.keys())
            qty = min(abs(-self.LIMIT[product] - voucher_position), abs(voucher_order_depth.buy_orders[bid]))
            return [Order(product, bid, -qty)], []
        if z <= -thr and voucher_position != self.LIMIT[product] and voucher_order_depth.sell_orders:
            ask = min(voucher_order_depth.sell_orders.keys())
            qty = min(abs(self.LIMIT[product] - voucher_position), abs(voucher_order_depth.sell_orders[ask]))
            return [Order(product, ask, qty)], []
        return None, None

    def compute_volcanic_rock_hedge_orders(
        self,
        volcanic_rock_order_depth: OrderDepth,
        current_position: int,
        net_voucher_delta_exposure: float,
    ) -> List[Order]:
        # Full neutral target
        full_target = -round(net_voucher_delta_exposure)
        # Tilted partial hedge
        tilted = round(full_target * self.hedge_ratio)
        delta_qty = tilted - current_position
        orders: List[Order] = []
        if delta_qty > 0 and volcanic_rock_order_depth.sell_orders:
            ask = min(volcanic_rock_order_depth.sell_orders.keys())
            qty = min(delta_qty, self.LIMIT[Product.VOLCANICROCK] - current_position)
            if qty > 0:
                orders.append(Order(Product.VOLCANICROCK, ask, qty))
        elif delta_qty < 0 and volcanic_rock_order_depth.buy_orders:
            bid = max(volcanic_rock_order_depth.buy_orders.keys())
            qty = min(abs(delta_qty), self.LIMIT[Product.VOLCANICROCK] + current_position)
            if qty > 0:
                orders.append(Order(Product.VOLCANICROCK, bid, -qty))
        return orders

    def run(self, state: TradingState) -> Tuple[Dict[str, List[Order]], int, str]:
        traderData = jsonpickle.decode(state.traderData) if state.traderData else {}
        result: Dict[str, List[Order]] = {}

        # Initialize voucher data
        for p in self.params:
            if p.startswith("VOLCANIC_ROCK_VOUCHER"):
                traderData.setdefault(p, {"prev_voucher_price": 0.0, "past_voucher_vol": []})

        # Check VOLCANIC_ROCK depth
        if Product.VOLCANICROCK not in state.order_depths:
            return result, 1, jsonpickle.encode(traderData)
        vr_depth = state.order_depths[Product.VOLCANICROCK]
        if not(vr_depth.buy_orders and vr_depth.sell_orders):
            return result, 1, jsonpickle.encode(traderData)

        mid = 0.5 * (max(vr_depth.buy_orders.keys()) + min(vr_depth.sell_orders.keys()))
        net_delta = 0.0

        # Voucher loops
        for vp in self.params:
            if not vp.startswith("VOLCANIC_ROCK_VOUCHER") or vp not in state.order_depths:
                continue
            od = state.order_depths[vp]
            pos = state.position.get(vp, 0)
            mid_price = self.get_volcanic_rock_voucher_mid_price(od, traderData[vp])
            time_per = 1_000_000
            curr = (self.ROUND - 1) * time_per + state.timestamp
            tte = max((7 * time_per - curr) / (7 * time_per), 0)
            if tte <= 0:
                continue
            strike = self.params[vp]["strike"]
            exp_vol = BlackScholes.expected_volatility(strike, mid, tte)
            vol = BlackScholes.implied_volatility(mid_price, mid, strike, tte)
            delta_val = BlackScholes.delta(mid, strike, tte, vol)

            take, make = self.voucher_orders(vp, od, pos, traderData[vp], vol, exp_vol)
            if take or make:
                orders = []
                if take:  orders.extend(take)
                if make:  orders.extend(make)
                result[vp] = orders
                executed = sum(o.quantity for o in (take or []))
                pos += executed
            net_delta += pos * delta_val

        # Hedge with tilt (always)
        rock_pos = state.position.get(Product.VOLCANICROCK, 0)
        hedge = self.compute_volcanic_rock_hedge_orders(vr_depth, rock_pos, net_delta)
        if hedge:
            result[Product.VOLCANICROCK] = hedge

        return result, 1, jsonpickle.encode(traderData)
