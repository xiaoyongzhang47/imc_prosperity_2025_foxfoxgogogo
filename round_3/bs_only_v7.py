from datamodel import OrderDepth, TradingState, Order
from typing import List, Dict, Any, Tuple, Optional
import jsonpickle  # type: ignore
import numpy as np
from math import log, sqrt
from statistics import NormalDist
import random


class Product:
    VOLCANICROCK = "VOLCANIC_ROCK"
    VR_VOUCHER_1000 = "VOLCANIC_ROCK_VOUCHER_10000"
    VR_VOUCHER_1025 = "VOLCANIC_ROCK_VOUCHER_10250"
    VR_VOUCHER_1050 = "VOLCANIC_ROCK_VOUCHER_10500"
    VR_VOUCHER_0950 = "VOLCANIC_ROCK_VOUCHER_9500"
    VR_VOUCHER_0975 = "VOLCANIC_ROCK_VOUCHER_9750"


PARAMS = {
    Product.VR_VOUCHER_1000: {
        "strike": 10000,
        "std_window": 10,
        "zscore_threshold": 20,
    },
    Product.VR_VOUCHER_1025: {
        "strike": 10250,
        "std_window": 10,
        "zscore_threshold": 20,
    },
    Product.VR_VOUCHER_1050: {
        "strike": 10500,
        "std_window": 10,
        "zscore_threshold": 20,
    },
    Product.VR_VOUCHER_0950: {
        "strike": 9500,
        "std_window": 10,
        "zscore_threshold": 20,
    },
    Product.VR_VOUCHER_0975: {
        "strike": 9750,
        "std_window": 10,
        "zscore_threshold": 20,
    },
}

class BlackScholes:

    @staticmethod
    def black_scholes_call(spot: float, strike: float, time_to_expiry: float, volatility: float) -> float:
        norm = NormalDist()
        d1 = (log(spot) - log(strike) + (0.5 * volatility * volatility) * time_to_expiry) / (volatility * sqrt(time_to_expiry))
        d2 = d1 - volatility * sqrt(time_to_expiry)
        call_price = spot * norm.cdf(d1) - strike * norm.cdf(d2)
        return call_price

    @staticmethod
    def black_scholes_put(spot: float, strike: float, time_to_expiry: float, volatility: float) -> float:
        norm = NormalDist()
        d1 = (log(spot / strike) + (0.5 * volatility * volatility) * time_to_expiry) / (volatility * sqrt(time_to_expiry))
        d2 = d1 - volatility * sqrt(time_to_expiry)
        put_price = strike * norm.cdf(-d2) - spot * norm.cdf(-d1)
        return put_price

    @staticmethod
    def delta(spot: float, strike: float, time_to_expiry: float, volatility: float) -> float:
        norm = NormalDist()
        d1 = (log(spot) - log(strike) + (0.5 * volatility * volatility) * time_to_expiry) / (volatility * sqrt(time_to_expiry))
        return norm.cdf(d1)

    @staticmethod
    def gamma(spot: float, strike: float, time_to_expiry: float, volatility: float) -> float:
        norm = NormalDist()
        d1 = (log(spot) - log(strike) + (0.5 * volatility * volatility) * time_to_expiry) / (volatility * sqrt(time_to_expiry))
        return norm.pdf(d1) / (spot * volatility * sqrt(time_to_expiry))

    @staticmethod
    def vega(spot: float, strike: float, time_to_expiry: float, volatility: float) -> float:
        norm = NormalDist()
        d1 = (log(spot) - log(strike) + 0.5 * volatility * volatility * time_to_expiry) / (volatility * sqrt(time_to_expiry))
        return norm.pdf(d1) * (spot * sqrt(time_to_expiry)) / 100

    @staticmethod
    def implied_volatility(
        call_price: float, spot: float, strike: float, time_to_expiry: float, max_iterations: int = 200, tolerance: float = 1e-10
    ) -> float:
        low_vol = 0.01
        high_vol = 1.0
        volatility = (low_vol + high_vol) / 2.0  # initial guess
        for _ in range(max_iterations):
            estimated_price = BlackScholes.black_scholes_call(spot, strike, time_to_expiry, volatility)
            diff = estimated_price - call_price
            if abs(diff) < tolerance:
                break
            if diff > 0:
                high_vol = volatility
            else:
                low_vol = volatility
            volatility = (low_vol + high_vol) / 2.0
        return volatility

    @staticmethod
    def expected_volatility(strike: float, volcanic_rock_mid_price: float, tte: float, a: float = 0.645, b: float = 0.00686) -> float:
        m_t = log(strike / volcanic_rock_mid_price) / sqrt(tte)
        return a * (m_t ** 2) + b


class Trader:
    def __init__(self, params: Optional[Dict[str, Any]] = None):
        self.params = params if params is not None else PARAMS
        self.ROUND = 2

        self.LIMIT = {
            Product.VOLCANICROCK:    400,  # effective limit (true limit is 400)
            Product.VR_VOUCHER_0950: 200,
            Product.VR_VOUCHER_0975: 200,
            Product.VR_VOUCHER_1000: 200,
            Product.VR_VOUCHER_1025: 200,
            Product.VR_VOUCHER_1050: 200,
        }
        self.save_list = []

    def is_good_to_trade(self, product: str, state: TradingState) -> bool:
        return product in self.params and product in state.order_depths


    def get_volcanic_rock_voucher_mid_price(self, voucher_order_depth: OrderDepth, traderData: Dict[str, Any]) -> float:
        if voucher_order_depth.buy_orders and voucher_order_depth.sell_orders:
            best_bid = max(voucher_order_depth.buy_orders.keys())
            best_ask = min(voucher_order_depth.sell_orders.keys())
            mid_price = (best_bid + best_ask) / 2
            traderData["prev_voucher_price"] = mid_price
            return mid_price
        else:
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
        # Maintain a rolling window of volatility observations.
        past_vol = traderData.setdefault("past_voucher_vol", [])
        past_vol.append(volatility)
        std_window = self.params[product]["std_window"]
        if len(past_vol) > std_window:
            past_vol.pop(0)
        if len(past_vol) < std_window:
            return None, None

        vol_std = np.std(past_vol)
        if vol_std == 0:
            vol_std = 1e-8
        vol_z_score = (volatility - expected_volatility) / vol_std

        # If the z-score exceeds thresholds, trigger orders.
        if vol_z_score >= self.params[product]["zscore_threshold"]:
            if voucher_position != -self.LIMIT[product] and voucher_order_depth.buy_orders:
                best_bid = max(voucher_order_depth.buy_orders.keys())
                target_voucher_position = -self.LIMIT[product]
                target_quantity = abs(target_voucher_position - voucher_position)
                available_qty = abs(voucher_order_depth.buy_orders[best_bid])
                quantity = min(target_quantity, available_qty)
                quote_quantity = target_quantity - quantity
                if quote_quantity == 0:
                    return [Order(product, best_bid, -quantity)], []
                else:
                    return ([Order(product, best_bid, -quantity)],
                            [Order(product, best_bid, -quote_quantity)])
        elif vol_z_score <= -self.params[product]["zscore_threshold"]:
            if voucher_position != self.LIMIT[product] and voucher_order_depth.sell_orders:
                best_ask = min(voucher_order_depth.sell_orders.keys())
                target_voucher_position = self.LIMIT[product]
                target_quantity = abs(target_voucher_position - voucher_position)
                available_qty = abs(voucher_order_depth.sell_orders[best_ask])
                quantity = min(target_quantity, available_qty)
                quote_quantity = target_quantity - quantity
                if quote_quantity == 0:
                    return [Order(product, best_ask, quantity)], []
                else:
                    return ([Order(product, best_ask, quantity)],
                            [Order(product, best_ask, quote_quantity)])
        return None, None

    def compute_volcanic_rock_hedge_orders(
        self,
        volcanic_rock_order_depth: OrderDepth,
        current_position: int,
        net_voucher_delta_exposure: float,
    ) -> List[Order]:
        target_position = -round(net_voucher_delta_exposure)
        delta_quantity = target_position - current_position
        orders: List[Order] = []
        if delta_quantity > 0 and volcanic_rock_order_depth.sell_orders:
            best_ask = min(volcanic_rock_order_depth.sell_orders.keys())
            quantity = min(delta_quantity, self.LIMIT[Product.VOLCANICROCK] - current_position)
            if quantity > 0:
                orders.append(Order(Product.VOLCANICROCK, best_ask, quantity))
        elif delta_quantity < 0 and volcanic_rock_order_depth.buy_orders:
            best_bid = max(volcanic_rock_order_depth.buy_orders.keys())
            quantity = min(abs(delta_quantity), self.LIMIT[Product.VOLCANICROCK] + current_position)
            if quantity > 0:
                orders.append(Order(Product.VOLCANICROCK, best_bid, -quantity))
        return orders

    def run(self, state: TradingState) -> Tuple[Dict[str, List[Order]], int, str]:
        traderObject: Dict[str, Any] = {}
        if state.traderData is not None and state.traderData != "":
            traderObject = jsonpickle.decode(state.traderData)
        result: Dict[str, List[Order]] = {}


        # Ensure each voucher product has an entry in traderObject.
        for product in self.params:
            if product.startswith("VOLCANIC_ROCK_VOUCHER") and product not in traderObject:
                traderObject[product] = {"prev_voucher_price": 0.0, "past_voucher_vol": []}

        # Check for VOLCANICROCK order depth availability.
        if Product.VOLCANICROCK not in state.order_depths:
            traderDataEncoded = jsonpickle.encode(traderObject)
            return result, 1, traderDataEncoded

        volcanic_rock_order_depth = state.order_depths[Product.VOLCANICROCK]
        if not (volcanic_rock_order_depth.buy_orders and volcanic_rock_order_depth.sell_orders):
            traderDataEncoded = jsonpickle.encode(traderObject)
            return result, 1, traderDataEncoded

        best_bid = max(volcanic_rock_order_depth.buy_orders.keys())
        best_ask = min(volcanic_rock_order_depth.sell_orders.keys())
        volcanic_rock_mid_price = (best_bid + best_ask) / 2

        net_voucher_delta_exposure = 0.0

        # Process each voucher product.
        for voucher_product in self.params:
            if not voucher_product.startswith("VOLCANIC_ROCK_VOUCHER") or voucher_product not in state.order_depths:
                continue

            voucher_order_depth = state.order_depths[voucher_product]
            voucher_position = state.position.get(voucher_product, 0)
            voucher_mid_price = self.get_volcanic_rock_voucher_mid_price(voucher_order_depth, traderObject[voucher_product])

            # Calculate time to expiry.
            time_per_round = 1000000
            current_time = (self.ROUND - 1) * time_per_round + state.timestamp
            total_time = 7 * time_per_round
            tte = (total_time - current_time) / total_time
            if tte <= 0:
                continue

            strike = self.params[voucher_product]["strike"]
            expected_vol = BlackScholes.expected_volatility(strike, volcanic_rock_mid_price, tte)
            vol = BlackScholes.implied_volatility(voucher_mid_price, volcanic_rock_mid_price, strike, tte)
            delta_val = BlackScholes.delta(volcanic_rock_mid_price, strike, tte, vol)

            take_orders, make_orders = self.voucher_orders(
                voucher_product,
                voucher_order_depth,
                voucher_position,
                traderObject[voucher_product],
                vol,
                expected_vol
            )

            if take_orders or make_orders:
                result[voucher_product] = []
                if take_orders:
                    result[voucher_product].extend(take_orders)
                if make_orders:
                    result[voucher_product].extend(make_orders)
                executed_qty = sum(order.quantity for order in take_orders) if take_orders else 0
                updated_voucher_position = voucher_position + executed_qty
            else:
                updated_voucher_position = voucher_position

            net_voucher_delta_exposure += updated_voucher_position * delta_val

        # Compute hedge orders for VOLCANICROCK.
        volcanic_rock_position = state.position.get(Product.VOLCANICROCK, 0)
        hedge_orders = self.compute_volcanic_rock_hedge_orders(volcanic_rock_order_depth, volcanic_rock_position, net_voucher_delta_exposure)
        
        if hedge_orders and random.random() < 7. / 10000:
            result[Product.VOLCANICROCK] = hedge_orders

        traderDataEncoded = jsonpickle.encode(traderObject)
        return result, 1, traderDataEncoded