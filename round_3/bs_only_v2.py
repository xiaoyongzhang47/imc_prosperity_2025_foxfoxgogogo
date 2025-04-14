from datamodel import OrderDepth, UserId, TradingState, Order
from typing import List, Dict, Any, Tuple, Optional
import jsonpickle  # type: ignore
import numpy as np
from math import log, sqrt
from statistics import NormalDist
import random
import time

# ----------------------------
# Product constants
# ----------------------------
class Product:
    VOLCANICROCK = "VOLCANIC_ROCK"

    VR_VOUCHER_1000 = "VOLCANIC_ROCK_VOUCHER_10000"
    VR_VOUCHER_1025 = "VOLCANIC_ROCK_VOUCHER_10250"
    VR_VOUCHER_1050 = "VOLCANIC_ROCK_VOUCHER_10500"
    VR_VOUCHER_0950 = "VOLCANIC_ROCK_VOUCHER_9500"
    VR_VOUCHER_0975 = "VOLCANIC_ROCK_VOUCHER_9750"

# ----------------------------
# Parameters per voucher
# ----------------------------
PARAMS = {
    Product.VR_VOUCHER_1000: {
        "mean_volatility": 0.0591542,
        "threshold": 0.00446339,
        "strike": 10000,
        "starting_time_to_expiry": 3 / 7,
        "std_window": 10,
        "zscore_threshold": 20,
    },
    Product.VR_VOUCHER_1025: {
        "mean_volatility": 0.0591542,
        "threshold": 0.00446339,
        "strike": 10250,
        "starting_time_to_expiry": 3 / 7,
        "std_window": 10,
        "zscore_threshold": 20,
    },
    Product.VR_VOUCHER_1050: {
        "mean_volatility": 0.0591542,
        "threshold": 0.00446339,
        "strike": 10500,
        "starting_time_to_expiry": 3 / 7,
        "std_window": 10,
        "zscore_threshold": 20,
    },
    Product.VR_VOUCHER_0950: {
        "mean_volatility": 0.0591542,
        "threshold": 0.00446339,
        "strike": 9500,
        "starting_time_to_expiry": 3 / 7,
        "std_window": 10,
        "zscore_threshold": 20,
    },
    Product.VR_VOUCHER_0975: {
        "mean_volatility": 0.0591542,
        "threshold": 0.00446339,
        "strike": 9750,
        "starting_time_to_expiry": 3 / 7,
        "std_window": 10,
        "zscore_threshold": 20,
    },
}

# ----------------------------
# Black-Scholes class for option calculations
# ----------------------------
class BlackScholes:
    @staticmethod
    def black_scholes_call(spot, strike, time_to_expiry, volatility):
        d1 = (log(spot) - log(strike) + (0.5 * volatility * volatility) * time_to_expiry) / (volatility * sqrt(time_to_expiry))
        d2 = d1 - volatility * sqrt(time_to_expiry)
        call_price = spot * NormalDist().cdf(d1) - strike * NormalDist().cdf(d2)
        return call_price

    @staticmethod
    def black_scholes_put(spot, strike, time_to_expiry, volatility):
        d1 = (log(spot / strike) + (0.5 * volatility * volatility) * time_to_expiry) / (volatility * sqrt(time_to_expiry))
        d2 = d1 - volatility * sqrt(time_to_expiry)
        put_price = strike * NormalDist().cdf(-d2) - spot * NormalDist().cdf(-d1)
        return put_price

    @staticmethod
    def delta(spot, strike, time_to_expiry, volatility):
        d1 = (log(spot) - log(strike) + (0.5 * volatility * volatility) * time_to_expiry) / (volatility * sqrt(time_to_expiry))
        return NormalDist().cdf(d1)

    @staticmethod
    def gamma(spot, strike, time_to_expiry, volatility):
        d1 = (log(spot) - log(strike) + (0.5 * volatility * volatility) * time_to_expiry) / (volatility * sqrt(time_to_expiry))
        return NormalDist().pdf(d1) / (spot * volatility * sqrt(time_to_expiry))

    @staticmethod
    def vega(spot, strike, time_to_expiry, volatility):
        d1 = (log(spot) - log(strike) + (0.5 * volatility * volatility) * time_to_expiry) / (volatility * sqrt(time_to_expiry))
        return NormalDist().pdf(d1) * (spot * sqrt(time_to_expiry)) / 100

    @staticmethod
    def implied_volatility(call_price, spot, strike, time_to_expiry, max_iterations=200, tolerance=1e-10):
        low_vol = 0.01
        high_vol = 1.0
        volatility = (low_vol + high_vol) / 2.0  # Initial guess as the midpoint
        for _ in range(max_iterations):
            estimated_price = BlackScholes.black_scholes_call(spot, strike, time_to_expiry, volatility)
            diff = estimated_price - call_price
            if abs(diff) < tolerance:
                break
            elif diff > 0:
                high_vol = volatility
            else:
                low_vol = volatility
            volatility = (low_vol + high_vol) / 2.0
        return volatility

# ----------------------------
# Trader class handling orders, hedging and strategy logic
# ----------------------------
class Trader:
    def __init__(self, params: Optional[Dict[str, Any]] = None):
        if params is None:
            params = PARAMS
        self.params = params
        self.LIMIT = {
            Product.VOLCANICROCK: 400,
            Product.VR_VOUCHER_0950: 200,
            Product.VR_VOUCHER_0975: 200,
            Product.VR_VOUCHER_1000: 200,
            Product.VR_VOUCHER_1025: 200,
            Product.VR_VOUCHER_1050: 200,
        }

    # ----------------------------
    # Helper functions
    # ----------------------------

    def get_volcanic_rock_voucher_mid_price(
        self, voucher_order_depth: OrderDepth, traderData: Dict[str, Any]
    ) -> float:
        if voucher_order_depth.buy_orders and voucher_order_depth.sell_orders:
            best_bid = max(voucher_order_depth.buy_orders.keys())
            best_ask = min(voucher_order_depth.sell_orders.keys())
            mid_price = (best_bid + best_ask) / 2
            # Ensure we have a default value stored
            traderData.setdefault("prev_voucher_price", mid_price)
            traderData["prev_voucher_price"] = mid_price
            return mid_price
        else:
            return traderData.get("prev_voucher_price", 0)

    def create_hedge_order(
        self,
        product: str,
        order_depth: OrderDepth,
        target_quantity: int,
        current_position: int,
        order_side: str,  # "buy" or "sell"
    ) -> Optional[Order]:
        if order_side == "buy":
            best_ask = min(order_depth.sell_orders.keys())
            available_quantity = -order_depth.sell_orders[best_ask]
            quantity = min(abs(target_quantity), available_quantity, self.LIMIT[Product.VOLCANICROCK] - current_position)
            if quantity > 0:
                return Order(product, best_ask, quantity)
        elif order_side == "sell":
            best_bid = max(order_depth.buy_orders.keys())
            available_quantity = order_depth.buy_orders[best_bid]
            quantity = min(abs(target_quantity), available_quantity, self.LIMIT[Product.VOLCANICROCK] + current_position)
            if quantity > 0:
                return Order(product, best_bid, -quantity)
        return None

    # ----------------------------
    # Delta hedging methods
    # ----------------------------

    def delta_hedge_volcanic_rock_position(
        self,
        volcanic_rock_order_depth: OrderDepth,
        voucher_position: int,
        volcanic_rock_position: int,
        volcanic_rock_buy_orders: int,
        volcanic_rock_sell_orders: int,
        delta: float,
    ) -> List[Order]:
        target_volcanic_rock_position = -int(delta * voucher_position)
        hedge_quantity = target_volcanic_rock_position - (volcanic_rock_position + volcanic_rock_buy_orders - volcanic_rock_sell_orders)
        orders: List[Order] = []

        if hedge_quantity > 0:
            # Buy underlying
            order = self.create_hedge_order(Product.VOLCANICROCK, volcanic_rock_order_depth, hedge_quantity, volcanic_rock_position + volcanic_rock_buy_orders, "buy")
            if order:
                orders.append(order)
        elif hedge_quantity < 0:
            # Sell underlying
            order = self.create_hedge_order(Product.VOLCANICROCK, volcanic_rock_order_depth, hedge_quantity, volcanic_rock_position - volcanic_rock_sell_orders, "sell")
            if order:
                orders.append(order)
        return orders

    def delta_hedge_volcanic_rock_voucher_orders(
        self,
        volcanic_rock_order_depth: OrderDepth,
        voucher_orders: List[Order],
        volcanic_rock_position: int,
        volcanic_rock_buy_orders: int,
        volcanic_rock_sell_orders: int,
        delta: float,
    ) -> List[Order]:
        if not voucher_orders:
            return []
        net_voucher_quantity = sum(order.quantity for order in voucher_orders)
        target_quantity = -int(delta * net_voucher_quantity)
        orders: List[Order] = []
        if target_quantity > 0:
            order = self.create_hedge_order(Product.VOLCANICROCK, volcanic_rock_order_depth, target_quantity, volcanic_rock_position + volcanic_rock_buy_orders, "buy")
            if order:
                orders.append(order)
        elif target_quantity < 0:
            order = self.create_hedge_order(Product.VOLCANICROCK, volcanic_rock_order_depth, target_quantity, volcanic_rock_position - volcanic_rock_sell_orders, "sell")
            if order:
                orders.append(order)
        return orders

    def volcanic_rock_hedge_orders(
        self,
        volcanic_rock_order_depth: OrderDepth,
        voucher_order_depth: OrderDepth,
        voucher_orders: List[Order],
        volcanic_rock_position: int,
        voucher_position: int,
        delta: float,
    ) -> List[Order]:
        if not voucher_orders:
            voucher_position_after_trade = voucher_position
        else:
            voucher_position_after_trade = voucher_position + sum(order.quantity for order in voucher_orders)

        target_volcanic_rock_position = -delta * voucher_position_after_trade
        if target_volcanic_rock_position == volcanic_rock_position:
            return []
        target_quantity = target_volcanic_rock_position - volcanic_rock_position
        orders: List[Order] = []
        if target_quantity > 0:
            best_ask = min(volcanic_rock_order_depth.sell_orders.keys())
            quantity = min(abs(target_quantity), self.LIMIT[Product.VOLCANICROCK] - volcanic_rock_position)
            if quantity > 0:
                orders.append(Order(Product.VOLCANICROCK, best_ask, round(quantity)))
        elif target_quantity < 0:
            best_bid = max(volcanic_rock_order_depth.buy_orders.keys())
            quantity = min(abs(target_quantity), self.LIMIT[Product.VOLCANICROCK] + volcanic_rock_position)
            if quantity > 0:
                orders.append(Order(Product.VOLCANICROCK, best_bid, -round(quantity)))
        return orders

    # ----------------------------
    # Voucher order generation method
    # ----------------------------
    def volcanic_rock_voucher_orders(
        self,
        voucher_order_depth: OrderDepth,
        voucher_position: int,
        traderData: Dict[str, Any],
        volatility: float,
    ) -> Tuple[Optional[List[Order]], Optional[List[Order]]]:
        # Append the latest volatility to trader data history.
        traderData.setdefault("past_voucher_vol", [])
        traderData["past_voucher_vol"].append(volatility)

        # Ensure we only keep a sliding window of volatilities.
        std_window = self.params[Product.VR_VOUCHER_1000]["std_window"]
        if len(traderData["past_voucher_vol"]) < std_window:
            return None, None
        if len(traderData["past_voucher_vol"]) > std_window:
            traderData["past_voucher_vol"].pop(0)

        vol_std = np.std(traderData["past_voucher_vol"])
        if vol_std == 0:
            return None, None  # Avoid division by zero

        vol_z_score = (volatility - self.params[Product.VR_VOUCHER_1000]["mean_volatility"]) / vol_std

        # Based on z-score thresholds, decide which side to trade.
        if vol_z_score >= self.params[Product.VR_VOUCHER_1000]["zscore_threshold"]:
            if voucher_position != -self.LIMIT[Product.VR_VOUCHER_1000]:
                target_position = -self.LIMIT[Product.VR_VOUCHER_1000]
                if voucher_order_depth.buy_orders:
                    best_bid = max(voucher_order_depth.buy_orders.keys())
                    target_quantity = abs(target_position - voucher_position)
                    quantity = min(target_quantity, abs(voucher_order_depth.buy_orders[best_bid]))
                    quote_quantity = target_quantity - quantity
                    if quote_quantity == 0:
                        return [Order(Product.VR_VOUCHER_1000, best_bid, -quantity)], []
                    else:
                        return [Order(Product.VR_VOUCHER_1000, best_bid, -quantity)], [Order(Product.VR_VOUCHER_1000, best_bid, -quote_quantity)]
        elif vol_z_score <= -self.params[Product.VR_VOUCHER_1000]["zscore_threshold"]:
            if voucher_position != self.LIMIT[Product.VR_VOUCHER_1000]:
                target_position = self.LIMIT[Product.VR_VOUCHER_1000]
                if voucher_order_depth.sell_orders:
                    best_ask = min(voucher_order_depth.sell_orders.keys())
                    target_quantity = abs(target_position - voucher_position)
                    quantity = min(target_quantity, abs(voucher_order_depth.sell_orders[best_ask]))
                    quote_quantity = target_quantity - quantity
                    if quote_quantity == 0:
                        return [Order(Product.VR_VOUCHER_1000, best_ask, quantity)], []
                    else:
                        return [Order(Product.VR_VOUCHER_1000, best_ask, quantity)], [Order(Product.VR_VOUCHER_1000, best_ask, quote_quantity)]
        return None, None

    # ----------------------------
    # Unified voucher handling method
    # ----------------------------
    def handle_voucher(
        self, product_voucher: str, state: TradingState, traderObject: Dict[str, Any]
    ) -> Tuple[Optional[List[Order]], Optional[List[Order]], float, float]:
        voucher_order_depth = state.order_depths[product_voucher]
        voucher_position = state.position.get(product_voucher, 0)
        # Initialize trader data for this voucher if missing.
        if product_voucher not in traderObject:
            traderObject[product_voucher] = {"prev_voucher_price": 0, "past_voucher_vol": []}
        mid_price = self.get_volcanic_rock_voucher_mid_price(voucher_order_depth, traderObject[product_voucher])
        # For the underlying, use the volcanic rock mid price.
        underlying_depth = state.order_depths[Product.VOLCANICROCK]
        underlying_mid_price = (min(underlying_depth.buy_orders.keys()) + max(underlying_depth.sell_orders.keys())) / 2
        tte = self.params[product_voucher]["starting_time_to_expiry"] - (state.timestamp) / 1000000 / 7
        volatility = BlackScholes.implied_volatility(
            mid_price,
            underlying_mid_price,
            self.params[product_voucher]["strike"],
            tte,
        )
        delta = BlackScholes.delta(
            underlying_mid_price,
            self.params[product_voucher]["strike"],
            tte,
            volatility,
        )
        take_orders, make_orders = self.volcanic_rock_voucher_orders(voucher_order_depth, voucher_position, traderObject[product_voucher], volatility)
        return take_orders, make_orders, delta, volatility

    # ----------------------------
    # Main run method
    # ----------------------------
    def run(self, state: TradingState) -> Tuple[Dict[str, List[Order]], int, str]:
        traderObject = jsonpickle.decode(state.traderData) if state.traderData not in [None, ""] else {}
        result: Dict[str, List[Order]] = {}
        conversions = 1

        # List of voucher products to process.
        voucher_products = [
            Product.VR_VOUCHER_1000,
            Product.VR_VOUCHER_1025,
            Product.VR_VOUCHER_1050,
            Product.VR_VOUCHER_0975,
        ]

        aggregated_delta = 0.0
        for voucher in voucher_products:
            if voucher not in traderObject:
                traderObject[voucher] = {"prev_voucher_price": 0, "past_voucher_vol": []}
            if voucher in self.params and voucher in state.order_depths:
                take_orders, make_orders, delta, volatility = self.handle_voucher(voucher, state, traderObject)
                net_voucher_order = 0
                if take_orders is not None:
                    for order in take_orders:
                        net_voucher_order += order.quantity
                    result.setdefault(voucher, []).extend(take_orders)
                if make_orders is not None:
                    for order in make_orders:
                        net_voucher_order += order.quantity
                    result.setdefault(voucher, []).extend(make_orders)
                voucher_position = state.position.get(voucher, 0)
                aggregated_delta += delta * voucher_position

        # Hedge the aggregated delta with the underlying volcanic rock.
        if Product.VOLCANICROCK in state.order_depths:
            underlying_depth = state.order_depths[Product.VOLCANICROCK]
            underlying_position = state.position.get(Product.VOLCANICROCK, 0)
            if underlying_depth and aggregated_delta != 0:
                target_underlying_position = -aggregated_delta
                diff = target_underlying_position - underlying_position
                if diff > 0:
                    best_ask = min(underlying_depth.sell_orders.keys())
                    order_qty = min(diff, self.LIMIT[Product.VOLCANICROCK] - underlying_position)
                    if order_qty > 0:
                        result.setdefault(Product.VOLCANICROCK, []).append(Order(Product.VOLCANICROCK, best_ask, order_qty))
                elif diff < 0:
                    best_bid = max(underlying_depth.buy_orders.keys())
                    order_qty = min(abs(diff), self.LIMIT[Product.VOLCANICROCK] + underlying_position)
                    if order_qty > 0:
                        result.setdefault(Product.VOLCANICROCK, []).append(Order(Product.VOLCANICROCK, best_bid, -order_qty))

        traderData = jsonpickle.encode(traderObject)
        return result, conversions, traderData