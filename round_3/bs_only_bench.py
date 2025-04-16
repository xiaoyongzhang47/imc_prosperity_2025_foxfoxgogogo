from datamodel import OrderDepth, UserId, TradingState, Order
from typing import List, Dict, Any, Tuple, Optional
import string 
import jsonpickle # type: ignore
import numpy as np
from math import log, sqrt, exp
from statistics import NormalDist
import random
import time

class Product:
    VOLCANICROCK    = "VOLCANIC_ROCK"
    VR_VOUCHER_1000 = "VOLCANIC_ROCK_VOUCHER_10000"
    VR_VOUCHER_1025 = "VOLCANIC_ROCK_VOUCHER_10250"
    VR_VOUCHER_1050 = "VOLCANIC_ROCK_VOUCHER_10500"
    VR_VOUCHER_0950 = "VOLCANIC_ROCK_VOUCHER_9500"
    VR_VOUCHER_0975 = "VOLCANIC_ROCK_VOUCHER_9750"

PARAMS = {
    Product.VR_VOUCHER_1000: {
        "mean_volatility": 0.00776601703694467,
        "threshold": 0.000787393298129869,
        "strike": 10000,
        "starting_time_to_expiry": 1 / 7,
        "std_window": 10,
        "zscore_threshold": 20,
    },
    Product.VR_VOUCHER_1025: {
        "mean_volatility": 0.00736529669943686,
        "threshold": 0.000458157996135842,
        "strike": 10250,
        "starting_time_to_expiry": 1 / 7,
        "std_window": 10,
        "zscore_threshold": 20,
    },
    Product.VR_VOUCHER_1050: {
        "mean_volatility": 0.00766316841477163,
        "threshold": 0.000323820198178233,
        "strike": 10500,
        "starting_time_to_expiry": 1 / 7,
        "std_window": 10,
        "zscore_threshold": 20,
    },
    Product.VR_VOUCHER_0950: {
        "mean_volatility": 0.00666366978501945,
        "threshold": 0.00571505002926161,
        "strike": 9500,
        "starting_time_to_expiry": 1 / 7,
        "std_window": 10,
        "zscore_threshold": 20,
    },
    Product.VR_VOUCHER_0975: {
        "mean_volatility": 0.00835174751202528,
        "threshold": 0.00241303098370153,
        "strike": 9750,
        "starting_time_to_expiry": 1 / 7,
        "std_window": 10,
        "zscore_threshold": 20,
    },
    }

class BlackScholes:
    @staticmethod
    def black_scholes_call(spot, strike, time_to_expiry, volatility):
        d1 = (
            log(spot) - log(strike) + (0.5 * volatility * volatility) * time_to_expiry
        ) / (volatility * sqrt(time_to_expiry))
        d2 = d1 - volatility * sqrt(time_to_expiry)
        call_price = spot * NormalDist().cdf(d1) - strike * NormalDist().cdf(d2)
        return call_price

    @staticmethod
    def black_scholes_put(spot, strike, time_to_expiry, volatility):
        d1 = (log(spot / strike) + (0.5 * volatility * volatility) * time_to_expiry) / (
            volatility * sqrt(time_to_expiry)
        )
        d2 = d1 - volatility * sqrt(time_to_expiry)
        put_price = strike * NormalDist().cdf(-d2) - spot * NormalDist().cdf(-d1)
        return put_price

    @staticmethod
    def delta(spot, strike, time_to_expiry, volatility):
        d1 = (
            log(spot) - log(strike) + (0.5 * volatility * volatility) * time_to_expiry
        ) / (volatility * sqrt(time_to_expiry))
        return NormalDist().cdf(d1)

    @staticmethod
    def gamma(spot, strike, time_to_expiry, volatility):
        d1 = (
            log(spot) - log(strike) + (0.5 * volatility * volatility) * time_to_expiry
        ) / (volatility * sqrt(time_to_expiry))
        return NormalDist().pdf(d1) / (spot * volatility * sqrt(time_to_expiry))

    @staticmethod
    def vega(spot: float, strike: float, time_to_expiry: float, volatility: float) -> float:
        d1 = (log(spot) - log(strike) + 0.5 * volatility * volatility * time_to_expiry) / (volatility * sqrt(time_to_expiry))
        return NormalDist().pdf(d1) * (spot * sqrt(time_to_expiry)) / 100


    @staticmethod
    def implied_volatility(
        call_price, spot, strike, time_to_expiry, max_iterations=200, tolerance=1e-10
    ):
        low_vol = 0.01
        high_vol = 1.0
        volatility = (low_vol + high_vol) / 2.0  # Initial guess as the midpoint
        for _ in range(max_iterations):
            estimated_price = BlackScholes.black_scholes_call(
                spot, strike, time_to_expiry, volatility
            )
            diff = estimated_price - call_price
            if abs(diff) < tolerance:
                break
            elif diff > 0:
                high_vol = volatility
            else:
                low_vol = volatility
            volatility = (low_vol + high_vol) / 2.0
        return volatility


class Trader:
    def __init__(self, params=None):
        if params is None:
            params = PARAMS
        self.params = params

        self.ROUND = 2
     
        self.voucher_fit_param_a = 2.0
        self.voucher_fit_param_b = 0.0207

        self.LIMIT = {
            # this is the effective limit: The true limit is 400
            Product.VOLCANICROCK:    400,

            Product.VR_VOUCHER_0950: 200,
            Product.VR_VOUCHER_0975: 200,
            Product.VR_VOUCHER_1000: 200,
            Product.VR_VOUCHER_1025: 200,
            Product.VR_VOUCHER_1050: 200,
            }

        self.save_list = []

    def is_good_to_trade(self, product: str, state: TradingState) -> bool:
        return product in self.params and product in state.order_depths

    # ---------------------
    # VOUCHER TRADING
    # ---------------------

    def voucher_starting_time_to_expiry_update(self):
        for product, params in self.params.items():
            if product.startswith("VOLCANIC_ROCK_VOUCHER"):
                params["starting_time_to_expiry"] = (7 - self.ROUND) / 7

    def get_volcanic_rock_voucher_mid_price(
        self, voucher_order_depth: OrderDepth, traderData: Dict[str, Any]
    ) -> float:
        # Use the best bid and best ask from the voucher order depth if available
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
    ) -> Tuple[Optional[List[Order]], Optional[List[Order]]]:
        # Maintain a rolling window of volatility observations
        traderData.setdefault("past_voucher_vol", [])
        traderData["past_voucher_vol"].append(volatility)
        std_window = self.params[product]["std_window"]
        if len(traderData["past_voucher_vol"]) < std_window:
            return None, None
        if len(traderData["past_voucher_vol"]) > std_window:
            traderData["past_voucher_vol"].pop(0)
        vol_std = np.std(traderData["past_voucher_vol"])
        if vol_std == 0:
            vol_std = 1e-8
        vol_z_score = (volatility - self.params[product]["mean_volatility"]) / vol_std

        # Check if the z-score exceeds thresholds to trigger orders
        if vol_z_score >= self.params[product]["zscore_threshold"]:
            if voucher_position != -self.LIMIT[product]:
                target_voucher_position = -self.LIMIT[product]
                if voucher_order_depth.buy_orders:
                    best_bid = max(voucher_order_depth.buy_orders.keys())
                    target_quantity = abs(target_voucher_position - voucher_position)
                    quantity = min(target_quantity, abs(voucher_order_depth.buy_orders[best_bid]))
                    quote_quantity = target_quantity - quantity
                    if quote_quantity == 0:
                        return [Order(product, best_bid, -quantity)], []
                    else:
                        return ([Order(product, best_bid, -quantity)],
                                [Order(product, best_bid, -quote_quantity)])
        elif vol_z_score <= -self.params[product]["zscore_threshold"]:
            if voucher_position != self.LIMIT[product]:
                target_voucher_position = self.LIMIT[product]
                if voucher_order_depth.sell_orders:
                    best_ask = min(voucher_order_depth.sell_orders.keys())
                    target_quantity = abs(target_voucher_position - voucher_position)
                    quantity = min(target_quantity, abs(voucher_order_depth.sell_orders[best_ask]))
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
        # Determine the target position for VOLCANICROCK based on the aggregated delta exposure from vouchers.
        target_position = -round(net_voucher_delta_exposure)
        delta_quantity = target_position - current_position
        orders: List[Order] = []
        if delta_quantity > 0 and volcanic_rock_order_depth.sell_orders:
            best_ask = min(volcanic_rock_order_depth.sell_orders.keys())
            # Limit check for buying
            quantity = min(delta_quantity, self.LIMIT[Product.VOLCANICROCK] - current_position)
            if quantity > 0:
                orders.append(Order(Product.VOLCANICROCK, best_ask, quantity))
        elif delta_quantity < 0 and volcanic_rock_order_depth.buy_orders:
            best_bid = max(volcanic_rock_order_depth.buy_orders.keys())
            quantity = min(abs(delta_quantity), self.LIMIT[Product.VOLCANICROCK] + current_position)
            if quantity > 0:
                orders.append(Order(Product.VOLCANICROCK, best_bid, -quantity))
        return orders

    def run(self, state: TradingState):
        traderObject = {}
        if state.traderData != None and state.traderData != "":
            traderObject = jsonpickle.decode(state.traderData)

        result = {}

        self.voucher_starting_time_to_expiry_update()
        # Ensure traderObject has an entry for every voucher product
        for product in self.params:
            if product.startswith("VOLCANIC_ROCK_VOUCHER"):
                if product not in traderObject:
                    traderObject[product] = {"prev_voucher_price": 0.0, "past_voucher_vol": []}

        # Ensure VOLCANICROCK order depth is available for hedging.
        if Product.VOLCANICROCK not in state.order_depths:
            conversions = 1
            traderDataEncoded = jsonpickle.encode(traderObject)
            return result, conversions, traderDataEncoded

        volcanic_rock_order_depth = state.order_depths[Product.VOLCANICROCK]
        if not (volcanic_rock_order_depth.buy_orders and volcanic_rock_order_depth.sell_orders):
            conversions = 1
            traderDataEncoded = jsonpickle.encode(traderObject)
            return result, conversions, traderDataEncoded

        best_bid = max(volcanic_rock_order_depth.buy_orders.keys())
        best_ask = min(volcanic_rock_order_depth.sell_orders.keys())
        volcanic_rock_mid_price = (best_bid + best_ask) / 2

        net_voucher_delta_exposure = 0.0

        # Process each voucher product
        for voucher_product in self.params:
            if not voucher_product.startswith("VOLCANIC_ROCK_VOUCHER"):
                continue
            if voucher_product not in state.order_depths:
                continue

            voucher_order_depth = state.order_depths[voucher_product]
            voucher_position = state.position.get(voucher_product, 0)
            voucher_mid_price = self.get_volcanic_rock_voucher_mid_price(
                voucher_order_depth, traderObject[voucher_product]
            )
            # time to expiry
            tte = self.params[voucher_product]["starting_time_to_expiry"] - (state.timestamp / 1000000 / 7)

            if tte <= 0:
                continue

            volatility = BlackScholes.implied_volatility(
                voucher_mid_price,
                volcanic_rock_mid_price,
                self.params[voucher_product]["strike"],
                tte,
            )
            delta_val = BlackScholes.delta(
                volcanic_rock_mid_price,
                self.params[voucher_product]["strike"],
                tte,
                volatility,
            )

            take_orders, make_orders = self.voucher_orders(
                voucher_product,
                voucher_order_depth,
                voucher_position,
                traderObject[voucher_product],
                volatility,
            )
            
            # If orders were generated for this voucher, add them to the results.
            if take_orders or make_orders:
                result[voucher_product] = []
                if take_orders:
                    result[voucher_product].extend(take_orders)
                if make_orders:
                    result[voucher_product].extend(make_orders)
                # Assume that "take" orders are executed, updating position.
                executed_qty = sum(order.quantity for order in take_orders) if take_orders else 0
                updated_voucher_position = voucher_position + executed_qty
            else:
                updated_voucher_position = voucher_position

            net_voucher_delta_exposure += updated_voucher_position * delta_val

        # Compute hedge orders for VOLCANICROCK based on the net voucher delta exposure.
        volcanic_rock_position = state.position.get(Product.VOLCANICROCK, 0)
        hedge_orders = self.compute_volcanic_rock_hedge_orders(
            volcanic_rock_order_depth, volcanic_rock_position, net_voucher_delta_exposure
        )
        
        if hedge_orders and random.random() < 7. / 10000:
             result[Product.VOLCANICROCK] = hedge_orders

        conversions = 1
        traderData = jsonpickle.encode(traderObject)

        return result, conversions, traderData
