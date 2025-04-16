from datamodel import OrderDepth, UserId, TradingState, Order
from typing import List, Dict, Any, Tuple, Optional
import jsonpickle  # type: ignore
import numpy as np
from math import log, sqrt
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


# Corrected parameter dictionary – note that duplicate keys have been fixed.
PARAMS: Dict[str, Dict[str, Any]] = {
    Product.VR_VOUCHER_1000: {
        "mean_volatility": 0.0591542,
        "threshold": 0.00446339,  # threshold for deviation from the fitted curve
        "strike": 10000,
        "starting_time_to_expiry": 3 / 7,
        "curve_window": 10,  # number of data points to fit the quadratic curve
        "min_curve_points": 3,
    },
    Product.VR_VOUCHER_1025: {
        "mean_volatility": 0.0591542,
        "threshold": 0.00446339,
        "strike": 10250,
        "starting_time_to_expiry": 3 / 7,
        "curve_window": 10,
        "min_curve_points": 3,
    },
    Product.VR_VOUCHER_1050: {
        "mean_volatility": 0.0591542,
        "threshold": 0.00446339,
        "strike": 10500,
        "starting_time_to_expiry": 3 / 7,
        "curve_window": 10,
        "min_curve_points": 3,
    },
    Product.VR_VOUCHER_0950: {
        "mean_volatility": 0.0591542,
        "threshold": 0.00446339,
        "strike": 9500,
        "starting_time_to_expiry": 3 / 7,
        "curve_window": 10,
        "min_curve_points": 3,
    },
    Product.VR_VOUCHER_0975: {
        "mean_volatility": 0.0591542,
        "threshold": 0.00446339,
        "strike": 9750,
        "starting_time_to_expiry": 3 / 7,
        "curve_window": 10,
        "min_curve_points": 3,
    },
}


class BlackScholes:
    @staticmethod
    def black_scholes_call(spot: float, strike: float, time_to_expiry: float, volatility: float) -> float:
        d1 = (log(spot) - log(strike) + 0.5 * volatility * volatility * time_to_expiry) / (volatility * sqrt(time_to_expiry))
        d2 = d1 - volatility * sqrt(time_to_expiry)
        call_price = spot * NormalDist().cdf(d1) - strike * NormalDist().cdf(d2)
        return call_price

    @staticmethod
    def black_scholes_put(spot: float, strike: float, time_to_expiry: float, volatility: float) -> float:
        d1 = (log(spot / strike) + 0.5 * volatility * volatility * time_to_expiry) / (volatility * sqrt(time_to_expiry))
        d2 = d1 - volatility * sqrt(time_to_expiry)
        put_price = strike * NormalDist().cdf(-d2) - spot * NormalDist().cdf(-d1)
        return put_price

    @staticmethod
    def delta(spot: float, strike: float, time_to_expiry: float, volatility: float) -> float:
        d1 = (log(spot) - log(strike) + 0.5 * volatility * volatility * time_to_expiry) / (volatility * sqrt(time_to_expiry))
        return NormalDist().cdf(d1)

    @staticmethod
    def gamma(spot: float, strike: float, time_to_expiry: float, volatility: float) -> float:
        d1 = (log(spot) - log(strike) + 0.5 * volatility * volatility * time_to_expiry) / (volatility * sqrt(time_to_expiry))
        return NormalDist().pdf(d1) / (spot * volatility * sqrt(time_to_expiry))

    @staticmethod
    def vega(spot: float, strike: float, time_to_expiry: float, volatility: float) -> float:
        d1 = (log(spot) - log(strike) + 0.5 * volatility * volatility * time_to_expiry) / (volatility * sqrt(time_to_expiry))
        return NormalDist().pdf(d1) * (spot * sqrt(time_to_expiry)) / 100

    @staticmethod
    def implied_volatility(call_price: float, spot: float, strike: float, time_to_expiry: float,
                             max_iterations: int = 200, tolerance: float = 1e-10) -> float:
        low_vol = 0.01
        high_vol = 1.0
        volatility = (low_vol + high_vol) / 2.0  # initial guess
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


class Trader:
    def __init__(self, params: Optional[Dict[str, Dict[str, Any]]] = None):
        if params is None:
            params = PARAMS
        self.params = params

        # Limits for each product
        self.LIMIT: Dict[str, int] = {
            Product.VOLCANICROCK:    400,
            Product.VR_VOUCHER_0950: 200,
            Product.VR_VOUCHER_0975: 200,
            Product.VR_VOUCHER_1000: 200,
            Product.VR_VOUCHER_1025: 200,
            Product.VR_VOUCHER_1050: 200,
        }

        self.voucher_fit_param_a = 2
        self.voucher_fit_param_b = 0.02

    def get_volcanic_rock_voucher_mid_price(
        self, voucher_order_depth: OrderDepth, traderData: Dict[str, Any]
    ) -> float:
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
        m_t: float,
    ) -> Tuple[Optional[List[Order]], Optional[List[Order]]]:
        """
        Implements the new approach:
        For each voucher product, record the (m_t, volatility) pair and fit a parabolic curve.
        Evaluate the fitted curve at m_t to obtain the expected volatility.
        Use the deviation (residual) between the observed volatility and the fitted value to trigger orders.
        """


        # Store (m_t, volatility) pair for quadratic curve fitting.
        traderData.setdefault("vol_curve", [])
        traderData["vol_curve"].append((m_t, volatility))
        curve_window = self.params[product]["curve_window"]
        if len(traderData["vol_curve"]) > curve_window:
            traderData["vol_curve"].pop(0)

        # Fit quadratic curve if enough data points are available.
        min_points = self.params[product]["min_curve_points"]
        if len(traderData["vol_curve"]) >= min_points:
            X = np.array([pt[0] for pt in traderData["vol_curve"]])
            Y = np.array([pt[1] for pt in traderData["vol_curve"]])
            coeffs = np.polyfit(X, Y, 2)  # quadratic fit: a*m_t^2 + b*m_t + c
            # Calculate fitted volatility at current m_t.
            v_fit = coeffs[0] * m_t * m_t + coeffs[1] * m_t + coeffs[2]
        else:
            v_fit = self.params[product]["mean_volatility"]

        # Compute the residual as deviation from the fitted curve.
        residual = volatility - v_fit
        threshold = self.params[product]["threshold"]

        # Trigger orders based on the residual.
        if residual >= threshold:
            # Voucher appears overpriced; sell to reduce long exposure.
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
        elif residual <= -threshold:
            # Voucher appears underpriced; buy to increase long exposure.
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
        # Determine target position for VOLCANIC_ROCK based on aggregated voucher delta exposure.
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
        # Decode persistent trader data.
        traderObject: Dict[str, Any] = {}
        if state.traderData is not None and state.traderData != "":
            traderObject = jsonpickle.decode(state.traderData)

        result: Dict[str, List[Order]] = {}

        # Ensure traderObject contains entries for each voucher.
        for product in self.params:
            if product.startswith("VOLCANIC_ROCK_VOUCHER"):
                if product not in traderObject:
                    traderObject[product] = {"prev_voucher_price": 0.0, "vol_curve": []}

        # Check VOLCANIC_ROCK order depth availability.
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

        # Process each voucher product.
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
            # Calculate time-to-expiry for the voucher.
            tte = self.params[voucher_product]["starting_time_to_expiry"] - (state.timestamp / 1000000 / 7)
            if tte <= 0:
                continue

            # Compute moneyness measure m_t.
            strike = self.params[voucher_product]["strike"]
            m_t = log(strike / volcanic_rock_mid_price) / sqrt(tte)

            volatility = BlackScholes.implied_volatility(
                voucher_mid_price,
                volcanic_rock_mid_price,
                strike,
                tte,
            )
            delta_val = BlackScholes.delta(
                volcanic_rock_mid_price,
                strike,
                tte,
                volatility,
            )

            take_orders, make_orders = self.voucher_orders(
                voucher_product,
                voucher_order_depth,
                voucher_position,
                traderObject[voucher_product],
                volatility,
                m_t,
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

        volcanic_rock_position = state.position.get(Product.VOLCANICROCK, 0)
        hedge_orders = self.compute_volcanic_rock_hedge_orders(
            volcanic_rock_order_depth, volcanic_rock_position, net_voucher_delta_exposure
        )
        if hedge_orders:
            result[Product.VOLCANICROCK] = hedge_orders

        conversions = 1
        traderDataEncoded = jsonpickle.encode(traderObject)
        return result, conversions, traderDataEncoded