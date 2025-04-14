from datamodel import OrderDepth, UserId, TradingState, Order
from typing import List, Dict, Any
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
        "mean_volatility": 0.0591542,
        "threshold": 0.00446339,
        "strike": 10000,
        "starting_time_to_expiry": 3 / 7,
        "std_window": 10,
        "zscore_threshold": 20,
    }, Product.VR_VOUCHER_1025: {
        "mean_volatility": 0.0591542,
        "threshold": 0.00446339,
        "strike": 10250,
        "starting_time_to_expiry": 3 / 7,
        "std_window": 10,
        "zscore_threshold": 20,
    }, Product.VR_VOUCHER_1050: {
        "mean_volatility": 0.0591542,
        "threshold": 0.00446339,
        "strike": 10500,
        "starting_time_to_expiry": 3 / 7,
        "std_window": 10,
        "zscore_threshold": 20,
    }, Product.VR_VOUCHER_0950: {
        "mean_volatility": 0.0591542,
        "threshold": 0.00446339,
        "strike": 9500,
        "starting_time_to_expiry": 3 / 7,
        "std_window": 10,
        "zscore_threshold": 20,
    },
    Product.VR_VOUCHER_0950: {
        "mean_volatility": 0.0591542,
        "threshold": 0.00446339,
        "strike": 9750,
        "starting_time_to_expiry": 3 / 7,
        "std_window": 10,
        "zscore_threshold": 20,
    }
    
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
    def vega(spot, strike, time_to_expiry, volatility):
        d1 = (
            log(spot) - log(strike) + (0.5 * volatility * volatility) * time_to_expiry
        ) / (volatility * sqrt(time_to_expiry))
        # print(f"d1: {d1}")
        # print(f"vol: {volatility}")
        # print(f"spot: {spot}")
        # print(f"strike: {strike}")
        # print(f"time: {time_to_expiry}")
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

        self.LIMIT = {

            Product.VOLCANICROCK:   400,

            Product.VR_VOUCHER_0950:200,
            Product.VR_VOUCHER_0975:200,
            Product.VR_VOUCHER_1000:200,
            Product.VR_VOUCHER_1025:200,
            Product.VR_VOUCHER_1050:200,
            }



    # ---------------------
    # VOUCHER TRADING
    # ---------------------

    def get_volcanic_rock_voucher_mid_price(
        self, volcanic_rock_voucher_order_depth: OrderDepth, traderData: Dict[str, Any]
    ):
        if (
            len(volcanic_rock_voucher_order_depth.buy_orders) > 0
            and len(volcanic_rock_voucher_order_depth.sell_orders) > 0
        ):
            best_bid = max(volcanic_rock_voucher_order_depth.buy_orders.keys())
            best_ask = min(volcanic_rock_voucher_order_depth.sell_orders.keys())
            traderData["prev_voucher_price"] = (best_bid + best_ask) / 2
            return (best_bid + best_ask) / 2
        else:
            return traderData["prev_voucher_price"]

    def delta_hedge_volcanic_rock_position(
        self,
        volcanic_rock_order_depth: OrderDepth,
        volcanic_rock_voucher_position: int,
        volcanic_rock_position: int,
        volcanic_rock_buy_orders: int,
        volcanic_rock_sell_orders: int,
        delta: float,
    ) -> List[Order]:
        """
        Delta hedge the overall position in VOCANICROCK_VOUCHER by creating orders in VOLCANICROCK .

        Args:
            volcanic_rock_order_depth (OrderDepth): The order depth for the VOLCANICROCK  product.
            volcanic_rock_voucher_position (int): The current position in VOCANICROCK_VOUCHER.
            volcanic_rock_position (int): The current position in VOLCANICROCK .
            volcanic_rock_buy_orders (int): The total quantity of buy orders for VOLCANICROCK  in the current iteration.
            volcanic_rock_sell_orders (int): The total quantity of sell orders for VOLCANICROCK  in the current iteration.
            delta (float): The current value of delta for the VOCANICROCK_VOUCHER product.
            traderData (Dict[str, Any]): The trader data for the VOCANICROCK_VOUCHER product.

        Returns:
            List[Order]: A list of orders to delta hedge the VOCANICROCK_VOUCHER position.
        """

        target_volcanic_rock_position = -int(delta * volcanic_rock_voucher_position)
        hedge_quantity = target_volcanic_rock_position - (
            volcanic_rock_position + volcanic_rock_buy_orders - volcanic_rock_sell_orders
        )

        orders: List[Order] = []
        if hedge_quantity > 0:
            # Buy VOLCANICROCK 
            best_ask = min(volcanic_rock_order_depth.sell_orders.keys())
            quantity = min(
                abs(hedge_quantity), -volcanic_rock_order_depth.sell_orders[best_ask]
            )
            quantity = min(
                quantity,
                self.LIMIT[Product.VOLCANICROCK] - (volcanic_rock_position + volcanic_rock_buy_orders),
            )
            if quantity > 0:
                orders.append(Order(Product.VOLCANICROCK, best_ask, quantity))
        elif hedge_quantity < 0:
            # Sell VOLCANICROCK 
            best_bid = max(volcanic_rock_order_depth.buy_orders.keys())
            quantity = min(
                abs(hedge_quantity), volcanic_rock_order_depth.buy_orders[best_bid]
            )
            quantity = min(
                quantity,
                self.LIMIT[Product.VOLCANICROCK] + (volcanic_rock_position - volcanic_rock_sell_orders),
            )
            if quantity > 0:
                orders.append(Order(Product.VOLCANICROCK , best_bid, -quantity))

        return orders

    def delta_hedge_volcanic_rock_voucher_orders(
        self,
        volcanic_rock_order_depth: OrderDepth,
        volcanic_rock_voucher_orders: List[Order],
        volcanic_rock_position: int,
        volcanic_rock_buy_orders: int,
        volcanic_rock_sell_orders: int,
        delta: float,
    ) -> List[Order]:
        """
        Delta hedge the new orders for VOLCANICROCK_VOUCHER by creating orders in VOLCANICROCK.

        Args:
            volcanic_rock_order_depth (OrderDepth): The order depth for the VOLCANICROCK product.
            volcanic_rock_voucher_orders (List[Order]): The new orders for VOLCANICROCK_VOUCHER.
            volcanic_rock_position (int): The current position in VOLCANICROCK.
            volcanic_rock_buy_orders (int): The total quantity of buy orders for VOLCANICROCK in the current iteration.
            volcanic_rock_sell_orders (int): The total quantity of sell orders for VOLCANICROCK in the current iteration.
            delta (float): The current value of delta for the VOLCANICROCK_VOUCHER product.

        Returns:
            List[Order]: A list of orders to delta hedge the new VOLCANICROCK_VOUCHER orders.
        """
        if len(volcanic_rock_voucher_orders) == 0:
            return None

        net_volcanic_rock_voucher_quantity = sum(
            order.quantity for order in volcanic_rock_voucher_orders
        )
        target_volcanic_rock_quantity = -int(delta * net_volcanic_rock_voucher_quantity)

        orders: List[Order] = []
        if target_volcanic_rock_quantity > 0:
            # Buy VOLCANICROCK
            best_ask = min(volcanic_rock_order_depth.sell_orders.keys())
            quantity = min(
                abs(target_volcanic_rock_quantity), -volcanic_rock_order_depth.sell_orders[best_ask]
            )
            quantity = min(
                quantity,
                self.LIMIT[Product.VOLCANICROCK] - (volcanic_rock_position + volcanic_rock_buy_orders),
            )
            if quantity > 0:
                orders.append(Order(Product.VOLCANICROCK, best_ask, quantity))
        elif target_volcanic_rock_quantity < 0:
            # Sell VOLCANICROCK
            best_bid = max(volcanic_rock_order_depth.buy_orders.keys())
            quantity = min(
                abs(target_volcanic_rock_quantity), volcanic_rock_order_depth.buy_orders[best_bid]
            )
            quantity = min(
                quantity,
                self.LIMIT[Product.VOLCANICROCK] + (volcanic_rock_position - volcanic_rock_sell_orders),
            )
            if quantity > 0:
                orders.append(Order(Product.VOLCANICROCK, best_bid, -quantity))

        return orders


    def volcanic_rock_hedge_orders(
        self,
        volcanic_rock_order_depth: OrderDepth,
        volcanic_rock_voucher_order_depth: OrderDepth,
        volcanic_rock_voucher_orders: List[Order],
        volcanic_rock_position: int,
        volcanic_rock_voucher_position: int,
        delta: float,
    ) -> List[Order]:
        
        if volcanic_rock_voucher_orders == None or len(volcanic_rock_voucher_orders) == 0:
            volcanic_rock_voucher_position_after_trade = volcanic_rock_voucher_position
        else:
            volcanic_rock_voucher_position_after_trade = volcanic_rock_voucher_position + sum(
                order.quantity for order in volcanic_rock_voucher_orders
            )

        target_volcanic_rock_position = -delta * volcanic_rock_voucher_position_after_trade

        if target_volcanic_rock_position == volcanic_rock_position:
            return None

        target_volcanic_rock_quantity = target_volcanic_rock_position - volcanic_rock_position

        orders: List[Order] = []
        if target_volcanic_rock_quantity > 0:
            # Buy VOLCANICROCK
            best_ask = min(volcanic_rock_order_depth.sell_orders.keys())
            quantity = min(
                abs(target_volcanic_rock_quantity),
                self.LIMIT[Product.VOLCANICROCK] - volcanic_rock_position,
            )
            if quantity > 0:
                orders.append(Order(Product.VOLCANICROCK, best_ask, round(quantity)))

        elif target_volcanic_rock_quantity < 0:
            # Sell VOLCANICROCK
            best_bid = max(volcanic_rock_order_depth.buy_orders.keys())
            quantity = min(
                abs(target_volcanic_rock_quantity),
                self.LIMIT[Product.VOLCANICROCK] + volcanic_rock_position,
            )
            if quantity > 0:
                orders.append(Order(Product.VOLCANICROCK, best_bid, -round(quantity)))

        return orders

    def volcanic_rock_voucher_orders(
        self,
        volcanic_rock_voucher_order_depth: OrderDepth,
        volcanic_rock_voucher_position: int,
        traderData: Dict[str, Any],
        volatility: float,
    ) -> List[Order]:
        traderData["past_voucher_vol"].append(volatility)



        # print( len(traderData["past_voucher_vol"]))

        if (
            len(traderData["past_voucher_vol"])
            < self.params[Product.VR_VOUCHER_1000]["std_window"]
        ):
            return None, None

        if (
            len(traderData["past_voucher_vol"])
            > self.params[Product.VR_VOUCHER_1000]["std_window"]
        ):
            traderData["past_voucher_vol"].pop(0)

        vol_z_score = (
            volatility - self.params[Product.VR_VOUCHER_1000]["mean_volatility"]
        ) / np.std(traderData["past_voucher_vol"])

        # Optionally save deviation history (for debugging or offline analysis)
        # if not hasattr(self, 'save_list'):
        #     self.save_list = []
                
        # self.save_list.append(volatility)
        # np.savetxt('volatility1.csv', self.save_list, delimiter=',', fmt='%f')


        # print(f"vol_z_score: {vol_z_score}")
        # print(f"zscore_threshold: {self.params[Product.VR_VOUCHER_1000]['zscore_threshold']}")

        if vol_z_score >= self.params[Product.VR_VOUCHER_1000]["zscore_threshold"]:
            if volcanic_rock_voucher_position != -self.LIMIT[Product.VR_VOUCHER_1000]:
                target_volcanic_rock_voucher_position = -self.LIMIT[Product.VR_VOUCHER_1000]
                if len(volcanic_rock_voucher_order_depth.buy_orders) > 0:
                    best_bid = max(volcanic_rock_voucher_order_depth.buy_orders.keys())
                    target_quantity = abs(
                        target_volcanic_rock_voucher_position - volcanic_rock_voucher_position
                    )
                    quantity = min(
                        target_quantity,
                        abs(volcanic_rock_voucher_order_depth.buy_orders[best_bid]),
                    )
                    quote_quantity = target_quantity - quantity
                    if quote_quantity == 0:
                        return [Order(Product.VR_VOUCHER_1000, best_bid, -quantity)], []
                    else:
                        return [Order(Product.VR_VOUCHER_1000, best_bid, -quantity)], [
                            Order(Product.VR_VOUCHER_1000, best_bid, -quote_quantity)
                        ]

        elif vol_z_score <= -self.params[Product.VR_VOUCHER_1000]["zscore_threshold"]:
            if volcanic_rock_voucher_position != self.LIMIT[Product.VR_VOUCHER_1000]:
                target_volcanic_rock_voucher_position = self.LIMIT[Product.VR_VOUCHER_1000]
                if len(volcanic_rock_voucher_order_depth.sell_orders) > 0:
                    best_ask = min(volcanic_rock_voucher_order_depth.sell_orders.keys())
                    target_quantity = abs(
                        target_volcanic_rock_voucher_position - volcanic_rock_voucher_position
                    )
                    quantity = min(
                        target_quantity,
                        abs(volcanic_rock_voucher_order_depth.sell_orders[best_ask]),
                    )
                    quote_quantity = target_quantity - quantity
                    if quote_quantity == 0:
                        return [Order(Product.VR_VOUCHER_1000, best_ask, quantity)], []
                    else:
                        return [Order(Product.VR_VOUCHER_1000, best_ask, quantity)], [
                            Order(Product.VR_VOUCHER_1000, best_ask, quote_quantity)
                        ]

        return None, None


    def run(self, state: TradingState):
        traderObject = {}
        if state.traderData != None and state.traderData != "":
            traderObject = jsonpickle.decode(state.traderData)

        result = {}

        # TODO: 
        # 1. Try to make it a function, so you dont need to write this 3 times.
        # 2. Try to find a good stratage to trade between 3 different vouchers (european options) and the volcanic rock
       
         
        if Product.VR_VOUCHER_1000 not in traderObject:
            traderObject[Product.VR_VOUCHER_1000] = {
                "prev_voucher_price": 0,
                "past_voucher_vol": [],
            }

        if (
            Product.VR_VOUCHER_1000 in self.params
            and Product.VR_VOUCHER_1000 in state.order_depths
        ):
            volcanic_rock_voucher_position = (
                state.position[Product.VR_VOUCHER_1000]
                if Product.VR_VOUCHER_1000 in state.position
                else 0
            )

            volcanic_rock_position = (
                state.position[Product.VOLCANICROCK]
                if Product.VOLCANICROCK in state.position
                else 0
            )
            # print(f"volcanic_rock_voucher_position: {volcanic_rock_voucher_position}")
            # print(f"volcanic_rock_position: {volcanic_rock_position}")
            volcanic_rock_order_depth = state.order_depths[Product.VOLCANICROCK]
            volcanic_rock_voucher_order_depth = state.order_depths[Product.VR_VOUCHER_1000]
            volcanic_rock_mid_price = (
                min(volcanic_rock_order_depth.buy_orders.keys())
                + max(volcanic_rock_order_depth.sell_orders.keys())
            ) / 2
            volcanic_rock_voucher_mid_price = self.get_volcanic_rock_voucher_mid_price(
                volcanic_rock_voucher_order_depth, traderObject[Product.VR_VOUCHER_1000]
            )
            tte = (
                self.params[Product.VR_VOUCHER_1000]["starting_time_to_expiry"]
                - (state.timestamp) / 1000000 / 7
            )
            volatility = BlackScholes.implied_volatility(
                volcanic_rock_voucher_mid_price,
                volcanic_rock_mid_price,
                self.params[Product.VR_VOUCHER_1000]["strike"],
                tte,
            )
            delta = BlackScholes.delta(
                volcanic_rock_mid_price,
                self.params[Product.VR_VOUCHER_1000]["strike"],
                tte,
                volatility,
            )

            volcanic_rock_voucher_take_orders, volcanic_rock_voucher_make_orders = (
                self.volcanic_rock_voucher_orders(
                    state.order_depths[Product.VR_VOUCHER_1000],
                    volcanic_rock_voucher_position,
                    traderObject[Product.VR_VOUCHER_1000],
                    volatility,
                )
            )

            volcanic_rock_orders = self.volcanic_rock_hedge_orders(
                state.order_depths[Product.VOLCANICROCK],
                state.order_depths[Product.VR_VOUCHER_1000],
                volcanic_rock_voucher_take_orders,
                volcanic_rock_position,
                volcanic_rock_voucher_position,
                delta,
            )

            if volcanic_rock_voucher_take_orders != None or volcanic_rock_voucher_make_orders != None:
                result[Product.VR_VOUCHER_1000] = (
                    volcanic_rock_voucher_take_orders + volcanic_rock_voucher_make_orders
                )
                # print(f"VR_VOUCHER_1000: {result[Product.VR_VOUCHER_1000]}")

            if volcanic_rock_orders != None:
                result[Product.VOLCANICROCK] = volcanic_rock_orders
                # print(f"VOLCANICROCK: {result[Product.VOLCANICROCK]}")





        conversions = 1
        traderData = jsonpickle.encode(traderObject)

        
        return result, conversions, traderData
