from datamodel import OrderDepth, UserId, TradingState, Order
from typing import List
import string
import jsonpickle # type: ignore
import numpy as np
import math
import random


class Product:
    KELP = "KELP"
    RAINFORESTRESIN = "RAINFOREST_RESIN"
    SQUIDINK = "SQUID_INK"


PARAMS = {
    Product.RAINFORESTRESIN: {
        "fair_value": 10000,
        "take_width": 1,
        "clear_width": 0,
        # for making
        "disregard_edge": 1,  # disregards orders for joining or pennying within this value from fair
        "join_edge": 2,  # joins orders within this edge
        "default_edge": 4,
        "soft_position_limit": 10,
    },
    Product.KELP: {
        "take_width": 1,
        "clear_width": 0,
        "prevent_adverse": True,
        "adverse_volume": 15,
        "reversion_beta": -0.23,
        "disregard_edge": 1,
        "join_edge": 0,
        "default_edge": 5,
    },
        Product.SQUIDINK: {
        "do_trade": True,
        "take_width": 1,            
        "clear_width": 1,           
        "prevent_adverse": True,    
        "adverse_volume": 15,       
        "reversion_beta": -0.1,
        "disregard_edge": 1,        
        "join_edge": 0,
        "default_edge": 1,
        "moving_average_window": 50,  
        "deviation_threshold": 0.01,
        "slope_threshold": 0.26
    },
}

#TODO: 
# 1. Merge the fucntion with the same structure, with one funciton
# 2. The SQUID INK price have a patter that gradually grow and a sudden drop, try to make a stratage based on that part


class Trader:
    def __init__(self, params=None):
        if params is None:
            params = PARAMS
        self.params = params

        self.LIMIT = {Product.RAINFORESTRESIN: 50, Product.KELP: 50, Product.SQUIDINK:20}
    def KELP_fair_value(self, order_depth: OrderDepth, traderObject) -> float:
        if len(order_depth.sell_orders) != 0 and len(order_depth.buy_orders) != 0:
            best_ask = min(order_depth.sell_orders.keys())
            best_bid = max(order_depth.buy_orders.keys())
            filtered_ask = [
                price
                for price in order_depth.sell_orders.keys()
                if abs(order_depth.sell_orders[price])
                >= self.params[Product.KELP]["adverse_volume"]
            ]
            filtered_bid = [
                price
                for price in order_depth.buy_orders.keys()
                if abs(order_depth.buy_orders[price])
                >= self.params[Product.KELP]["adverse_volume"]
            ]
            mm_ask = min(filtered_ask) if len(filtered_ask) > 0 else None
            mm_bid = max(filtered_bid) if len(filtered_bid) > 0 else None
            if mm_ask == None or mm_bid == None:
                if traderObject.get("kelp_last_price", None) == None:
                    mmmid_price = (best_ask + best_bid) / 2
                else:
                    mmmid_price = traderObject["kelp_last_price"]
            else:
                mmmid_price = (mm_ask + mm_bid) / 2

            if traderObject.get("kelp_last_price", None) != None:
                last_price = traderObject["kelp_last_price"]
                last_returns = (mmmid_price - last_price) / last_price
                pred_returns = (
                    last_returns * self.params[Product.KELP]["reversion_beta"]
                )
                fair = mmmid_price + (mmmid_price * pred_returns)
            else:
                fair = mmmid_price
            traderObject["kelp_last_price"] = mmmid_price
            return fair
        return None

    def SQUIDINK_fair_value(self, order_depth: OrderDepth, traderObject) -> float:
        if len(order_depth.sell_orders) != 0 and len(order_depth.buy_orders) != 0:
            best_ask = min(order_depth.sell_orders.keys())
            best_bid = max(order_depth.buy_orders.keys())

            filtered_ask = [
                price
                for price in order_depth.sell_orders.keys()
                if abs(order_depth.sell_orders[price])
                >= self.params[Product.SQUIDINK]["adverse_volume"]
            ]
            filtered_bid = [
                price
                for price in order_depth.buy_orders.keys()
                if abs(order_depth.buy_orders[price])
                >= self.params[Product.SQUIDINK]["adverse_volume"]
            ]
            
            mm_ask = min(filtered_ask) if len(filtered_ask) > 0 else None
            mm_bid = max(filtered_bid) if len(filtered_bid) > 0 else None
            
            if mm_ask == None or mm_bid == None:
                if traderObject.get("squidink_last_price", None) == None:
                    mmmid_price = (best_ask + best_bid) / 2
                else:
                    mmmid_price = traderObject["squidink_last_price"]
            else:
                mmmid_price = (mm_ask + mm_bid) / 2

            
            if traderObject.get("squidink_last_price", None) != None:
                last_price = traderObject["squidink_last_price"]
                last_returns = (mmmid_price - last_price) / last_price
                pred_returns = (
                    last_returns * self.params[Product.SQUIDINK]["reversion_beta"]
                )

                fair = mmmid_price + (mmmid_price * pred_returns)
            else:
                fair = mmmid_price
            traderObject["squidink_last_price"] = mmmid_price

            return fair
        return None

    def clear_position_order(
        self,
        product: str,
        fair_value: float,
        width: int,
        orders: List[Order],
        order_depth: OrderDepth,
        position: int,
        buy_order_volume: int,
        sell_order_volume: int,
    ) -> List[Order]:
        
        position_after_take = position + buy_order_volume - sell_order_volume
        fair_for_bid = round(fair_value - width)
        fair_for_ask = round(fair_value + width)

        buy_quantity = self.LIMIT[product] - (position + buy_order_volume)
        sell_quantity = self.LIMIT[product] + (position - sell_order_volume)

        if position_after_take > 0:
            # Aggregate volume from all buy orders with price greater than fair_for_ask
            clear_quantity = sum(
                volume
                for price, volume in order_depth.buy_orders.items()
                if price >= fair_for_ask
            )

            clear_quantity = min(clear_quantity, position_after_take)

            sent_quantity = min(sell_quantity, clear_quantity)

            if sent_quantity > 0:
                orders.append(Order(product, fair_for_ask, -abs(sent_quantity)))
                sell_order_volume += abs(sent_quantity)

        if position_after_take < 0:
            # Aggregate volume from all sell orders with price lower than fair_for_bid
            clear_quantity = sum(
                abs(volume)
                for price, volume in order_depth.sell_orders.items()
                if price <= fair_for_bid
            )
            clear_quantity = min(clear_quantity, abs(position_after_take))
            sent_quantity = min(buy_quantity, clear_quantity)
            if sent_quantity > 0:
                orders.append(Order(product, fair_for_bid, abs(sent_quantity)))
                buy_order_volume += abs(sent_quantity)

        return buy_order_volume, sell_order_volume

    def take_best_orders(
        self,
        product: str,
        fair_value: int,
        take_width: float,
        orders: List[Order],
        order_depth: OrderDepth,
        position: int,
        buy_order_volume: int,
        sell_order_volume: int,
        prevent_adverse: bool = False,
        adverse_volume: int = 0,
    ) -> (int, int): # type: ignore
        position_limit = self.LIMIT[product]

        if len(order_depth.sell_orders) != 0:
            best_ask = min(order_depth.sell_orders.keys())
            best_ask_amount = -1 * order_depth.sell_orders[best_ask]

            if not prevent_adverse or abs(best_ask_amount) <= adverse_volume:
                if best_ask + take_width <= fair_value:
                    quantity = min(
                        best_ask_amount, position_limit - position
                    )  # max amt to buy
                    if quantity > 0:
                        orders.append(Order(product, best_ask, quantity))
                        buy_order_volume += quantity
                        order_depth.sell_orders[best_ask] += quantity
                        if order_depth.sell_orders[best_ask] == 0:
                            del order_depth.sell_orders[best_ask]

        if len(order_depth.buy_orders) != 0:
            best_bid = max(order_depth.buy_orders.keys())
            best_bid_amount = order_depth.buy_orders[best_bid]

            if not prevent_adverse or abs(best_bid_amount) <= adverse_volume:
                if best_bid - take_width >= fair_value:
                    quantity = min(
                        best_bid_amount, position_limit + position
                    )  # should be the max we can sell
                    if quantity > 0:
                        orders.append(Order(product, best_bid, -1 * quantity))
                        sell_order_volume += quantity
                        order_depth.buy_orders[best_bid] -= quantity
                        if order_depth.buy_orders[best_bid] == 0:
                            del order_depth.buy_orders[best_bid]

        return buy_order_volume, sell_order_volume

    def market_make(
        self,
        product: str,
        orders: List[Order],
        bid: int,
        ask: int,
        position: int,
        buy_order_volume: int,
        sell_order_volume: int,
    ) -> (int, int):
        
        buy_quantity = self.LIMIT[product] - (position + buy_order_volume)
        
        if buy_quantity > 0:
            orders.append(Order(product, round(bid), buy_quantity))  # Buy order

        sell_quantity = self.LIMIT[product] + (position - sell_order_volume)
        if sell_quantity > 0:
            orders.append(Order(product, round(ask), -sell_quantity))  # Sell order
        return buy_order_volume, sell_order_volume

    def take_orders(
        self,
        product: str,
        order_depth: OrderDepth,
        fair_value: float,
        take_width: float,
        position: int,
        prevent_adverse: bool = False,
        adverse_volume: int = 0,
    ) -> (List[Order], int, int): # type: ignore
        orders: List[Order] = []
        
        buy_order_volume = 0
        sell_order_volume = 0

        buy_order_volume, sell_order_volume = self.take_best_orders(
            product,
            fair_value,
            take_width,
            orders,
            order_depth,
            position,
            buy_order_volume,
            sell_order_volume,
            prevent_adverse,
            adverse_volume,
        )
        return orders, buy_order_volume, sell_order_volume

    def clear_orders(
        self,
        product: str,
        order_depth: OrderDepth,
        fair_value: float,
        clear_width: int,
        position: int,
        buy_order_volume: int,
        sell_order_volume: int,
    ) -> (List[Order], int, int): # type: ignore
        orders: List[Order] = []
        buy_order_volume, sell_order_volume = self.clear_position_order(
            product,
            fair_value,
            clear_width,
            orders,
            order_depth,
            position,
            buy_order_volume,
            sell_order_volume,
        )
        return orders, buy_order_volume, sell_order_volume
    
    def make_orders(
    self,
    product,
    order_depth: OrderDepth,
    fair_value: float,
    position: int,
    buy_order_volume: int,
    sell_order_volume: int,
    disregard_edge: float,  # disregard trades within this edge for pennying or joining
    join_edge: float,       # join trades within this edge
    default_edge: float,    # default edge to request if there are no levels to penny or join
    manage_position: bool = False,
    soft_position_limit: int = 0,
    ):
        orders: List[Order] = []
        # Build lists of prices that are away from fair_value.
        asks_above_fair = [
            price
            for price in order_depth.sell_orders.keys()
            if price > fair_value + disregard_edge
        ]
        bids_below_fair = [
            price
            for price in order_depth.buy_orders.keys()
            if price < fair_value - disregard_edge
        ]

        best_ask_above_fair = min(asks_above_fair) if len(asks_above_fair) > 0 else None
        best_bid_below_fair = max(bids_below_fair) if len(bids_below_fair) > 0 else None

        # Set the initial ask and bid using default_edge.
       
        if best_ask_above_fair is not None:
            if abs(best_ask_above_fair - fair_value) <= join_edge:
                ask = best_ask_above_fair  # join orders at this level
            else:
                ask = best_ask_above_fair - 1  # undercut by one tick to penny
        else:
            ask = round(fair_value + default_edge)


        if best_bid_below_fair is not None:
            if abs(fair_value - best_bid_below_fair) <= join_edge:
                bid = best_bid_below_fair  # join orders on the bid side
            else:
                bid = best_bid_below_fair + 1  # add one tick on the bid side
        else:
            bid = round(fair_value - default_edge)

        if manage_position:
            if position > soft_position_limit:
                ask -= 1
            elif position < -soft_position_limit:
                bid += 1

        buy_order_volume, sell_order_volume = self.market_make(
            product,
            orders,
            bid,
            ask,
            position,
            buy_order_volume,
            sell_order_volume,
        )

        return orders, buy_order_volume, sell_order_volume

    def run(self, state: TradingState):
        traderObject = {}
        if state.traderData != None and state.traderData != "":
            traderObject = jsonpickle.decode(state.traderData)

        result = {}

        if Product.RAINFORESTRESIN in self.params and Product.RAINFORESTRESIN in state.order_depths:
            RAINFORESTRESIN_position = (
                state.position[Product.RAINFORESTRESIN]
                if Product.RAINFORESTRESIN in state.position
                else 0
            )
            RAINFORESTRESIN_take_orders, buy_order_volume, sell_order_volume = (
                self.take_orders(
                    Product.RAINFORESTRESIN,
                    state.order_depths[Product.RAINFORESTRESIN],
                    self.params[Product.RAINFORESTRESIN]["fair_value"],
                    self.params[Product.RAINFORESTRESIN]["take_width"],
                    RAINFORESTRESIN_position,
                )
            )
            RAINFORESTRESIN_clear_orders, buy_order_volume, sell_order_volume = (
                self.clear_orders(
                    Product.RAINFORESTRESIN,
                    state.order_depths[Product.RAINFORESTRESIN],
                    self.params[Product.RAINFORESTRESIN]["fair_value"],
                    self.params[Product.RAINFORESTRESIN]["clear_width"],
                    RAINFORESTRESIN_position,
                    buy_order_volume,
                    sell_order_volume,
                )
            )
            RAINFORESTRESIN_make_orders, _, _ = self.make_orders(
                Product.RAINFORESTRESIN,
                state.order_depths[Product.RAINFORESTRESIN],
                self.params[Product.RAINFORESTRESIN]["fair_value"],
                RAINFORESTRESIN_position,
                buy_order_volume,
                sell_order_volume,
                self.params[Product.RAINFORESTRESIN]["disregard_edge"],
                self.params[Product.RAINFORESTRESIN]["join_edge"],
                self.params[Product.RAINFORESTRESIN]["default_edge"],
                True,
                self.params[Product.RAINFORESTRESIN]["soft_position_limit"],
            )
            result[Product.RAINFORESTRESIN] = (
                RAINFORESTRESIN_take_orders + RAINFORESTRESIN_clear_orders + RAINFORESTRESIN_make_orders
            )

        if Product.KELP in self.params and Product.KELP in state.order_depths:
            KELP_position = (
                state.position[Product.KELP]
                if Product.KELP in state.position
                else 0
            )
            KELP_fair_value = self.KELP_fair_value(
                state.order_depths[Product.KELP], traderObject
            )
            KELP_take_orders, buy_order_volume, sell_order_volume = (
                self.take_orders(
                    Product.KELP,
                    state.order_depths[Product.KELP],
                    KELP_fair_value,
                    self.params[Product.KELP]["take_width"],
                    KELP_position,
                    self.params[Product.KELP]["prevent_adverse"],
                    self.params[Product.KELP]["adverse_volume"],
                )
            )
            KELP_clear_orders, buy_order_volume, sell_order_volume = (
                self.clear_orders(
                    Product.KELP,
                    state.order_depths[Product.KELP],
                    KELP_fair_value,
                    self.params[Product.KELP]["clear_width"],
                    KELP_position,
                    buy_order_volume,
                    sell_order_volume,
                )
            )
            KELP_make_orders, _, _ = self.make_orders(
                Product.KELP,
                state.order_depths[Product.KELP],
                KELP_fair_value,
                KELP_position,
                buy_order_volume,
                sell_order_volume,
                self.params[Product.KELP]["disregard_edge"],
                self.params[Product.KELP]["join_edge"],
                self.params[Product.KELP]["default_edge"],
            )
            result[Product.KELP] = (
                KELP_take_orders + KELP_clear_orders + KELP_make_orders
            )

        if Product.SQUIDINK in self.params and Product.SQUIDINK in state.order_depths and self.params[Product.SQUIDINK]["do_trade"]:
            
            # Get the current position and compute a fair value
            SQUIDINK_position = state.position.get(Product.SQUIDINK, 0)
            SQUIDINK_fair_value = self.SQUIDINK_fair_value(
                state.order_depths[Product.SQUIDINK], traderObject
            )
            
            # --- Maintain a history of recent prices ---
            if not hasattr(self, 'squidink_recent_prices'):
                self.squidink_recent_prices = []
            self.squidink_recent_prices.append(SQUIDINK_fair_value)
            
            # Use the moving average window parameter (e.g., 100 or 1000, adjust as needed)
            moving_average_window = self.params[Product.SQUIDINK].get("moving_average_window", 100)
            if len(self.squidink_recent_prices) > moving_average_window:
                self.squidink_recent_prices.pop(0)
            
            # Calculate the moving average & deviation
            moving_average = sum(self.squidink_recent_prices) / len(self.squidink_recent_prices)
            deviation = SQUIDINK_fair_value - moving_average
            normalized_deviation = deviation / moving_average if moving_average != 0 else 0
            deviation_threshold = self.params[Product.SQUIDINK]["deviation_threshold"]
            

            # --- New: Compute the recent price slope ---
            # Only compute slope if we have a full window of data.
            if len(self.squidink_recent_prices) >= moving_average_window:
                x = np.arange(moving_average_window)
                prices_window = np.array(self.squidink_recent_prices[-moving_average_window:])
                slope, _ = np.polyfit(x, prices_window, 1)
            else:
                slope = 0.0  # if not enough data, default to 0

            slope_threshold = self.params[Product.SQUIDINK].get("slope_threshold")

            delta = 0.01  # small adjustment factor
            
            if normalized_deviation > deviation_threshold and slope < slope_threshold:
                adjusted_fair_value = SQUIDINK_fair_value * (1 - delta)
            elif normalized_deviation < -deviation_threshold and slope > -slope_threshold:
                adjusted_fair_value = SQUIDINK_fair_value * (1 + delta)
            else:
                # Otherwise, use the original fair value (or combine both adjustments as needed)
                adjusted_fair_value = SQUIDINK_fair_value
            
            # --- Order placement using the adjusted fair value ---
            SQUIDINK_take_orders, buy_order_volume, sell_order_volume = self.take_orders(
                Product.SQUIDINK,
                state.order_depths[Product.SQUIDINK],
                adjusted_fair_value,
                self.params[Product.SQUIDINK]["take_width"], 
                SQUIDINK_position,
                self.params[Product.SQUIDINK]["prevent_adverse"],
                self.params[Product.SQUIDINK]["adverse_volume"],
            )
            
            SQUIDINK_clear_orders, buy_order_volume, sell_order_volume = self.clear_orders(
                Product.SQUIDINK,
                state.order_depths[Product.SQUIDINK],
                adjusted_fair_value,
                self.params[Product.SQUIDINK]["clear_width"],
                SQUIDINK_position,
                buy_order_volume,
                sell_order_volume,
            )
            
            SQUIDINK_make_orders, _, _ = self.make_orders(
                Product.SQUIDINK,
                state.order_depths[Product.SQUIDINK],
                adjusted_fair_value,
                SQUIDINK_position,
                buy_order_volume,
                sell_order_volume,
                self.params[Product.SQUIDINK]["disregard_edge"],
                self.params[Product.SQUIDINK]["join_edge"],
                self.params[Product.SQUIDINK]["default_edge"],
            )
            
            result[Product.SQUIDINK] = (
                SQUIDINK_take_orders + SQUIDINK_clear_orders + SQUIDINK_make_orders
            )

        conversions = 1
        traderData = jsonpickle.encode(traderObject)

        return result, conversions, traderData
