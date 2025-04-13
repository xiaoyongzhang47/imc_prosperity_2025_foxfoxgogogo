from datamodel import OrderDepth, UserId, TradingState, Order
from typing import List
import string 
import jsonpickle # type: ignore
import numpy as np
import math
import random
import time

class Product:
    KELP            = "KELP"
    RAINFORESTRESIN = "RAINFOREST_RESIN"
    SQUIDINK        = "SQUID_INK"

    CROISSANTS      = "CROISSANTS"
    DJEMBES         = "DJEMBES"
    JAMS            = "JAMS"

    PICNICBASKET1   = "PICNIC_BASKET1"
    PICNICBASKET2   = "PICNIC_BASKET2"


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
        "reversion_beta": -0.28,
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
    Product.CROISSANTS: {
        "take_width": 1,            
        "clear_width": 1,           
        "prevent_adverse": True,    
        "adverse_volume": 85,       
        "reversion_beta":  -0.0739,
        "disregard_edge": 1,        
        "join_edge": 0,
        "default_edge": 1,
        },
    Product.DJEMBES: {
        "take_width": 1,            
        "clear_width": 1,           
        "prevent_adverse": True,    
        "adverse_volume": 130,       
        "reversion_beta":  0.0618,
        "disregard_edge": 1,        
        "join_edge": 0,
        "default_edge": 1,
        },
    Product.JAMS: {
        "take_width": 1,            
        "clear_width": 1,           
        "prevent_adverse": True,    
        "adverse_volume": 120,       
        "reversion_beta": 0.001,
        "disregard_edge": 1,        
        "join_edge": 0,
        "default_edge": 1,
        },
    Product.PICNICBASKET1: {
        "take_width": 1,            
        "clear_width": 1,           
        "prevent_adverse": True,    
        "adverse_volume": 10,       
        "reversion_beta": 0.0351,
        "disregard_edge": 1,        
        "join_edge": 0,
        "default_edge": 1,


        "alpha":853.31,
        "beta" :0.98641,
        "mean_spread":-2.69245,
        "std_spread":85.4813,
        "entry_threshold":0.5,
        "exit_threshould":0.5
        },
    Product.PICNICBASKET2: {
        "take_width": 1,            
        "clear_width": 1,           
        "prevent_adverse": True,    
        "adverse_volume": 25,       
        "reversion_beta": 0.0345,
        "disregard_edge": 1,        
        "join_edge": 0,
        "default_edge": 1,

        "alpha":-2076.33,
        "beta" :1.07114,
        "mean_spread":-54.518,
        "std_spread":62.0239,
        "entry_threshold":1.5,
        "exit_threshould":0.5
        },
    }


class Trader:
    def __init__(self, params=None):
        if params is None:
            params = PARAMS
        self.params = params

        self.LIMIT = {
            Product.RAINFORESTRESIN: 50, 
            Product.KELP:            50, 
            Product.SQUIDINK:        50,
            Product.CROISSANTS:     250,
            Product.JAMS:           350,
            Product.DJEMBES:         60,
            Product.PICNICBASKET1:   60,
            Product.PICNICBASKET2:  100,
            }

    # ------
    # Round 1 fair values
    # ------

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

    # ------
    # Round 2 fair values
    # ------

    def fair_value(self, product:str,order_depth: OrderDepth, traderObject) -> float:
        
        item_last_price = product.lower()+"_last_price"

        if len(order_depth.sell_orders) != 0 and len(order_depth.buy_orders) != 0:
            best_ask = min(order_depth.sell_orders.keys())
            best_bid = max(order_depth.buy_orders.keys())
            filtered_ask = [
                price
                for price in order_depth.sell_orders.keys()
                if abs(order_depth.sell_orders[price])
                >= self.params[product]["adverse_volume"]
            ]
            filtered_bid = [
                price
                for price in order_depth.buy_orders.keys()
                if abs(order_depth.buy_orders[price])
                >= self.params[product]["adverse_volume"]
            ]
            mm_ask = min(filtered_ask) if len(filtered_ask) > 0 else None
            mm_bid = max(filtered_bid) if len(filtered_bid) > 0 else None
            if mm_ask == None or mm_bid == None:

                
                if traderObject.get(item_last_price, None) == None:
                    mmmid_price = (best_ask + best_bid) / 2
                else:
                    mmmid_price = traderObject[item_last_price]
            else:
                mmmid_price = (mm_ask + mm_bid) / 2

            if traderObject.get(item_last_price, None) != None:
                last_price = traderObject[item_last_price]
                last_returns = (mmmid_price - last_price) / last_price
                pred_returns = (
                    last_returns * self.params[product]["reversion_beta"]
                )
                fair = mmmid_price + (mmmid_price * pred_returns)
            else:
                fair = mmmid_price
            traderObject[product] = mmmid_price
            return fair
        return None

    def is_good_to_trade(self, product:str, state: TradingState):
        isgood = product in self.params and product in state.order_depths
        return isgood

    # ------
    # Round 1 functions
    # ------

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
            buy_order_volume += buy_quantity
            orders.append(Order(product, round(bid), buy_quantity))  # Buy order

        sell_quantity = self.LIMIT[product] + (position - sell_order_volume)

        if sell_quantity > 0:
            sell_order_volume += sell_quantity
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
    
    # --------------------
    # PAIRS TRADING FUNCTIONS
    # --------------------
    def pair_trading_best_sells(
        self,
        product: str,
        fair_value: float,
        sell_width: float,
        orders: List[Order],
        order_depth: OrderDepth,
        position: int,
        target_sell_order_volume: int,
        sell_order_volume: int,
        prevent_adverse: bool = False,
        adverse_volume: int = 0,
    ) -> (int, int): # type: ignore
        # Here we fills all bids that falls in the region [fair - sell_width, fair]
        # which are not covered by the best_take orders 

        position_limit = self.LIMIT[product]

        if len(order_depth.buy_orders) != 0:
            best_bid = max(order_depth.buy_orders.keys())
            best_bid_amount = order_depth.buy_orders[best_bid]

            if not prevent_adverse or abs(best_bid_amount) <= adverse_volume:
                # slightly beyong the fair_value
                if best_bid + sell_width >= fair_value:
                    quantity = min(
                        target_sell_order_volume,
                        best_bid_amount, 
                        position_limit + position
                    )  # should be the max we can sell
                    if quantity > 0:
                        orders.append(Order(product, best_bid, -1 * quantity))
                        sell_order_volume += quantity
                        order_depth.buy_orders[best_bid] -= quantity
                        if order_depth.buy_orders[best_bid] == 0:
                            del order_depth.buy_orders[best_bid]

        return sell_order_volume
    
    def pair_trading_best_buys(
        self,
        product: str,
        fair_value: float,
        buy_width: float,
        orders: List[Order],
        order_depth: OrderDepth,
        position: int,
        target_buy_order_volume: int,
        buy_order_volume:int,
        prevent_adverse: bool = False,
        adverse_volume: int = 0,
    ) -> (int, int): # type: ignore
        
        position_limit = self.LIMIT[product]
        # Here we fills all asks that falls in the region [fair, fair + buy_width]
        # which are not covered by the best_take orders 
        
        if len(order_depth.sell_orders) != 0:
            best_ask = min(order_depth.sell_orders.keys())
            best_ask_amount = -1 * order_depth.sell_orders[best_ask]
            
            if not prevent_adverse or abs(best_ask_amount) <= adverse_volume:
                if best_ask - buy_width <= fair_value:
                    quantity = min(
                        best_ask_amount,
                        target_buy_order_volume,
                        position_limit - position
                    )  # max allowed amt to buy

                    if quantity > 0:
                        orders.append(Order(product, best_ask, quantity))
                        buy_order_volume += quantity
                        order_depth.sell_orders[best_ask] += quantity
                        if order_depth.sell_orders[best_ask] == 0:
                            del order_depth.sell_orders[best_ask]
        return buy_order_volume

    def pair_trading_orders(
        self,
        order_type: str,
        product: str,
        target_volume: float,
        order_depth: OrderDepth,
        fair_value: float,
        width: float,
        position: int,
        prevent_adverse: bool = False,
        adverse_volume: int = 0,
    ) -> (List[Order], int, int): # type: ignore
        
        orders: List[Order] = []
        
        buy_order_volume = 0
        sell_order_volume = 0

        if order_type == 'buy':
            buy_order_volume = self.pair_trading_best_buys(
                product=product,
                fair_value=fair_value,
                buy_width=width,
                orders=orders,
                order_depth=order_depth,
                position=position,
                buy_order_volume=buy_order_volume,
                target_buy_order_volume=target_volume,
                prevent_adverse=prevent_adverse,
                adverse_volume=adverse_volume
                )
            
        elif order_type == 'sell':
            sell_order_volume = self.pair_trading_best_sells(
                product=product,
                fair_value=fair_value,
                sell_width=width,
                orders=orders,
                order_depth=order_depth,
                position=position,
                sell_order_volume=sell_order_volume,
                target_sell_order_volume=target_volume,
                prevent_adverse=prevent_adverse,
                adverse_volume=adverse_volume,
            )
        else:
            print("what are you talking about ???")

        return orders, buy_order_volume, sell_order_volume

    def run(self, state: TradingState):
        traderObject = {}
        if state.traderData != None and state.traderData != "":
            traderObject = jsonpickle.decode(state.traderData)

        result = {}

        # --------------------
        # RAINFORESTRESIN
        # --------------------

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

        # --------------------
        # KELP
        # --------------------

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

        # --------------------
        # FKING SQUID_INK
        # --------------------
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

            # Optionally save deviation history (for debugging or offline analysis)
            # if not hasattr(self, 'slop_fun'):
            #     self.slop_fun = []
                
            # self.slop_fun.append(slope)
            # np.savetxt('slope2.csv', self.slop_fun, delimiter=',', fmt='%f')
            
            
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

        # --------------------
        # PICNICBASKET1 / PICNICBASKET2 PairTrading
        # --------------------

        croissants_are_good = self.is_good_to_trade(Product.CROISSANTS, state)
        jams_are_good = self.is_good_to_trade(Product.JAMS, state)
        djembs_are_good = self.is_good_to_trade(Product.DJEMBES, state)

        croissant_solo_trading: List[Order] =[]
        jams_solo_trading: List[Order] = []
        djembes_solo_trading: List[Order] = []

        
        allow_solo_trading = random.random() < 0.7
        # tracks their position as it

        CROISSANTS_position = 0
        JAMS_position = 0
        DJEMBES_position = 0

        if allow_solo_trading:
            if croissants_are_good:
                CROISSANTS_position = (
                    state.position[Product.CROISSANTS]
                    if Product.CROISSANTS in state.position
                    else 0
                )
                CROISSANTS_fair_value = self.fair_value(
                    Product.CROISSANTS, state.order_depths[Product.CROISSANTS], traderObject
                )


                CROISSANTS_take_orders, buy_order_volume, sell_order_volume = (
                    self.take_orders(
                        Product.CROISSANTS,
                        state.order_depths[Product.CROISSANTS],
                        CROISSANTS_fair_value,
                        self.params[Product.CROISSANTS]["take_width"],
                        CROISSANTS_position,
                        self.params[Product.CROISSANTS]["prevent_adverse"],
                        self.params[Product.CROISSANTS]["adverse_volume"],
                    )
                )

                

                CROISSANTS_clear_orders, buy_order_volume, sell_order_volume = (
                    self.clear_orders(
                        Product.CROISSANTS,
                        state.order_depths[Product.CROISSANTS],
                        CROISSANTS_fair_value,
                        self.params[Product.CROISSANTS]["clear_width"],
                        CROISSANTS_position,
                        buy_order_volume,
                        sell_order_volume,
                    )
                )

                
                
                CROISSANTS_make_orders, buy_order_volume, sell_order_volume = self.make_orders(
                    Product.CROISSANTS,
                    state.order_depths[Product.CROISSANTS],
                    CROISSANTS_fair_value,
                    CROISSANTS_position,
                    buy_order_volume,
                    sell_order_volume,
                    self.params[Product.CROISSANTS]["disregard_edge"],
                    self.params[Product.CROISSANTS]["join_edge"],
                    self.params[Product.CROISSANTS]["default_edge"],
                )

                CROISSANTS_position += buy_order_volume
                CROISSANTS_position -= sell_order_volume

                

                croissant_solo_trading =  (
                    CROISSANTS_take_orders + CROISSANTS_clear_orders + CROISSANTS_make_orders
                )

                

            if jams_are_good:
                JAMS_position = (
                    state.position[Product.JAMS]
                    if Product.JAMS in state.position
                    else 0
                )
                JAMS_fair_value = self.fair_value(
                    Product.JAMS, state.order_depths[Product.JAMS], traderObject
                )

                JAMS_take_orders, buy_order_volume, sell_order_volume = (
                    self.take_orders(
                        Product.JAMS,
                        state.order_depths[Product.JAMS],
                        JAMS_fair_value,
                        self.params[Product.JAMS]["take_width"],
                        JAMS_position,
                        self.params[Product.JAMS]["prevent_adverse"],
                        self.params[Product.JAMS]["adverse_volume"],
                    )
                )
                JAMS_clear_orders, buy_order_volume, sell_order_volume = (
                    self.clear_orders(
                        Product.JAMS,
                        state.order_depths[Product.JAMS],
                        JAMS_fair_value,
                        self.params[Product.JAMS]["clear_width"],
                        JAMS_position,
                        buy_order_volume,
                        sell_order_volume,
                    )
                )
                
                JAMS_make_orders, buy_order_volume, sell_order_volume = self.make_orders(
                    Product.JAMS,
                    state.order_depths[Product.JAMS],
                    JAMS_fair_value,
                    JAMS_position,
                    buy_order_volume,
                    sell_order_volume,
                    self.params[Product.JAMS]["disregard_edge"],
                    self.params[Product.JAMS]["join_edge"],
                    self.params[Product.JAMS]["default_edge"],
                )

                JAMS_position += buy_order_volume
                JAMS_position -= sell_order_volume

                jams_solo_trading =  (
                    JAMS_take_orders + JAMS_clear_orders + JAMS_make_orders
                )

            if djembs_are_good:
                DJEMBES_position = (
                    state.position[Product.DJEMBES]
                    if Product.DJEMBES in state.position
                    else 0
                )
                DJEMBES_fair_value = self.fair_value(
                    Product.DJEMBES, state.order_depths[Product.DJEMBES], traderObject
                )

                DJEMBES_take_orders, buy_order_volume, sell_order_volume = (
                    self.take_orders(
                        Product.DJEMBES,
                        state.order_depths[Product.DJEMBES],
                        DJEMBES_fair_value,
                        self.params[Product.DJEMBES]["take_width"],
                        DJEMBES_position,
                        self.params[Product.DJEMBES]["prevent_adverse"],
                        self.params[Product.DJEMBES]["adverse_volume"],
                    )
                )
                DJEMBES_clear_orders, buy_order_volume, sell_order_volume = (
                    self.clear_orders(
                        Product.DJEMBES,
                        state.order_depths[Product.DJEMBES],
                        DJEMBES_fair_value,
                        self.params[Product.DJEMBES]["clear_width"],
                        DJEMBES_position,
                        buy_order_volume,
                        sell_order_volume,
                    )
                )
        
                DJEMBES_make_orders, buy_order_volume, sell_order_volume = self.make_orders(
                    Product.DJEMBES,
                    state.order_depths[Product.DJEMBES],
                    DJEMBES_fair_value,
                    DJEMBES_position,
                    buy_order_volume,
                    sell_order_volume,
                    self.params[Product.DJEMBES]["disregard_edge"],
                    self.params[Product.DJEMBES]["join_edge"],
                    self.params[Product.DJEMBES]["default_edge"],
                )

                DJEMBES_position += buy_order_volume
                DJEMBES_position -= sell_order_volume

                djembes_solo_trading =  (
                    DJEMBES_take_orders + DJEMBES_clear_orders + DJEMBES_make_orders
                )

        every_thing_is_good = croissants_are_good and jams_are_good and djembs_are_good
        picnicbask1_is_good = self.is_good_to_trade(Product.PICNICBASKET1, state)
        picnicbask2_is_good = self.is_good_to_trade(Product.PICNICBASKET2, state)

        
        
        # --------------------
        # Pair: PICNICBASKET1 = 6 * CROISSANTS + 3 * JAMS + DJEMB  
        # Pair: PICNICBASKET2 = 4 * CROISSANTS + 2 * JAMS
        # --------------------

        pb_strength = 1
        prod_strength = 0.3

        wd_coef = 10
        wd_coef2 = 5

        if every_thing_is_good:
            # get positions

            if not allow_solo_trading:
                CROISSANTS_position = (
                    state.position[Product.CROISSANTS]
                    if Product.CROISSANTS in state.position
                    else 0
                )

                JAMS_position = (
                    state.position[Product.JAMS]
                    if Product.JAMS in state.position
                    else 0
                )

                DJEMBES_position =(
                    state.position[Product.DJEMBES]
                    if Product.DJEMBES in state.position
                    else 0
                )

            # get fair values
            CROISSANTS_fair_value   = self.fair_value(Product.CROISSANTS,
                state.order_depths[Product.CROISSANTS], traderObject
            )

            JAMS_fair_value   = self.fair_value(Product.JAMS,
                state.order_depths[Product.JAMS], traderObject
            )

            DJEMBES_fair_value   = self.fair_value(Product.DJEMBES,
                state.order_depths[Product.DJEMBES], traderObject
            )


            UNITE_VOLUME = 5

            if picnicbask1_is_good:
                # do pair trading with basket 1
                # get position
                PICNICBASK1_position = (
                    state.position[Product.PICNICBASKET1]
                    if Product.PICNICBASKET1 in state.position
                    else 0
                )
                # get fair value
                PICNICBASKET1_fair_value  = self.fair_value(Product.PICNICBASKET1,
                    state.order_depths[Product.PICNICBASKET1], traderObject
                )

                picnicbask1_solo_trading:List[Order] = []

                #-----------------
                # Pairing Trading
                #-----------------

                combined_value = 6 * CROISSANTS_fair_value + 3 * JAMS_fair_value + DJEMBES_fair_value

                # Retrieve the cointegration regression parameters for the pair.
                # mid_picnicbask1 = alpha + beta * mid_croissants + residual
                params_pair     = self.params.get(Product.PICNICBASKET1, {})
                alpha           = params_pair.get("alpha")             # regression intercept
                beta            = params_pair.get("beta")              # regression slope
                mean_spread     = params_pair.get("mean_spread")
                std_spread      = params_pair.get("std_spread")          # avoid division by zero
                entry_threshold = params_pair.get("entry_threshold")  # typically a z-score of 1 or 2

                # Calculate the current spread.
                # residual (spread) = mid_picnicbask1 - (alpha + beta * mid_croissants)
                spread = PICNICBASKET1_fair_value - (alpha + beta * combined_value)
                zscore = (spread - mean_spread) / std_spread

                # Optionally save deviation history (for debugging or offline analysis)
                # if not hasattr(self, 'save_list'):
                #     self.save_list = []
                    
                # self.save_list.append(zscore)
                # np.savetxt('zscore3.csv', self.save_list, delimiter=',', fmt='%f')


                CROISSANTS_pt_orders: List[Order] = []
                JAMS_pt_orders: List[Order] = []
                DJEMBES_pt_orders: List[Order] = []
                PICNICBASKET1_orders: List[Order] = []

                # PICNICBASKET1_buy_order_volume = 0
                # PICNICBASKET1_sell_order_volume = 0

                # abs_price_diff = PICNICBASKET1_fair_value - 
                
                sell_width   = wd_coef * abs(zscore)
                buy_width    = wd_coef * abs(zscore)
                if zscore > entry_threshold:
                    # The spread is too wide – PICNICBASKET1 appears overvalued compared to CROISSANTS.
                    # Strategy: Sell PICNICBASKET1 and buy CROISSANTS.

                    PICNICBASKET1_buy_order_volume = 0
                    PICNICBASKET1_sell_order_volume = 0

                    if random.uniform(0, 1) < pb_strength:

                        PICNICBASKET1_pt_orders, PICNICBASKET1_buy_order_volume,PICNICBASKET1_sell_order_volume = (
                            self.pair_trading_orders(
                                order_type='sell',
                                product=Product.PICNICBASKET1,
                                target_volume=UNITE_VOLUME,
                                order_depth=state.order_depths[Product.PICNICBASKET1],
                                fair_value=PICNICBASKET1_fair_value,
                                width=sell_width,
                                position= PICNICBASK1_position
                            ))
                        
                        PICNICBASK1_position -= sell_order_volume
                        PICNICBASKET1_orders += PICNICBASKET1_pt_orders
                    
                    CROISSANTS_target_buy_volume   = 6 * PICNICBASKET1_sell_order_volume
                    JAMS_target_buy_volume         = 3 * PICNICBASKET1_sell_order_volume
                    DJEMBES_target_buy_volume      = 1 * PICNICBASKET1_sell_order_volume

                    if random.uniform(0, 1) < prod_strength:
                        CROISSANTS_pt_orders, CROISSANTS_buy_order_volume, CROISSANTS_sell_order_volume = (
                            self.pair_trading_orders(
                                order_type='buy',
                                product=Product.CROISSANTS,
                                target_volume=CROISSANTS_target_buy_volume,
                                order_depth=state.order_depths[Product.CROISSANTS],
                                fair_value=CROISSANTS_fair_value,
                                width = buy_width,
                                position =CROISSANTS_position,
                                ))
                        
                        JAMS_pt_orders, JAMS_buy_order_volume, JAMS_sell_order_volume = (
                            self.pair_trading_orders(
                                order_type='buy',
                                product=Product.JAMS,
                                target_volume=JAMS_target_buy_volume,
                                order_depth=state.order_depths[Product.JAMS],
                                fair_value=JAMS_fair_value,
                                width = buy_width,
                                position =JAMS_position,
                                ))
                        
                        DJEMBES_pt_orders, DJEMBES_buy_order_volume, DJEMBES_sell_order_volume = (
                            self.pair_trading_orders(
                                order_type='buy',
                                product=Product.DJEMBES,
                                target_volume=DJEMBES_target_buy_volume,
                                order_depth=state.order_depths[Product.DJEMBES],
                                fair_value=DJEMBES_fair_value,
                                width = buy_width,
                                position =DJEMBES_position,
                                ))
                        
                        CROISSANTS_position += CROISSANTS_buy_order_volume
                        JAMS_position       += JAMS_buy_order_volume
                        DJEMBES_position    += DJEMBES_buy_order_volume

                        croissant_solo_trading   += (CROISSANTS_pt_orders)
                        jams_solo_trading        += (JAMS_pt_orders)
                        djembes_solo_trading     += (DJEMBES_pt_orders)
                    
                elif zscore < -entry_threshold:
                    # The spread is too low – PICNICBASKET1 appears undervalued relative to CROISSANTS.
                    # Strategy: Buy PICNICBASKET1 and sell CROISSANTS.
                    PICNICBASKET1_buy_order_volume = 0
                    PICNICBASKET1_sell_order_volume= 0
                    
                    if random.uniform(0, 1) < pb_strength:

                        PICNICBASKET1_pt_orders, PICNICBASKET1_buy_order_volume,PICNICBASKET1_sell_order_volume = (
                            self.pair_trading_orders(
                                order_type='buy',
                                product=Product.PICNICBASKET1,
                                target_volume=UNITE_VOLUME,
                                order_depth=state.order_depths[Product.PICNICBASKET1],
                                fair_value=PICNICBASKET1_fair_value,
                                width=sell_width,
                                position= PICNICBASK1_position
                            ))
                        
                        PICNICBASK1_position += buy_order_volume
                        PICNICBASKET1_orders += PICNICBASKET1_pt_orders

                    CROISSANTS_target_sell_volume = 6 * PICNICBASKET1_buy_order_volume
                    JAMS_target_sell_volume       = 3 * PICNICBASKET1_buy_order_volume
                    DJEMBES_target_sell_volume    = 1 * PICNICBASKET1_buy_order_volume

                    if random.uniform(0, 1) < prod_strength:
                        CROISSANTS_pt_orders, CROISSANTS_buy_order_volume, CROISSANTS_sell_order_volume = (
                            self.pair_trading_orders(
                                order_type='sell',
                                product=Product.CROISSANTS,
                                target_volume=CROISSANTS_target_sell_volume,
                                order_depth=state.order_depths[Product.CROISSANTS],
                                fair_value=CROISSANTS_fair_value,
                                width = buy_width,
                                position =CROISSANTS_position,
                                ))
                        
                        JAMS_pt_orders, JAMS_buy_order_volume, JAMS_sell_order_volume = (
                            self.pair_trading_orders(
                                order_type='sell',
                                product=Product.JAMS,
                                target_volume=JAMS_target_sell_volume,
                                order_depth=state.order_depths[Product.JAMS],
                                fair_value=JAMS_fair_value,
                                width = buy_width,
                                position =JAMS_position,
                                ))
                        
                        DJEMBES_pt_orders, DJEMBES_buy_order_volume, DJEMBES_sell_order_volume = (
                            self.pair_trading_orders(
                                order_type='sell',
                                product=Product.DJEMBES,
                                target_volume=DJEMBES_target_sell_volume,
                                order_depth=state.order_depths[Product.DJEMBES],
                                fair_value=DJEMBES_fair_value,
                                width = buy_width,
                                position =DJEMBES_position,
                                ))
                        
                        CROISSANTS_position -= CROISSANTS_sell_order_volume
                        JAMS_position       -= JAMS_sell_order_volume
                        DJEMBES_position    -= DJEMBES_sell_order_volume
                        
                        croissant_solo_trading   += (CROISSANTS_pt_orders)
                        jams_solo_trading        += (JAMS_pt_orders)
                        djembes_solo_trading     += (DJEMBES_pt_orders)
                

                result[Product.PICNICBASKET1] = (PICNICBASKET1_orders + picnicbask1_solo_trading)

            if picnicbask2_is_good:
                # do pair trading with basket 1
                # get position

                PICNICBASK2_position = (
                    state.position[Product.PICNICBASKET2]
                    if Product.PICNICBASKET2 in state.position
                    else 0
                )
                # get fair value
                PICNICBASKET2_fair_value  = self.fair_value(Product.PICNICBASKET2,
                    state.order_depths[Product.PICNICBASKET2], traderObject
                )

                combined_value = 4 * CROISSANTS_fair_value + 2 * JAMS_fair_value

                # Retrieve the cointegration regression parameters for the pair.
                # mid_picnicbask1 = alpha + beta * mid_croissants + residual
                params_pair     = self.params.get(Product.PICNICBASKET2, {})
                alpha           = params_pair.get("alpha")             # regression intercept
                beta            = params_pair.get("beta")              # regression slope
                mean_spread     = params_pair.get("mean_spread")
                std_spread      = params_pair.get("std_spread")          # avoid division by zero
                entry_threshold = params_pair.get("entry_threshold")  # typically a z-score of 1 or 2

                # Calculate the current spread.
                # residual (spread) = mid_picnicbask1 - (alpha + beta * mid_croissants)
                spread = PICNICBASKET2_fair_value - (alpha + beta * combined_value)
                zscore = (spread - mean_spread) / std_spread

                # Optionally save deviation history (for debugging or offline analysis)
                # if not hasattr(self, 'save_list'):
                #     self.save_list = []
                    
                # self.save_list.append(zscore)
                # np.savetxt('zscoreB1.csv', self.save_list, delimiter=',', fmt='%f')

                CROISSANTS_pt_orders: List[Order] = []
                JAMS_pt_orders: List[Order] = []
                DJEMBES_pt_orders: List[Order] = []
                PICNICBASKET2_orders: List[Order] = []

                # PICNICBASKET2_buy_order_volume = 0
                # PICNICBASKET2_sell_order_volume = 0


                
                sell_width   = wd_coef2 * abs(zscore)
                buy_width    = wd_coef2 * abs(zscore)
            
                if zscore > entry_threshold:
                    # The spread is too wide – PICNICBASKET2 appears overvalued compared to CROISSANTS.
                    # Strategy: Sell PICNICBASKET2 and buy CROISSANTS.

                    PICNICBASKET2_buy_order_volume = 0
                    PICNICBASKET2_sell_order_volume = 0

                    if random.uniform(0, 1) < pb_strength:

                        PICNICBASKET2_pt_orders, PICNICBASKET2_buy_order_volume,PICNICBASKET2_sell_order_volume = (
                            self.pair_trading_orders(
                                order_type='sell',
                                product=Product.PICNICBASKET2,
                                target_volume=UNITE_VOLUME,
                                order_depth=state.order_depths[Product.PICNICBASKET2],
                                fair_value=PICNICBASKET2_fair_value,
                                width=sell_width,
                                position= PICNICBASK2_position
                            ))
                        
                        PICNICBASK2_position -= PICNICBASKET2_sell_order_volume
                        PICNICBASKET2_orders += PICNICBASKET2_pt_orders
                    
                    
                    CROISSANTS_target_buy_volume = 4 * PICNICBASKET2_sell_order_volume
                    JAMS_target_buy_volume       = 2 * PICNICBASKET2_sell_order_volume
                    

                    if random.uniform(0, 1) < prod_strength:
                        CROISSANTS_pt_orders, CROISSANTS_buy_order_volume, CROISSANTS_sell_order_volume = (
                            self.pair_trading_orders(
                                order_type='buy',
                                product=Product.CROISSANTS,
                                target_volume=CROISSANTS_target_buy_volume,
                                order_depth=state.order_depths[Product.CROISSANTS],
                                fair_value=CROISSANTS_fair_value,
                                width = buy_width,
                                position =CROISSANTS_position,
                                ))
                        
                
                        JAMS_pt_orders, JAMS_buy_order_volume, JAMS_sell_order_volume = (
                            self.pair_trading_orders(
                                order_type='buy',
                                product=Product.JAMS,
                                target_volume=JAMS_target_buy_volume,
                                order_depth=state.order_depths[Product.JAMS],
                                fair_value=JAMS_fair_value,
                                width = buy_width,
                                position =JAMS_position,
                                ))
                        
                        CROISSANTS_position += CROISSANTS_buy_order_volume
                        JAMS_position       += JAMS_buy_order_volume
                        
                        croissant_solo_trading += CROISSANTS_pt_orders
                        jams_solo_trading      += JAMS_pt_orders
                        
                    
                elif zscore < -entry_threshold:
                    # The spread is too low – PICNICBASKET2 appears undervalued relative to CROISSANTS.
                    # Strategy: Buy PICNICBASKET2 and sell CROISSANTS.
                    PICNICBASKET2_buy_order_volume = 0
                    PICNICBASKET2_sell_order_volume= 0
                    
                    if random.uniform(0, 1) < pb_strength:
                        PICNICBASKET2_pt_orders, PICNICBASKET2_buy_order_volume,PICNICBASKET2_sell_order_volume = (
                            self.pair_trading_orders(
                                order_type='buy',
                                product=Product.PICNICBASKET2,
                                target_volume=UNITE_VOLUME,
                                order_depth=state.order_depths[Product.PICNICBASKET2],
                                fair_value=PICNICBASKET2_fair_value,
                                width=sell_width,
                                position= PICNICBASK1_position
                            ))
                        
                        PICNICBASK1_position += PICNICBASKET2_buy_order_volume
                        PICNICBASKET2_orders += PICNICBASKET2_pt_orders
                    
                    CROISSANTS_target_sell_volume = 4 * PICNICBASKET2_buy_order_volume
                    JAMS_target_sell_volume       = 2 * PICNICBASKET2_buy_order_volume

                    if random.uniform(0, 1) < prod_strength:
                        CROISSANTS_pt_orders, CROISSANTS_buy_order_volume, CROISSANTS_sell_order_volume = (
                            self.pair_trading_orders(
                                order_type='sell',
                                product=Product.CROISSANTS,
                                target_volume=CROISSANTS_target_sell_volume,
                                order_depth=state.order_depths[Product.CROISSANTS],
                                fair_value=CROISSANTS_fair_value,
                                width = buy_width,
                                position =CROISSANTS_position,
                                ))
                        
                        JAMS_pt_orders, JAMS_buy_order_volume, JAMS_sell_order_volume = (
                            self.pair_trading_orders(
                                order_type='sell',
                                product=Product.JAMS,
                                target_volume=JAMS_target_sell_volume,
                                order_depth=state.order_depths[Product.JAMS],
                                fair_value=JAMS_fair_value,
                                width = buy_width,
                                position =JAMS_position,
                                ))
                        
                        CROISSANTS_position -= CROISSANTS_sell_order_volume
                        JAMS_position       -= JAMS_sell_order_volume
                        croissant_solo_trading += CROISSANTS_pt_orders
                        jams_solo_trading      += JAMS_pt_orders
                        

                result[Product.PICNICBASKET2] = PICNICBASKET2_orders



            result[Product.CROISSANTS] = croissant_solo_trading
            result[Product.DJEMBES]    = djembes_solo_trading
            result[Product.JAMS]       = jams_solo_trading



        conversions = 1
        traderData = jsonpickle.encode(traderObject)

        
        return result, conversions, traderData
