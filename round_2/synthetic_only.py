from datamodel import OrderDepth, TradingState, Order
from typing import List, Dict, Tuple, Any
import jsonpickle  # type: ignore
import random
import numpy as np

# Define products
class Product:
    CROISSANTS    = "CROISSANTS"
    DJEMBES       = "DJEMBES"
    JAMS          = "JAMS"
    PICNICBASKET1 = "PICNIC_BASKET1"
    PICNICBASKET2 = "PICNIC_BASKET2"
    SYNTHETIC     = "SYNTHETIC"  # For synthetic underlying orders

# Basket composition constants
BASKET1_COMPOSITION = {Product.CROISSANTS: 6, Product.JAMS: 3, Product.DJEMBES: 1}
BASKET2_COMPOSITION = {Product.CROISSANTS: 4, Product.JAMS: 2}

# Parameter settings for products
PARAMS = {
    Product.CROISSANTS: {
        "take_width": 1,
        "clear_width": 1,
        "prevent_adverse": True,
        "adverse_volume": 85,
        "reversion_beta": -0.0739,
        "disregard_edge": 1,
        "join_edge": 0,
        "default_edge": 1,
    },
    Product.DJEMBES: {
        "take_width": 1,
        "clear_width": 1,
        "prevent_adverse": True,
        "adverse_volume": 130,
        "reversion_beta": 0.0618,
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
        "alpha": 853.31,
        "beta": 0.98641,
        "mean_spread": -2.69245,
        "std_spread": 85.4813,
        "entry_threshold": 0.5,
        "exit_threshold": 0.3
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
        "alpha": -2076.33,
        "beta": 1.07114,
        "mean_spread": -54.518,
        "std_spread": 62.0239,
        "entry_threshold": 1.5,
        "exit_threshold": 0.8,
    },
}

class Trader:
    def __init__(self, params: Dict = None):
        if params is None:
            params = PARAMS
        self.params = params
        self.LIMIT = {
            Product.CROISSANTS: 250,
            Product.JAMS: 350,
            Product.DJEMBES: 60,

            Product.SYNTHETIC: 41,

            Product.PICNICBASKET1: 60,
            Product.PICNICBASKET2: 100,
        }

    def fair_value(self, product: str, order_depth: OrderDepth, traderObject: dict) -> float:
        key_last_price = product.lower() + "_last_price"
        if order_depth.sell_orders and order_depth.buy_orders:
            best_ask = min(order_depth.sell_orders.keys())
            best_bid = max(order_depth.buy_orders.keys())
            filtered_ask = [p for p in order_depth.sell_orders.keys() if abs(order_depth.sell_orders[p]) >= self.params[product]["adverse_volume"]]
            filtered_bid = [p for p in order_depth.buy_orders.keys() if abs(order_depth.buy_orders[p]) >= self.params[product]["adverse_volume"]]
            mm_ask = min(filtered_ask) if filtered_ask else None
            mm_bid = max(filtered_bid) if filtered_bid else None
            if mm_ask is None or mm_bid is None:
                mid_price = (best_ask + best_bid) / 2 if traderObject.get(key_last_price) is None else traderObject[key_last_price]
            else:
                mid_price = (mm_ask + mm_bid) / 2
            if traderObject.get(key_last_price) is not None:
                last_price = traderObject[key_last_price]
                ret = (mid_price - last_price) / last_price
                adjustment = ret * self.params[product]["reversion_beta"]
                fair = mid_price + (mid_price * adjustment)
            else:
                fair = mid_price
            traderObject[product] = mid_price
            return fair
        return None

    def is_good_to_trade(self, product: str, state: TradingState) -> bool:
        return product in self.params and product in state.order_depths

    def pair_trading_best_sells(
        self,
        product: str,
        fair_value: float,
        width: float,
        orders: List[Order],
        order_depth: OrderDepth,
        position: int,
        target_volume: int,
        current_sell_volume: int,
        prevent_adverse: bool = False,
        adverse_volume: int = 0,
    ) -> int:
        limit = self.LIMIT[product]
        if order_depth.buy_orders:
            best_bid = max(order_depth.buy_orders.keys())
            amount = order_depth.buy_orders[best_bid]
            if not prevent_adverse or abs(amount) <= adverse_volume:
                if best_bid + width >= fair_value:
                    quantity = min(target_volume, amount, limit + position)
                    if quantity > 0:
                        orders.append(Order(product, best_bid, -quantity))
                        current_sell_volume += quantity
                        order_depth.buy_orders[best_bid] -= quantity
                        if order_depth.buy_orders[best_bid] == 0:
                            del order_depth.buy_orders[best_bid]
        return current_sell_volume

    def pair_trading_best_buys(
        self,
        product: str,
        fair_value: float,
        width: float,
        orders: List[Order],
        order_depth: OrderDepth,
        position: int,
        target_volume: int,
        current_buy_volume: int,
        prevent_adverse: bool = False,
        adverse_volume: int = 0,
    ) -> int:
        limit = self.LIMIT[product]
        if order_depth.sell_orders:
            best_ask = min(order_depth.sell_orders.keys())
            amount = -order_depth.sell_orders[best_ask]
            if not prevent_adverse or abs(amount) <= adverse_volume:
                if best_ask - width <= fair_value:
                    quantity = min(amount, target_volume, limit - position)
                    if quantity > 0:
                        orders.append(Order(product, best_ask, quantity))
                        current_buy_volume += quantity
                        order_depth.sell_orders[best_ask] += quantity
                        if order_depth.sell_orders[best_ask] == 0:
                            del order_depth.sell_orders[best_ask]
        return current_buy_volume

    def pair_trading_orders(
        self,
        side: str,
        product: str,
        target_volume: int,
        order_depth: OrderDepth,
        fair_value: float,
        width: float,
        position: int,
        prevent_adverse: bool = False,
        adverse_volume: int = 0,
    ) -> Tuple[List[Order], int, int]:
        orders: List[Order] = []
        buy_vol = 0
        sell_vol = 0
        if side == 'buy':
            buy_vol = self.pair_trading_best_buys(product, fair_value, width, orders, order_depth, position, target_volume, buy_vol, prevent_adverse, adverse_volume)
        elif side == 'sell':
            sell_vol = self.pair_trading_best_sells(product, fair_value, width, orders, order_depth, position, target_volume, sell_vol, prevent_adverse, adverse_volume)
        else:
            print("Invalid side specified")
        return orders, buy_vol, sell_vol

    # --- Synthetic underlying functions ---

    def get_synthetic_underlying_order_depth(
        self, composition: Dict[str, int], order_depths: Dict[str, OrderDepth]
    ) -> OrderDepth:
        synthetic_od = OrderDepth()
        # For buy side: sum weighted best bids and compute available volume
        bid_prices = []
        bid_volumes = []
        for prod, weight in composition.items():
            od = order_depths[prod]
            if od.buy_orders:
                best_bid = max(od.buy_orders.keys())
                bid_prices.append(best_bid * weight)
                bid_vol = od.buy_orders[best_bid] // weight
                bid_volumes.append(bid_vol)
            else:
                bid_prices.append(0)
                bid_volumes.append(0)
        if sum(bid_prices) > 0:
            implied_bid = sum(bid_prices)
            available_bid = min(bid_volumes) if bid_volumes else 0
            synthetic_od.buy_orders[implied_bid] = available_bid

        # For sell side: sum weighted best asks and compute available volume
        ask_prices = []
        ask_volumes = []
        for prod, weight in composition.items():
            od = order_depths[prod]
            if od.sell_orders:
                best_ask = min(od.sell_orders.keys())
                ask_prices.append(best_ask * weight)
                ask_vol = -od.sell_orders[best_ask] // weight
                ask_volumes.append(ask_vol)
            else:
                ask_prices.append(float("inf"))
                ask_volumes.append(0)
        if all(p < float("inf") for p in ask_prices):
            implied_ask = sum(ask_prices)
            available_ask = min(ask_volumes) if ask_volumes else 0
            synthetic_od.sell_orders[implied_ask] = -available_ask
        return synthetic_od

    def convert_synthetic_underlying_orders(
        self, synthetic_orders: List[Order], composition: Dict[str, int], order_depths: Dict[str, OrderDepth]
    ) -> Dict[str, List[Order]]:
        component_orders: Dict[str, List[Order]] = {prod: [] for prod in composition}
        syn_od = self.get_synthetic_underlying_order_depth(composition, order_depths)
        best_bid = max(syn_od.buy_orders.keys()) if syn_od.buy_orders else 0
        best_ask = min(syn_od.sell_orders.keys()) if syn_od.sell_orders else float("inf")
        for order in synthetic_orders:
            price = order.price
            qty = order.quantity
            # For a buy synthetic order, use component sell prices
            if qty > 0 and price >= best_ask:
                for prod, weight in composition.items():
                    od = order_depths[prod]
                    if od.sell_orders:
                        comp_price = min(od.sell_orders.keys())
                        comp_order = Order(prod, comp_price, qty * weight)
                        component_orders[prod].append(comp_order)
            # For a sell synthetic order, use component buy prices
            elif qty < 0 and price <= best_bid:
                for prod, weight in composition.items():
                    od = order_depths[prod]
                    if od.buy_orders:
                        comp_price = max(od.buy_orders.keys())
                        comp_order = Order(prod, comp_price, qty * weight)
                        component_orders[prod].append(comp_order)
        return component_orders

    # --- Basket pair trading with synthetic underlying orders ---
    def execute_basket_pair_trading(
        self,
        basket: str,
        composition: Dict[str, int],
        state: TradingState,
        traderObject: dict,
        under_fairs: Dict[str, float],
        under_positions: Dict[str, int],
        width_coef: float,
        unit_volume: int,
        signal_strength: float,
        synth_prod: str = Product.SYNTHETIC,
    ) -> Tuple[List[Order], Dict[str, List[Order]]]:
        basket_orders: List[Order] = []
        under_orders: Dict[str, List[Order]] = {prod: [] for prod in composition}
        basket_pos = state.position.get(basket, 0)
        basket_fair = self.fair_value(basket, state.order_depths[basket], traderObject)
        combined_under = sum(under_fairs[prod] * composition[prod] for prod in composition)
        params = self.params.get(basket, {})
        alpha = params.get("alpha", 0)
        beta = params.get("beta", 0)
        mean_spread = params.get("mean_spread", 0)
        std_spread = params.get("std_spread", 1)
        entry_threshold = params.get("entry_threshold", 0)
        exit_threshold = params.get("exit_threshold", 0)
        spread = basket_fair - (alpha + beta * combined_under)
        zscore = (spread - mean_spread) / std_spread if std_spread != 0 else 0
        width = width_coef * abs(zscore)

        basket_buy_vol = 0
        basket_sell_vol = 0

        # Determine basket order signal and corresponding synthetic underlying orders
        synthetic_under_fair = combined_under  # using weighted sum as synthetic fair value

        if zscore > entry_threshold:
            # Basket overvalued: sell basket, buy underlying synthetic
            if random.uniform(0, 1) < signal_strength:
                ords, basket_buy_vol, basket_sell_vol = self.pair_trading_orders(
                    side='sell',
                    product=basket,
                    target_volume=unit_volume,
                    order_depth=state.order_depths[basket],
                    fair_value=basket_fair,
                    width=width,
                    position=basket_pos,
                )
                basket_orders.extend(ords)
                basket_pos -= basket_sell_vol
            # Execute synthetic underlying orders: buy underlying
            syn_orders, syn_buy, _ = self.pair_trading_orders(
                side='buy',
                product=synth_prod,
                target_volume=unit_volume,
                order_depth=self.get_synthetic_underlying_order_depth(composition, state.order_depths),
                fair_value=synthetic_under_fair,
                width=width,
                position=0,
            )
            converted_orders = self.convert_synthetic_underlying_orders(syn_orders, composition, state.order_depths)
            for prod in converted_orders:
                under_orders[prod].extend(converted_orders[prod])
        elif zscore < -entry_threshold:
            # Basket undervalued: buy basket, sell underlying synthetic
            if random.uniform(0, 1) < signal_strength:
                ords, basket_buy_vol, basket_sell_vol = self.pair_trading_orders(
                    side='buy',
                    product=basket,
                    target_volume=unit_volume,
                    order_depth=state.order_depths[basket],
                    fair_value=basket_fair,
                    width=width,
                    position=basket_pos,
                )
                basket_orders.extend(ords)
                basket_pos += basket_buy_vol
            # Execute synthetic underlying orders: sell underlying
            syn_orders, _, syn_sell = self.pair_trading_orders(
                side='sell',
                product=synth_prod,
                target_volume=unit_volume,
                order_depth=self.get_synthetic_underlying_order_depth(composition, state.order_depths),
                fair_value=synthetic_under_fair,
                width=width,
                position=0,
            )
            converted_orders = self.convert_synthetic_underlying_orders(syn_orders, composition, state.order_depths)
            for prod in converted_orders:
                under_orders[prod].extend(converted_orders[prod])
        else:
            # Exit logic: if zscore near zero, attempt to unwind basket positions
            if abs(zscore) < exit_threshold:
                if basket_pos > 0 and random.uniform(0, 1) < signal_strength:
                    ords, _, vol = self.pair_trading_orders(
                        side='sell',
                        product=basket,
                        target_volume=abs(basket_pos),
                        order_depth=state.order_depths[basket],
                        fair_value=basket_fair,
                        width=width,
                        position=basket_pos,
                    )
                    basket_orders.extend(ords)
                    basket_pos -= vol
                elif basket_pos < 0 and random.uniform(0, 1) < signal_strength:
                    ords, vol, _ = self.pair_trading_orders(
                        side='buy',
                        product=basket,
                        target_volume=abs(basket_pos),
                        order_depth=state.order_depths[basket],
                        fair_value=basket_fair,
                        width=width,
                        position=basket_pos,
                    )
                    basket_orders.extend(ords)
                    basket_pos += vol

        return basket_orders, under_orders

    def run(self, state: TradingState) -> Tuple[Dict[str, List[Order]], int, str]:
        traderObject = {}
        if state.traderData:
            traderObject = jsonpickle.decode(state.traderData)
        result: Dict[str, List[Order]] = {}

        # Underlying products tradability check
        croissant_ok = self.is_good_to_trade(Product.CROISSANTS, state)
        jams_ok = self.is_good_to_trade(Product.JAMS, state)
        djembes_ok = self.is_good_to_trade(Product.DJEMBES, state)

        # Get positions and fair values for underlying products
        pos_croissant = state.position.get(Product.CROISSANTS, 0)
        pos_jams = state.position.get(Product.JAMS, 0)
        pos_djembes = state.position.get(Product.DJEMBES, 0)
        croissant_fair = self.fair_value(Product.CROISSANTS, state.order_depths[Product.CROISSANTS], traderObject)
        jams_fair = self.fair_value(Product.JAMS, state.order_depths[Product.JAMS], traderObject)
        djembes_fair = self.fair_value(Product.DJEMBES, state.order_depths[Product.DJEMBES], traderObject)
        under_fairs = {
            Product.CROISSANTS: croissant_fair,
            Product.JAMS: jams_fair,
            Product.DJEMBES: djembes_fair,
        }
        under_positions = {
            Product.CROISSANTS: pos_croissant,
            Product.JAMS: pos_jams,
            Product.DJEMBES: pos_djembes,
        }

        # Basket trading flags
        basket1_ok = self.is_good_to_trade(Product.PICNICBASKET1, state)
        basket2_ok = self.is_good_to_trade(Product.PICNICBASKET2, state)

        UNITS = 5
        signal_strength = 1.0

        # Result orders containers for baskets and underlying
        basket1_orders: List[Order] = []
        basket2_orders: List[Order] = []
        synthetic_underlying_orders: Dict[str, List[Order]] = {prod: [] for prod in [Product.CROISSANTS, Product.JAMS, Product.DJEMBES]}

        if croissant_ok and jams_ok and djembes_ok:
            # Execute trading on Picnic Basket 1 (composition: 6 CROISSANTS, 3 JAMS, 1 DJEMBES)
            if basket1_ok:
                b1_orders, under_orders_b1 = self.execute_basket_pair_trading(
                    basket=Product.PICNICBASKET1,
                    composition=BASKET1_COMPOSITION,
                    state=state,
                    traderObject=traderObject,
                    under_fairs=under_fairs,
                    under_positions=under_positions,
                    width_coef=10,
                    unit_volume=UNITS,
                    signal_strength=signal_strength,
                )
                basket1_orders.extend(b1_orders)
                for prod, orders_list in under_orders_b1.items():
                    synthetic_underlying_orders.setdefault(prod, []).extend(orders_list)
            # Execute trading on Picnic Basket 2 (composition: 4 CROISSANTS, 2 JAMS)
            if basket2_ok:
                b2_orders, under_orders_b2 = self.execute_basket_pair_trading(
                    basket=Product.PICNICBASKET2,
                    composition=BASKET2_COMPOSITION,
                    state=state,
                    traderObject=traderObject,
                    under_fairs=under_fairs,
                    under_positions=under_positions,
                    width_coef=5,
                    unit_volume=UNITS,
                    signal_strength=signal_strength,
                )
                basket2_orders.extend(b2_orders)
                for prod, orders_list in under_orders_b2.items():
                    synthetic_underlying_orders.setdefault(prod, []).extend(orders_list)

            result[Product.PICNICBASKET1] = basket1_orders
            result[Product.PICNICBASKET2] = basket2_orders
            # Place synthetic underlying orders as a single group
            for prod in synthetic_underlying_orders:
                result.setdefault(prod, []).extend(synthetic_underlying_orders[prod])

        traderData = jsonpickle.encode(traderObject)
        conversions = 1
        return result, conversions, traderData