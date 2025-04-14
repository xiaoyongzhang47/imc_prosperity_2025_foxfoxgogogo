from datamodel import OrderDepth, TradingState, Order
from typing import List, Dict, Tuple
import jsonpickle  # type: ignore
import random

class Product:
    CROISSANTS    = "CROISSANTS"
    DJEMBES       = "DJEMBES"
    JAMS          = "JAMS"
    PICNICBASKET1 = "PICNIC_BASKET1"
    PICNICBASKET2 = "PICNIC_BASKET2"

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
        "entry_threshold": 1.5,
        "exit_threshold": 0.25
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
        "exit_threshold": 0.25
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
            Product.PICNICBASKET1: 60,
            Product.PICNICBASKET2: 100,
        }

    def fair_value(self, product: str, order_depth: OrderDepth, traderObject: dict) -> float:
        key_last_price = product.lower() + "_last_price"
        if order_depth.sell_orders and order_depth.buy_orders:
            best_ask = min(order_depth.sell_orders.keys())
            best_bid = max(order_depth.buy_orders.keys())
            filtered_ask = [price for price in order_depth.sell_orders.keys()
                            if abs(order_depth.sell_orders[price]) >= self.params[product]["adverse_volume"]]
            filtered_bid = [price for price in order_depth.buy_orders.keys()
                            if abs(order_depth.buy_orders[price]) >= self.params[product]["adverse_volume"]]
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
        prod_strength: float,
    ) -> Tuple[List[Order], Dict[str, List[Order]]]:
        basket_orders: List[Order] = []
        under_orders: Dict[str, List[Order]] = {p: [] for p in composition}
        basket_pos = state.position.get(basket, 0)
        basket_fair = self.fair_value(basket, state.order_depths[basket], traderObject)
        combined_under = sum(mult * under_fairs[p] for p, mult in composition.items())
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

        # Determine signal and apply entry/exit logic
        basket_buy_vol = 0
        basket_sell_vol = 0
        # Entry signal: zscore exceeds thresholds
        if zscore > entry_threshold:
            # Basket appears overvalued: sell basket, buy underlying
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
            for prod, mult in composition.items():
                target = mult * basket_sell_vol
                if random.uniform(0, 1) < prod_strength:
                    ords, bvol, _ = self.pair_trading_orders(
                        side='buy',
                        product=prod,
                        target_volume=target,
                        order_depth=state.order_depths[prod],
                        fair_value=under_fairs[prod],
                        width=width,
                        position=under_positions[prod],
                    )
                    under_orders[prod].extend(ords)
                    under_positions[prod] += bvol
        elif zscore < -entry_threshold:
            # Basket appears undervalued: buy basket, sell underlying
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
            for prod, mult in composition.items():
                target = mult * basket_buy_vol
                if random.uniform(0, 1) < prod_strength:
                    ords, _, svol = self.pair_trading_orders(
                        side='sell',
                        product=prod,
                        target_volume=target,
                        order_depth=state.order_depths[prod],
                        fair_value=under_fairs[prod],
                        width=width,
                        position=under_positions[prod],
                    )
                    under_orders[prod].extend(ords)
                    under_positions[prod] -= svol
        else:
            # Exit signal: if zscore is close to zero (mean reversion), try to unwind positions.
            # For simplicity, we attempt to flatten basket position.
            if abs(zscore) < exit_threshold:
                if basket_pos > 0 and random.uniform(0, 1) < signal_strength:
                    ords, _, sell_vol = self.pair_trading_orders(
                        side='sell',
                        product=basket,
                        target_volume=abs(basket_pos),
                        order_depth=state.order_depths[basket],
                        fair_value=basket_fair,
                        width=width,
                        position=basket_pos,
                    )
                    basket_orders.extend(ords)
                    basket_pos -= sell_vol
                elif basket_pos < 0 and random.uniform(0, 1) < signal_strength:
                    ords, buy_vol, _ = self.pair_trading_orders(
                        side='buy',
                        product=basket,
                        target_volume=abs(basket_pos),
                        order_depth=state.order_depths[basket],
                        fair_value=basket_fair,
                        width=width,
                        position=basket_pos,
                    )
                    basket_orders.extend(ords)
                    basket_pos += buy_vol

        return basket_orders, under_orders

    def run(self, state: TradingState) -> Tuple[Dict[str, List[Order]], int, str]:
        traderObject = {}
        if state.traderData:
            traderObject = jsonpickle.decode(state.traderData)
        result: Dict[str, List[Order]] = {}

        # Check tradability for individual products
        croissant_ok = self.is_good_to_trade(Product.CROISSANTS, state)
        jams_ok = self.is_good_to_trade(Product.JAMS, state)
        djembes_ok = self.is_good_to_trade(Product.DJEMBES, state)

        croissant_orders: List[Order] = []
        jams_orders: List[Order] = []
        djembes_orders: List[Order] = []

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
        prod_strength = 0.3

        if croissant_ok and jams_ok and djembes_ok:
            # Execute trading on Picnic Basket 1: 6 CROISSANTS, 3 JAMS, 1 DJEMBES
            if basket1_ok:
                b1_orders, under_orders_b1 = self.execute_basket_pair_trading(
                    basket=Product.PICNICBASKET1,
                    composition={Product.CROISSANTS: 6, Product.JAMS: 3, Product.DJEMBES: 1},
                    state=state,
                    traderObject=traderObject,
                    under_fairs=under_fairs,
                    under_positions=under_positions,
                    width_coef=10,
                    unit_volume=UNITS,
                    signal_strength=signal_strength,
                    prod_strength=prod_strength,
                )
                result[Product.PICNICBASKET1] = b1_orders
                croissant_orders.extend(under_orders_b1.get(Product.CROISSANTS, []))
                jams_orders.extend(under_orders_b1.get(Product.JAMS, []))
                djembes_orders.extend(under_orders_b1.get(Product.DJEMBES, []))
            # Execute trading on Picnic Basket 2: 4 CROISSANTS, 2 JAMS
            if basket2_ok:
                b2_orders, under_orders_b2 = self.execute_basket_pair_trading(
                    basket=Product.PICNICBASKET2,
                    composition={Product.CROISSANTS: 4, Product.JAMS: 2},
                    state=state,
                    traderObject=traderObject,
                    under_fairs=under_fairs,
                    under_positions=under_positions,
                    width_coef=5,
                    unit_volume=UNITS,
                    signal_strength=signal_strength,
                    prod_strength=prod_strength,
                )
                result[Product.PICNICBASKET2] = b2_orders
                croissant_orders.extend(under_orders_b2.get(Product.CROISSANTS, []))
                jams_orders.extend(under_orders_b2.get(Product.JAMS, []))
            # Append individual product orders
            result[Product.CROISSANTS] = croissant_orders
            result[Product.JAMS] = jams_orders
            result[Product.DJEMBES] = djembes_orders

        traderData = jsonpickle.encode(traderObject)
        conversions = 1
        return result, conversions, traderData