from datamodel import OrderDepth, TradingState, Order
from typing import List, Dict, Tuple
import jsonpickle

class Product:
    CROISSANTS    = "CROISSANTS"
    DJEMBES       = "DJEMBES"
    JAMS          = "JAMS"
    PICNICBASKET1 = "PICNIC_BASKET1"
    PICNICBASKET2 = "PICNIC_BASKET2"

# Static parameters for each product
PARAMS = {
    Product.CROISSANTS: {
        "prevent_adverse": True,
        "adverse_volume": 100,
    },
    Product.JAMS: {
        "prevent_adverse": True,
        "adverse_volume": 100,
    },
    Product.DJEMBES: {
        "prevent_adverse": True,
        "adverse_volume": 50,
    },
    Product.PICNICBASKET1: {

        "alpha": 18764.9,
        "beta": 0.683181,
        "mean_spread": 3.2223*10**-11,
        "std_spread": 54.6538,
        "entry_threshold": 0.3,
        "exit_threshold": 1.1,
        "width_coef": 5.0,
        "unit_volume": 5,
    },
    Product.PICNICBASKET2: {
        "alpha": 12629.6,
        "beta": 0.584684,
        "mean_spread": -6.38829*10**(-13),
        "std_spread": 29.3605,
        "entry_threshold": 0.3,
        "exit_threshold": 0.8,
        "width_coef": 3.0,
        "unit_volume": 5,
    },
}

class Trader:
    def __init__(self, params: Dict = None):
        self.params = params or PARAMS
        # Hard position limits
        self.LIMIT = {
            Product.CROISSANTS:    250,
            Product.JAMS:          350,
            Product.DJEMBES:       60,
            Product.PICNICBASKET1: 60,
            Product.PICNICBASKET2: 100,
        }

    def swmid_fair_value(self, od: OrderDepth) -> float:
        """Size‑weighted mid price."""
        best_bid = max(od.buy_orders)
        best_ask = min(od.sell_orders)
        bid_vol = abs(od.buy_orders[best_bid])
        ask_vol = abs(od.sell_orders[best_ask])
        return (best_bid*ask_vol + best_ask*bid_vol) / (bid_vol + ask_vol)

    def pair_trading_orders(
        self, side: str, product: str, target_vol: int,
        od: OrderDepth, fair: float, width: float,
        position: int
    ) -> Tuple[List[Order], int]:
        """
        Place limit orders to buy/sell up to target_vol against the fair price ± width.
        Returns (orders, executed_volume).
        """
        orders: List[Order] = []
        executed = 0
        limit = self.LIMIT[product]
        p = self.params.get(product, {})
        prevent_adv = p.get("prevent_adverse", False)
        adv_vol = p.get("adverse_volume", 0)

        if side == 'buy' and od.sell_orders:
            best_ask = min(od.sell_orders)
            avail = -od.sell_orders[best_ask]
            if (not prevent_adv or avail <= adv_vol) and best_ask <= fair + width:
                qty = min(avail, target_vol, limit - position)
                if qty > 0:
                    orders.append(Order(product, best_ask, qty))
                    executed = qty

        elif side == 'sell' and od.buy_orders:
            best_bid = max(od.buy_orders)
            avail = od.buy_orders[best_bid]
            if (not prevent_adv or avail <= adv_vol) and best_bid >= fair - width:
                qty = min(avail, target_vol, limit + position)
                if qty > 0:
                    orders.append(Order(product, best_bid, -qty))
                    executed = qty

        return orders, executed

    def execute_pair_basket(
        self,
        basket: str,
        composition: Dict[str,int],
        state: TradingState,
        trader_data: dict
    ) -> Tuple[List[Order], Dict[str,List[Order]], dict]:
        """
        Single basket round: compute zscore, open/close positions deterministically,
        update and return new trader_data.
        """
        params = self.params[basket]
        # Load persisted positions
        basket_pos = trader_data.get(f"{basket}_pos", 0)
        under_pos  = trader_data.get("under_pos", {}).copy()

        # 1) compute fair prices
        basket_fair = self.swmid_fair_value(state.order_depths[basket])
        under_fairs = {
            p: self.swmid_fair_value(state.order_depths[p])
            for p in composition
        }
        # 2) zscore
        combined = sum(composition[p] * under_fairs[p] for p in composition)
        spread = basket_fair - (params["alpha"] + params["beta"] * combined)
        if params["std_spread"] != 0:
            zscore = (spread - params["mean_spread"]) / params["std_spread"]
        else:
            zscore = 0.0

        # 3) width & volume
        width = max(params["width_coef"] * abs(zscore), 1.0)
        unit = params["unit_volume"]

        basket_orders: List[Order] = []
        under_orders: Dict[str,List[Order]] = {p:[] for p in composition}

        # ENTRY
        if abs(zscore) >= params["entry_threshold"] and basket_pos == 0:
            # open new basket
            side = 'sell' if zscore > 0 else 'buy'
            orders, vol = self.pair_trading_orders(
                side, basket, unit, state.order_depths[basket],
                basket_fair, width, basket_pos
            )
            basket_orders += orders
            basket_pos += (-vol if side=='sell' else vol)

            # hedge legs
            for p, mult in composition.items():
                hedge_side = 'buy' if side=='sell' else 'sell'
                target = mult * vol
                o2, v2 = self.pair_trading_orders(
                    hedge_side, p, target, state.order_depths[p],
                    under_fairs[p], width, under_pos.get(p,0)
                )
                under_orders[p] += o2
                under_pos[p] = under_pos.get(p,0) + ( v2 if hedge_side=='buy' else -v2 )

        # EXIT
        elif abs(zscore) <= params["exit_threshold"] and basket_pos != 0:
            # flatten basket
            side = 'buy' if basket_pos < 0 else 'sell'
            target = abs(basket_pos)
            orders, vol = self.pair_trading_orders(
                side, basket, target, state.order_depths[basket],
                basket_fair, width, basket_pos
            )
            basket_orders += orders
            basket_pos += (-vol if side=='sell' else vol)

            # flatten legs
            for p, mult in composition.items():
                hedge_side = 'sell' if side=='sell' else 'buy'
                o2, v2 = self.pair_trading_orders(
                    hedge_side, p, abs(mult*vol),
                    state.order_depths[p], under_fairs[p], width, under_pos.get(p,0)
                )
                under_orders[p] += o2
                under_pos[p] = under_pos.get(p,0) + ( -v2 if hedge_side=='sell' else v2 )

        # persist updated positions
        trader_data[f"{basket}_pos"] = basket_pos
        trader_data["under_pos"] = under_pos

        return basket_orders, under_orders, trader_data

    def run(self, state: TradingState) -> Tuple[Dict[str,List[Order]], int, str]:
        # restore state
        trader_data = {}
        if state.traderData:
            trader_data = jsonpickle.decode(state.traderData)

        result: Dict[str,List[Order]] = {}
        # only trade if all legs appear
        if all(p in state.order_depths for p in
               [Product.CROISSANTS, Product.JAMS, Product.DJEMBES,
                Product.PICNICBASKET1, Product.PICNICBASKET2]):
            # Basket 1
            b1, u1, trader_data = self.execute_pair_basket(
                Product.PICNICBASKET1,
                {Product.CROISSANTS:6, Product.JAMS:3, Product.DJEMBES:1},
                state, trader_data
            )
            result[Product.PICNICBASKET1] = b1
            # collect underlying orders
            for p in u1:
                result.setdefault(p, []).extend(u1[p])

            # Basket 2
            b2, u2, trader_data = self.execute_pair_basket(
                Product.PICNICBASKET2,
                {Product.CROISSANTS:4, Product.JAMS:2},
                state, trader_data
            )
            result[Product.PICNICBASKET2] = b2
            for p in u2:
                result.setdefault(p, []).extend(u2[p])

        # serialize
        traderData = jsonpickle.encode(trader_data)
        return result, 1, traderData