from datamodel import OrderDepth, TradingState, Order, ConversionObservation
from typing import Dict, List, Tuple
import jsonpickle
import numpy as np

INF = float('inf')


class Product:
    MACARONS = "MAGNIFICENT_MACARONS"


# === PARAMETERS (unused by exchange_arb but kept for style) ===
PARAMS = {
    Product.MACARONS: {
        "init_make_edge":    2.0,
        "step_size":         0.5,
        "min_edge":          0.5,
        "volume_bar":       75,
        "dec_edge_discount": 0.80,
        "volume_avg_window": 5,
        "take_edge":         0.75,
        # mr params omitted
    }
}

POSITION_LIMITS = {Product.MACARONS: 75}
CONVERSION_LIMIT = 10  # max ±10 per tick


def clamp(value: int, low: int, high: int) -> int:
    return max(low, min(high, value))


def implied_bid_ask(obs: ConversionObservation) -> Tuple[float, float]:
    """Compute implied fair bid/ask from conversion market."""
    bid = obs.bidPrice - obs.exportTariff - obs.transportFees - 0.1
    ask = obs.askPrice + obs.importTariff + obs.transportFees
    return bid, ask


class Trader:
    @staticmethod
    def exchange_arb(state, fair_price: float, next_price_move: float = 0) -> List[Order]:
        """Find optimal single-sided arb order via expected-profit loops."""
        orders: List[Order] = []
        # cost for selling side
        cost_sell = state.transportFees + state.importTariff
        my_ask = state.maxamt_bidprc
        best_ask_profit = 0.0
        opt_ask = INF

        # scan potential ask quotes
        while my_ask < fair_price:
            delta = my_ask - fair_price
            prob = 0.5
            if my_ask > state.best_bid:
                profit = my_ask - (state.orchid_south_askprc + next_price_move)
                exp = prob * (profit - cost_sell)
            else:
                probs, prices, amts = [], [], []
                for p, a in state.bids:
                    if p >= my_ask:
                        probs.append(1.0)
                        prices.append(p)
                        amts.append(a)
                total = np.sum(amts)
                if total < state.position_limit:
                    probs.append(ExecutionProb.orchids(delta))
                    prices.append(my_ask)
                    amts.append(state.position_limit - total)
                profits = np.array(prices) - (state.orchid_south_askprc + next_price_move)
                exp = (np.array(probs) * (profits - cost_sell) * np.array(amts) / state.position_limit).sum()

            if exp > best_ask_profit:
                best_ask_profit = exp
                opt_ask = my_ask
            my_ask += 1

        # cost for buying side
        cost_buy = state.transportFees + state.exportTariff + state.storageFees
        my_bid = state.maxamt_askprc
        best_bid_profit = 0.0
        opt_bid = 1

        # scan potential bid quotes
        while my_bid > fair_price:
            delta = fair_price - my_bid
            prob = ExecutionProb.orchids(delta)
            if my_bid < state.best_ask:
                profit = (state.orchid_south_bidprc + next_price_move) - my_bid
                exp = prob * (profit - cost_buy)
            else:
                probs, prices, amts = [], [], []
                for p, a in state.asks:
                    if p <= my_bid:
                        probs.append(1.0)
                        prices.append(p)
                        amts.append(abs(a))
                total = np.sum(amts)
                if total < state.position_limit:
                    probs.append(ExecutionProb.orchids(delta))
                    prices.append(my_bid)
                    amts.append(state.position_limit - total)
                profits = (state.orchid_south_bidprc + next_price_move) - np.array(prices)
                exp = (np.array(probs) * (profits - cost_buy) * np.array(amts) / state.position_limit).sum()

            if exp > best_bid_profit:
                best_bid_profit = exp
                opt_bid = my_bid
            my_bid -= 1

        # choose the more profitable side
        if best_ask_profit >= best_bid_profit and best_ask_profit > 0:
            orders.append(Order(state.product, int(opt_ask), -state.position_limit))
        elif best_bid_profit > best_ask_profit and best_bid_profit > 0:
            orders.append(Order(state.product, int(opt_bid),  state.position_limit))

        return orders

    def run(self, state: TradingState):
        # prepare result containers
        results: Dict[str, List[Order]] = {}
        conversions = 0

        if Product.MACARONS in state.order_depths and \
           Product.MACARONS in state.observations.conversionObservations:

            depth = state.order_depths[Product.MACARONS]
            obs = state.observations.conversionObservations[Product.MACARONS]
            pos = state.position.get(Product.MACARONS, 0)

            # build a minimal 'status' for exchange_arb
            status = type("Status", (), {})()
            status.transportFees = obs.transportFees
            status.importTariff = obs.importTariff
            status.exportTariff = obs.exportTariff
            status.storageFees = getattr(obs, "storageFees", 0.0)
            status.bids = list(depth.buy_orders.items())
            status.asks = list(depth.sell_orders.items())
            status.best_bid = max(depth.buy_orders) if depth.buy_orders else 0.0
            status.best_ask = min(depth.sell_orders) if depth.sell_orders else 0.0
            status.maxamt_bidprc = status.best_bid
            status.maxamt_askprc = status.best_ask
            status.orchid_south_askprc = obs.askPrice
            status.orchid_south_bidprc = obs.bidPrice
            status.position_limit = POSITION_LIMITS[Product.MACARONS]
            status.product = Product.MACARONS

            # compute fair price
            ib, ia = implied_bid_ask(obs)
            fair_price = (ib + ia) / 2.0

            # get arb orders
            orders = self.exchange_arb(status, fair_price)

            results[Product.MACARONS] = orders
            # flatten residual via conversion ≤ ±10
            conversions = clamp(-pos, -CONVERSION_LIMIT, CONVERSION_LIMIT)

        return results, conversions, jsonpickle.encode({})