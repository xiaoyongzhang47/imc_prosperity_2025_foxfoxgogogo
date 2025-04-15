import json
from typing import Any

from datamodel import Listing, Observation, Order, OrderDepth, ProsperityEncoder, Symbol, Trade, TradingState


class Logger:
    def __init__(self) -> None:
        self.logs = ""
        self.max_log_length = 3750

    def print(self, *objects: Any, sep: str = " ", end: str = "\n") -> None:
        self.logs += sep.join(map(str, objects)) + end

    def flush(self, state: TradingState, orders: dict[Symbol, list[Order]], conversions: int, trader_data: str) -> None:
        base_length = len(
            self.to_json(
                [
                    self.compress_state(state, ""),
                    self.compress_orders(orders),
                    conversions,
                    "",
                    "",
                ]
            )
        )

        # We truncate state.traderData, trader_data, and self.logs to the same max. length to fit the log limit
        max_item_length = (self.max_log_length - base_length) // 3

        print(
            self.to_json(
                [
                    self.compress_state(state, self.truncate(state.traderData, max_item_length)),
                    self.compress_orders(orders),
                    conversions,
                    self.truncate(trader_data, max_item_length),
                    self.truncate(self.logs, max_item_length),
                ]
            )
        )

        self.logs = ""

    def compress_state(self, state: TradingState, trader_data: str) -> list[Any]:
        return [
            state.timestamp,
            trader_data,
            self.compress_listings(state.listings),
            self.compress_order_depths(state.order_depths),
            self.compress_trades(state.own_trades),
            self.compress_trades(state.market_trades),
            state.position,
            self.compress_observations(state.observations),
        ]

    def compress_listings(self, listings: dict[Symbol, Listing]) -> list[list[Any]]:
        compressed = []
        for listing in listings.values():
            compressed.append([listing.symbol, listing.product, listing.denomination])

        return compressed

    def compress_order_depths(self, order_depths: dict[Symbol, OrderDepth]) -> dict[Symbol, list[Any]]:
        compressed = {}
        for symbol, order_depth in order_depths.items():
            compressed[symbol] = [order_depth.buy_orders, order_depth.sell_orders]

        return compressed

    def compress_trades(self, trades: dict[Symbol, list[Trade]]) -> list[list[Any]]:
        compressed = []
        for arr in trades.values():
            for trade in arr:
                compressed.append(
                    [
                        trade.symbol,
                        trade.price,
                        trade.quantity,
                        trade.buyer,
                        trade.seller,
                        trade.timestamp,
                    ]
                )

        return compressed

    def compress_observations(self, observations: Observation) -> list[Any]:
        conversion_observations = {}
        for product, observation in observations.conversionObservations.items():
            conversion_observations[product] = [
                observation.bidPrice,
                observation.askPrice,
                observation.transportFees,
                observation.exportTariff,
                observation.importTariff,
                observation.sugarPrice,
                observation.sunlightIndex,
            ]

        return [observations.plainValueObservations, conversion_observations]

    def compress_orders(self, orders: dict[Symbol, list[Order]]) -> list[list[Any]]:
        compressed = []
        for arr in orders.values():
            for order in arr:
                compressed.append([order.symbol, order.price, order.quantity])

        return compressed

    def to_json(self, value: Any) -> str:
        return json.dumps(value, cls=ProsperityEncoder, separators=(",", ":"))

    def truncate(self, value: str, max_length: int) -> str:
        lo, hi = 0, min(len(value), max_length)
        out = ""

        while lo <= hi:
            mid = (lo + hi) // 2

            candidate = value[:mid]
            if len(candidate) < len(value):
                candidate += "..."

            encoded_candidate = json.dumps(candidate)

            if len(encoded_candidate) <= max_length:
                out = candidate
                lo = mid + 1
            else:
                hi = mid - 1

        return out


logger = Logger()

from datamodel import OrderDepth, TradingState, Order
from typing import List, Dict, Any, Tuple, Optional
import jsonpickle  # type: ignore
import numpy as np
from math import log, sqrt
from statistics import NormalDist

class Product:
    VOLCANICROCK = "VOLCANIC_ROCK"

    VR_VOUCHER_0950 = "VOLCANIC_ROCK_VOUCHER_9500"
    VR_VOUCHER_0975 = "VOLCANIC_ROCK_VOUCHER_9750"
    VR_VOUCHER_1000 = "VOLCANIC_ROCK_VOUCHER_10000"
    VR_VOUCHER_1025 = "VOLCANIC_ROCK_VOUCHER_10250"
    VR_VOUCHER_1050 = "VOLCANIC_ROCK_VOUCHER_10500"

PARAMS: Dict[str, Dict[str, Any]] = {
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
    def black_scholes_call(spot: float, strike: float, time_to_expiry: float, volatility: float) -> float:
        d1 = (log(spot) - log(strike) + 0.5 * volatility * volatility * time_to_expiry) / (volatility * sqrt(time_to_expiry))
        d2 = d1 - volatility * sqrt(time_to_expiry)
        call_price = spot * NormalDist().cdf(d1) - strike * NormalDist().cdf(d2)
        return call_price

    @staticmethod
    def delta(spot: float, strike: float, time_to_expiry: float, volatility: float) -> float:
        d1 = (log(spot) - log(strike) + 0.5 * volatility * volatility * time_to_expiry) / (volatility * sqrt(time_to_expiry))
        return NormalDist().cdf(d1)

    @staticmethod
    def implied_volatility(call_price: float, spot: float, strike: float, time_to_expiry: float,
                             max_iterations: int = 200, tolerance: float = 1e-10) -> float:
        low_vol = 0.01
        high_vol = 1.0
        volatility = (low_vol + high_vol) / 2.0
        for _ in range(max_iterations):
            estimated_price = BlackScholes.black_scholes_call(spot, strike, time_to_expiry, volatility)
            diff = estimated_price - call_price
            if abs(diff) < tolerance:
                break
            if diff > 0:
                high_vol = volatility
            else:
                low_vol = volatility
            volatility = (low_vol + high_vol) / 2.0
        return volatility

class Trader:
    # Fixed spread trade parameters
    SPREAD_THRESHOLD = 0.1  # 1% threshold for spread mispricing
    TRADE_QTY_SPREAD = 10    # default trading quantity for spread orders
    TRADING_ROUND = 2
    

    def __init__(self, params: Optional[Dict[str, Dict[str, Any]]] = None):
        
        if params is None:
            params = PARAMS
        self.params = params

        self.LIMIT: Dict[str, int] = {
            Product.VOLCANICROCK:    400,
            Product.VR_VOUCHER_0950: 200,
            Product.VR_VOUCHER_0975: 200,
            Product.VR_VOUCHER_1000: 200,
            Product.VR_VOUCHER_1025: 200,
            Product.VR_VOUCHER_1050: 200,
        }

    def voucher_starting_time_to_expiry_update(self):
        for product, params in self.params.items():
            if product.startswith("VOLCANIC_ROCK_VOUCHER"):
                params["starting_time_to_expiry"] = self.TRADING_ROUND / 7


    def get_voucher_mid_price(self, order_depth: OrderDepth, traderData: Dict[str, Any]) -> float:
        if order_depth.buy_orders and order_depth.sell_orders:
            best_bid = max(order_depth.buy_orders.keys())
            best_ask = min(order_depth.sell_orders.keys())
            mid_price = (best_bid + best_ask) / 2.0
            traderData["prev_voucher_price"] = mid_price
            return mid_price
        return traderData.get("prev_voucher_price", 0.0)

    def voucher_orders(self,
                       product: str,
                       order_depth: OrderDepth,
                       position: int,
                       traderData: Dict[str, Any],
                       volatility: float) -> Tuple[Optional[List[Order]], Optional[List[Order]]]:
        traderData.setdefault("past_voucher_vol", [])
        traderData["past_voucher_vol"].append(volatility)
        std_window = self.params[product]["std_window"]
        if len(traderData["past_voucher_vol"]) < std_window:
            return None, None
        if len(traderData["past_voucher_vol"]) > std_window:
            traderData["past_voucher_vol"].pop(0)
        vol_std = np.std(traderData["past_voucher_vol"]) or 1e-8
        vol_z_score = (volatility - self.params[product]["mean_volatility"]) / vol_std

        # If the z-score is extreme, propose a directional trade
        if vol_z_score >= self.params[product]["zscore_threshold"]:
            if position != -self.LIMIT[product]:
                target_position = -self.LIMIT[product]
                if order_depth.buy_orders:
                    best_bid = max(order_depth.buy_orders.keys())
                    target_quantity = abs(target_position - position)
                    quantity = min(target_quantity, abs(order_depth.buy_orders[best_bid]))
                    quote_quantity = target_quantity - quantity
                    if quote_quantity == 0:
                        return [Order(product, best_bid, -quantity)], []
                    else:
                        return ([Order(product, best_bid, -quantity)],
                                [Order(product, best_bid, -quote_quantity)])
        elif vol_z_score <= -self.params[product]["zscore_threshold"]:
            if position != self.LIMIT[product]:
                target_position = self.LIMIT[product]
                if order_depth.sell_orders:
                    best_ask = min(order_depth.sell_orders.keys())
                    target_quantity = abs(target_position - position)
                    quantity = min(target_quantity, abs(order_depth.sell_orders[best_ask]))
                    quote_quantity = target_quantity - quantity
                    if quote_quantity == 0:
                        return [Order(product, best_ask, quantity)], []
                    else:
                        return ([Order(product, best_ask, quantity)],
                                [Order(product, best_ask, quote_quantity)])
        return None, None

    def compute_underlying_hedge_orders(self,
                                        underlying_order_depth: OrderDepth,
                                        underlying_position: int,
                                        net_delta_exposure: float) -> List[Order]:
        target_position = -round(net_delta_exposure)
        delta_quantity = target_position - underlying_position
        orders: List[Order] = []
        if delta_quantity > 0 and underlying_order_depth.sell_orders:
            best_ask = min(underlying_order_depth.sell_orders.keys())
            quantity = min(delta_quantity, self.LIMIT[Product.VOLCANICROCK] - underlying_position)
            if quantity > 0:
                orders.append(Order(Product.VOLCANICROCK, best_ask, quantity))
        elif delta_quantity < 0 and underlying_order_depth.buy_orders:
            best_bid = max(underlying_order_depth.buy_orders.keys())
            quantity = min(abs(delta_quantity), self.LIMIT[Product.VOLCANICROCK] + underlying_position)
            if quantity > 0:
                orders.append(Order(Product.VOLCANICROCK, best_bid, -quantity))
        return orders

    def spread_voucher_orders(self,
                              voucher_data: Dict[str, Dict[str, Any]],
                              underlying_mid: float) -> Dict[str, List[Order]]:
        """
        Identify spread opportunities between adjacent voucher strikes.
        If the observed market spread deviates from the theoretical vertical spread by more than
        SPREAD_THRESHOLD, generate orders to exploit the discrepancy.
        """
        spread_orders: Dict[str, List[Order]] = {}
        # Sort vouchers by strike price (ascending)
        sorted_vouchers = sorted(voucher_data.items(), key=lambda x: x[1]["strike"])
        for i in range(len(sorted_vouchers) - 1):
            lower_prod, lower_data = sorted_vouchers[i]
            higher_prod, higher_data = sorted_vouchers[i + 1]

            # Market mid prices
            mid_lower = lower_data["mid"]
            mid_higher = higher_data["mid"]
            market_spread = mid_lower - mid_higher

            # Theoretical vertical spread prices using Black-Scholes call values
            lower_theo = BlackScholes.black_scholes_call(underlying_mid,
                                                         lower_data["strike"],
                                                         lower_data["tte"],
                                                         lower_data["vol"])
            higher_theo = BlackScholes.black_scholes_call(underlying_mid,
                                                          higher_data["strike"],
                                                          higher_data["tte"],
                                                          higher_data["vol"])
            theo_spread = lower_theo - higher_theo

            # Check for a significant discrepancy
            if market_spread < theo_spread * (1 - self.SPREAD_THRESHOLD):
                # Lower voucher undervalued vs. higher voucher:
                # BUY lower voucher at ask, SELL higher voucher at bid.
                lower_od = lower_data["order_depth"]
                higher_od = higher_data["order_depth"]

                if lower_od.sell_orders and higher_od.buy_orders:
                    lower_ask = min(lower_od.sell_orders.keys())
                    higher_bid = max(higher_od.buy_orders.keys())
                    qty = self.TRADE_QTY_SPREAD
                    # Apply trading limits and available book size
                    qty_lower = min(qty, abs(lower_od.sell_orders[lower_ask]),
                                    self.LIMIT.get(lower_prod, qty))
                    qty_higher = min(qty, higher_od.buy_orders[higher_bid],
                                     self.LIMIT.get(higher_prod, qty))
                    trade_qty = min(qty_lower, qty_higher)
                    if trade_qty > 0:
                        spread_orders.setdefault(lower_prod, []).append(Order(lower_prod, lower_ask, trade_qty))
                        spread_orders.setdefault(higher_prod, []).append(Order(higher_prod, higher_bid, -trade_qty))
            elif market_spread > theo_spread * (1 + self.SPREAD_THRESHOLD):
                # Lower voucher overvalued vs. higher voucher:
                # SELL lower voucher at bid, BUY higher voucher at ask.
                lower_od = lower_data["order_depth"]
                higher_od = higher_data["order_depth"]

                if lower_od.buy_orders and higher_od.sell_orders:
                    lower_bid = max(lower_od.buy_orders.keys())
                    higher_ask = min(higher_od.sell_orders.keys())
                    qty = self.TRADE_QTY_SPREAD
                    qty_lower = min(qty, lower_od.buy_orders[lower_bid],
                                    self.LIMIT.get(lower_prod, qty))
                    qty_higher = min(qty, abs(higher_od.sell_orders[higher_ask]),
                                     self.LIMIT.get(higher_prod, qty))
                    trade_qty = min(qty_lower, qty_higher)
                    if trade_qty > 0:
                        spread_orders.setdefault(lower_prod, []).append(Order(lower_prod, lower_bid, -trade_qty))
                        spread_orders.setdefault(higher_prod, []).append(Order(higher_prod, higher_ask, trade_qty))
        return spread_orders

    def run(self, state: TradingState) -> Tuple[Dict[str, List[Order]], int, str]:

        conversions = 1

        # Decode persistent trader data
        traderObject: Dict[str, Any] = {}
        if state.traderData:
            traderObject = jsonpickle.decode(state.traderData)
        result: Dict[str, List[Order]] = {}

        # Ensure entries for each voucher product exist
        for product in self.params:
            if product.startswith("VOLCANIC_ROCK_VOUCHER") and product not in traderObject:
                traderObject[product] = {"prev_voucher_price": 0.0, "past_voucher_vol": []}

        # Underlying market data for VOLCANIC_ROCK
        if Product.VOLCANICROCK not in state.order_depths:
            return result, 1, jsonpickle.encode(traderObject)
        underlying_od = state.order_depths[Product.VOLCANICROCK]
        if not (underlying_od.buy_orders and underlying_od.sell_orders):
            return result, 1, jsonpickle.encode(traderObject)
        underlying_mid = (max(underlying_od.buy_orders.keys()) + min(underlying_od.sell_orders.keys())) / 2

        # Container to gather voucher data for spread analysis
        voucher_data: Dict[str, Dict[str, Any]] = {}
        net_delta_exposure = 0.0

        # Process individual voucher products
        for voucher in self.params:
            if not voucher.startswith("VOLCANIC_ROCK_VOUCHER") or voucher not in state.order_depths:
                continue
            od = state.order_depths[voucher]
            position = state.position.get(voucher, 0)
            traderInfo = traderObject[voucher]
            mid_price = self.get_voucher_mid_price(od, traderInfo)
            tte = self.params[voucher]["starting_time_to_expiry"] - (state.timestamp / 1000000 / 7)
            if tte <= 0:
                continue
            # Compute implied volatility and delta using the underlying mid price
            vol = BlackScholes.implied_volatility(mid_price,
                                                  underlying_mid,
                                                  self.params[voucher]["strike"],
                                                  tte)
            delta_val = BlackScholes.delta(underlying_mid,
                                           self.params[voucher]["strike"],
                                           tte,
                                           vol)
            # Store data for spread trading
            voucher_data[voucher] = {
                "mid": mid_price,
                "vol": vol,
                "tte": tte,
                "strike": self.params[voucher]["strike"],
                "order_depth": od,
                "position": position,
            }
            # Process individual directional orders from voucher_orders
            take_orders, make_orders = self.voucher_orders(voucher, od, position, traderInfo, vol)
            if take_orders or make_orders:
                result.setdefault(voucher, [])
                if take_orders:
                    result[voucher].extend(take_orders)
                if make_orders:
                    result[voucher].extend(make_orders)
                executed = sum(o.quantity for o in take_orders) if take_orders else 0
                position += executed
            net_delta_exposure += position * delta_val

        # Hedge underlying exposure based on aggregate delta
        underlying_position = state.position.get(Product.VOLCANICROCK, 0)
        hedge_orders = self.compute_underlying_hedge_orders(underlying_od, underlying_position, net_delta_exposure)
        if hedge_orders:
            result[Product.VOLCANICROCK] = hedge_orders

        # Process vertical spread opportunities across vouchers
        spread_orders = self.spread_voucher_orders(voucher_data, underlying_mid)
        for prod, orders in spread_orders.items():
            result.setdefault(prod, []).extend(orders)

        

        traderDataEncoded = jsonpickle.encode(traderObject)
        logger.flush(state, result, conversions, traderDataEncoded)
        return result, 1, traderDataEncoded
