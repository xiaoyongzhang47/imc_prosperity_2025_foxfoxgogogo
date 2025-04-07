from datamodel import OrderDepth, UserId, TradingState, Order
from typing import List
import string

class Trader:
    
    def run(self, state: TradingState):
        print("traderData: " + state.traderData)
        print("Observations: " + str(state.observations))
        
        result = {}
        # Initialize Kelp historical prices from traderData
        kelp_prices = []
        if state.traderData:
            try:
                kelp_prices = list(map(float, state.traderData.split(',')))
            except:
                kelp_prices = []
        
        for product in state.order_depths:
            order_depth: OrderDepth = state.order_depths[product]
            orders: List[Order] = []
            current_position = state.position.get(product, 0)
            
            if product == 'RAINFOREST_RESIN':
                # Calculate acceptable price as mid-price
                if len(order_depth.buy_orders) == 0 or len(order_depth.sell_orders) == 0:
                    continue  # Skip if no orders on either side
                best_bid = max(order_depth.buy_orders.keys())
                best_ask = min(order_depth.sell_orders.keys())
                best_bid = int(best_bid)
                best_ask = int(best_ask)
                mid_price = (best_bid + best_ask) / 2
                acceptable_price = mid_price
                
            elif product == 'KELP':
                # Calculate mid-price and update SMA
                if len(order_depth.buy_orders) == 0 or len(order_depth.sell_orders) == 0:
                    continue  # Skip if no orders on either side
                best_bid = max(order_depth.buy_orders.keys())
                best_ask = min(order_depth.sell_orders.keys())
                best_bid = int(best_bid)
                best_ask = int(best_ask)
                mid_price = (best_bid + best_ask) / 2
                kelp_prices.append(mid_price)
                # Keep up to the last 5 prices
                kelp_prices = kelp_prices[-5:]
                # Compute SMA
                if len(kelp_prices) == 0:
                    acceptable_price = mid_price  # Fallback if no data
                else:
                    acceptable_price = sum(kelp_prices) / len(kelp_prices)
            else:
                continue  # Skip other products
            
            print(f"Acceptable price for {product}: {acceptable_price}")
            print(f"Current position for {product}: {current_position}")
            
            # Process sell orders (asks)
            if len(order_depth.sell_orders) > 0:
                best_ask, best_ask_amount = list(order_depth.sell_orders.items())[0]
                best_ask = int(best_ask)
                best_ask_amount = int(best_ask_amount)
                if best_ask < acceptable_price:
                    max_buy = 50 - current_position
                    if max_buy > 0:
                        buy_qty = min(best_ask_amount, max_buy)
                        orders.append(Order(product, best_ask, buy_qty))
                        print(f"BUY {buy_qty}x {best_ask} for {product}")
            
            # Process buy orders (bids)
            if len(order_depth.buy_orders) > 0:
                best_bid, best_bid_amount = list(order_depth.buy_orders.items())[0]
                best_bid = int(best_bid)
                best_bid_amount = int(best_bid_amount)
                if best_bid > acceptable_price:
                    max_sell = current_position + 50  # current_pos - qty >= -50 => qty <= current_pos +50
                    sell_qty = min(best_bid_amount, max_sell)
                    if sell_qty > 0:
                        orders.append(Order(product, best_bid, -sell_qty))
                        print(f"SELL {sell_qty}x {best_bid} for {product}")
            
            result[product] = orders
        
        # Update traderData with Kelp's historical prices
        traderData = ','.join(map(str, kelp_prices))
        conversions = 0  # No conversions in this example
        
        return result, conversions, traderData
