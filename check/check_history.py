import sys
import os
import MetaTrader5 as mt5
from datetime import datetime, timedelta

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.config import SYMBOL

def main():
    if not mt5.initialize():
        print("Failed to initialize MT5")
        sys.exit()

    print("\n=======================================================")
    print("                NEXUS TRADE HISTORY SCANNER            ")
    print("=======================================================")

    date_from = datetime.now() - timedelta(days=7)
    date_to = datetime.now() + timedelta(days=1)
    
    orders = mt5.history_orders_get(date_from, date_to, group="*")
    if orders is None or len(orders) == 0:
        print("No historical orders found.")
        return
        
    print(f"Analyzing the last {len(orders)} historical orders...")
    print("-" * 60)
    
    for o in orders[-20:]: # Just look at the last 20 orders to prevent spam
        if o.symbol != SYMBOL: continue
        
        o_type = "UNKNOWN"
        if o.type == mt5.ORDER_TYPE_BUY: o_type = "BUY (Market)"
        elif o.type == mt5.ORDER_TYPE_SELL: o_type = "SELL (Market)"
        elif o.type == mt5.ORDER_TYPE_BUY_LIMIT: o_type = "BUY LIMIT"
        elif o.type == mt5.ORDER_TYPE_SELL_LIMIT: o_type = "SELL LIMIT"
        elif o.type == mt5.ORDER_TYPE_BUY_STOP: o_type = "BUY STOP"
        elif o.type == mt5.ORDER_TYPE_SELL_STOP: o_type = "SELL STOP"
        
        # Calculate Breathing Room (Distance between Entry and SL)
        if o.price_open > 0 and o.sl > 0:
            breathing_room = abs(o.price_open - o.sl)
        else:
            breathing_room = 0.0
            
        time_setup = datetime.fromtimestamp(o.time_setup)
        
        print(f"Time:    {time_setup}")
        print(f"Ticket:  {o.ticket} | {o_type} | Vol: {o.volume_initial} lots")
        print(f"Entry:   ${o.price_open:.2f}")
        print(f"SL:      ${o.sl:.2f} (Breathing Room: ${breathing_room:.2f})")
        print(f"TP:      ${o.tp:.2f}")
        print(f"Comment: {o.comment}")
        print("-" * 60)

    print("=======================================================\n")
    mt5.shutdown()

if __name__ == "__main__":
    main()
