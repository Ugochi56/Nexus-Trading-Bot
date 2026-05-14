import MetaTrader5 as mt5
import sys
import os
import datetime

# Add project root to sys.path so config can be loaded
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from core.config import SYMBOL

def main():
    if not mt5.initialize():
        print("Failed to connect to MT5. Is the terminal open?")
        sys.exit()

    print("\n=======================================================")
    print("                NEXUS ACTIVE TRADE SCANNER             ")
    print("=======================================================")

    # Fetch Active Market Positions
    positions = mt5.positions_get(symbol=SYMBOL)
    if positions is None or len(positions) == 0:
        print("\n[ACTIVE POSITIONS] (Live Trades)")
        print("  -> None. You have no active live trades running.")
    else:
        print(f"\n[ACTIVE POSITIONS] ({len(positions)} Running)")
        print("-" * 55)
        for p in positions:
            p_type = "BUY" if p.type == mt5.POSITION_TYPE_BUY else "SELL"
            open_time = datetime.datetime.fromtimestamp(p.time)
            print(f"Ticket: {p.ticket} | Type: {p_type} | Vol: {p.volume} lots")
            print(f"  Entry Price: ${p.price_open:.2f}  |  Current Price: ${p.price_current:.2f}")
            print(f"  Stop Loss:   ${p.sl:.2f}  |  Take Profit:   ${p.tp:.2f}")
            
            # Format profit in green/red if possible, or just standard
            profit_str = f"+${p.profit:.2f}" if p.profit >= 0 else f"-${abs(p.profit):.2f}"
            print(f"  Net Profit:  {profit_str}")
            print(f"  Open Time:   {open_time}")
            print(f"  Comment:     {p.comment}")
            print("-" * 55)

    # Fetch Pending Limit/Stop Orders
    orders = mt5.orders_get(symbol=SYMBOL)
    if orders is None or len(orders) == 0:
        print("\n[PENDING ORDERS] (Limit / Stop)")
        print("  -> None. You have no pending orders waiting.")
    else:
        print(f"\n[PENDING ORDERS] ({len(orders)} Waiting in queue)")
        print("-" * 55)
        for o in orders:
            o_type_str = "UNKNOWN"
            if o.type == mt5.ORDER_TYPE_BUY_LIMIT: o_type_str = "BUY LIMIT"
            elif o.type == mt5.ORDER_TYPE_SELL_LIMIT: o_type_str = "SELL LIMIT"
            elif o.type == mt5.ORDER_TYPE_BUY_STOP: o_type_str = "BUY STOP"
            elif o.type == mt5.ORDER_TYPE_SELL_STOP: o_type_str = "SELL STOP"
            
            setup_time = datetime.datetime.fromtimestamp(o.time_setup)
            print(f"Ticket: {o.ticket} | Type: {o_type_str} | Vol: {o.volume_initial} lots")
            print(f"  Target Entry Price: ${o.price_open:.2f}")
            print(f"  Stop Loss:          ${o.sl:.2f}  |  Take Profit: ${o.tp:.2f}")
            print(f"  Setup Time:         {setup_time}")
            print(f"  Comment:            {o.comment}")
            print("-" * 55)

    print("\n=======================================================\n")
    mt5.shutdown()

if __name__ == "__main__":
    main()
