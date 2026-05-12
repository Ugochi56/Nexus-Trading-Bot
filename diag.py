import sys
import os
import MetaTrader5 as mt5

from core.config import SYMBOL, MAGIC_NUMBER

def test():
    if not mt5.initialize():
        print("Init failed")
        return
        
    tick = mt5.symbol_info_tick(SYMBOL)
    if not tick:
        print("No tick")
        return
        
    price = tick.ask
    limit_price = price - 5.0
    sl = limit_price - 3.0
    tp = limit_price + 3.0
    
    request = {
        "action": mt5.TRADE_ACTION_PENDING,
        "symbol": SYMBOL,
        "volume": 0.01,
        "type": mt5.ORDER_TYPE_BUY_LIMIT,
        "price": limit_price,
        "sl": sl,
        "tp": tp,
        "deviation": 20,
        "magic": MAGIC_NUMBER,
        "comment": "Diag limit",
        "type_time": mt5.ORDER_TIME_GTC,
        "type_filling": mt5.ORDER_FILLING_RETURN,
    }
    
    print("Checking order...")
    check = mt5.order_check(request)
    if check:
        print(f"Check result: retcode={check.retcode}, margin={check.margin}, margin_free={check.margin_free}, comment={check.comment}")
    else:
        print(f"Check returned None! MT5 Error: {mt5.last_error()}")
        
    print("Sending order...")
    result = mt5.order_send(request)
    if result is None:
        print(f"Send returned None! MT5 Error: {mt5.last_error()}")
    else:
        print(f"Result: {result.retcode} {result.comment}")
        
    mt5.shutdown()

if __name__ == "__main__":
    test()
