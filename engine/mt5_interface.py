import MetaTrader5 as mt5
import pandas as pd
import pandas_ta as ta
import sys
from datetime import datetime
from core.config import *
from core.indicators import *
from core.state_manager import nexus_state

virtual_positions = {}
virtual_ticket_counter = 1000

daily_start_equity = 0.0
last_day_checked = -1

def connect_mt5():
    if not mt5.initialize():
        print(f"\n[ERROR] MT5 Init Failed")
        sys.exit()
    if not mt5.symbol_select(SYMBOL, True):
        print(f"\n[ERROR] Failed to select {SYMBOL}")
        sys.exit()
    account_info = mt5.account_info()
    if account_info:
        print(f"\n[SUCCESS] Connected: {account_info.login} | ${account_info.balance:,.2f}")

def check_daily_drawdown():
    account = mt5.account_info()
    if not account: return False
    
    current_day = datetime.now().day
    cached_day = nexus_state.get('last_day_checked', -1)
    
    if current_day != cached_day:
        nexus_state.set('daily_start_equity', account.equity)
        nexus_state.set('last_day_checked', current_day)
        nexus_state.set('is_halted', False)
        print(f"\n[DAY] New Day Equity: ${account.equity:.2f} (Written to Drive)")
        return True
        
    if nexus_state.get('is_halted', False):
        print(f"\r[HALT] Terminal structurally locked for the day. Wait for reset.".ljust(90), end='')
        return False
        
    start_equity = nexus_state.get('daily_start_equity', account.equity)
    current_equity = account.equity
    loss_percent = ((start_equity - current_equity) / start_equity) * 100
    
    if loss_percent >= MAX_DAILY_LOSS_PERCENT:
        print(f"\n[HALT] DAILY LIMIT HIT! (-{loss_percent:.2f}%). EXECUTING GLOBAL LIQUIDATION...")
        nexus_state.set('is_halted', True)
        close_all_positions(reason="DAILY_LIMIT_REACHED")
        return False 
    return True

def check_volatility_guard(df_m5):
    if not USE_VOLATILITY_GUARD: return True, 1.0
    baseline_atr = ta.atr(df_m5['high'], df_m5['low'], df_m5['close'], length=ATR_BASELINE_PERIOD).iloc[-1]
    recent_range = (df_m5['high'] - df_m5['low']).rolling(5).mean().iloc[-1]
    if baseline_atr == 0: return True, 1.0
    volatility_ratio = recent_range / baseline_atr
    if volatility_ratio > VOLATILITY_THRESHOLD: return False, volatility_ratio
    return True, volatility_ratio

def get_market_data(symbol, timeframe, n=1000):
    rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, n)
    if rates is None: return None
    df = pd.DataFrame(rates)
    df['time'] = pd.to_datetime(df['time'], unit='s')
    df['high'] = df['high'].astype(float)
    df['low'] = df['low'].astype(float)
    df['close'] = df['close'].astype(float)
    df['open'] = df['open'].astype(float)
    df['tick_volume'] = df['tick_volume'].astype(float) 
    return df

def get_dynamic_kelly_risk(base_risk):
    if DRY_RUN: return base_risk
    
    time_from = datetime.now().astimezone() - pd.Timedelta(hours=24)
    deals = mt5.history_deals_get(time_from, datetime.now().astimezone())
    if deals is None or len(deals) == 0: return base_risk
    
    # Isolate out-deals (closed positions) by this exact bot
    bot_deals = [d for d in deals if d.magic == MAGIC_NUMBER and d.entry == mt5.DEAL_ENTRY_OUT]
    if len(bot_deals) < 2: return base_risk
    
    # Track discrete win/loss sequences (ignoring tiny scratch BE trades under $0.50)
    outcomes = []
    for d in bot_deals:
        if d.profit > 0.50: outcomes.append(1)
        elif d.profit < -0.50: outcomes.append(-1)
        
    if len(outcomes) >= 3 and sum(outcomes[-3:]) == 3:
        return base_risk * 1.5
    elif len(outcomes) >= 2 and sum(outcomes[-2:]) == -2:
        return base_risk * 0.5
        
    return base_risk

def calculate_position_size(entry_price, sl_price, risk_pct):
    account, symbol_info = mt5.account_info(), mt5.symbol_info(SYMBOL)
    if not account or not symbol_info: return 0.01
    sl_dist = abs(entry_price - sl_price)
    if sl_dist == 0: return 0.01
    lots = (account.balance * (risk_pct / 100)) / (sl_dist * symbol_info.trade_contract_size)
    step = symbol_info.volume_step
    lots = round(lots / step) * step
    return max(symbol_info.volume_min, min(lots, symbol_info.volume_max))

def execute_trade(signal, sl_price, risk_pct, magic_num, comment_text, ai_conf=0.55):
    global virtual_ticket_counter
    
    if DYNAMIC_RISK:
        scale = max(0.5, min(1.5, (ai_conf - AI_CONFIDENCE_THRESHOLD) / (0.95 - AI_CONFIDENCE_THRESHOLD) * 1.0 + 0.5))
        risk_pct = risk_pct * scale

    if risk_pct <= 0: return False
    tick = mt5.symbol_info_tick(SYMBOL)
    price = tick.ask if signal == 'BUY' else tick.bid
    lots = calculate_position_size(price, sl_price, risk_pct)
    risk_dist = abs(price - sl_price)
    tp = price + (risk_dist * RISK_REWARD_RATIO) if signal == 'BUY' else price - (risk_dist * RISK_REWARD_RATIO)
    
    print(f"\n[EXEC] ({comment_text}) | Risk {risk_pct:.2f}%")
    print(f"       Entry: {price:.2f} | SL: {sl_price:.2f} | TP: {tp:.2f} | Lots: {lots}")
    
    if DRY_RUN:
        virtual_ticket_counter += 1
        virtual_positions[virtual_ticket_counter] = {
            'ticket': virtual_ticket_counter,
            'type': mt5.ORDER_TYPE_BUY if signal == 'BUY' else mt5.ORDER_TYPE_SELL,
            'price_open': price,
            'sl': sl_price,
            'tp': tp,
            'lots': lots,
            'magic': magic_num,
            'comment': comment_text
        }
        print(f"[VIRTUAL] ORDER PLACED! Ticket: {virtual_ticket_counter}")
        return True
    
    request = {
        "action": mt5.TRADE_ACTION_DEAL, "symbol": SYMBOL, "volume": lots, 
        "type": mt5.ORDER_TYPE_BUY if signal == 'BUY' else mt5.ORDER_TYPE_SELL,
        "price": price, "sl": sl_price, "tp": tp, "deviation": DEVIATION, 
        "magic": magic_num, "comment": comment_text,
        "type_time": mt5.ORDER_TIME_GTC, "type_filling": mt5.ORDER_FILLING_IOC,
    }
    result = mt5.order_send(request)
    if result is None:
        print("[ERROR] Order Failed: mt5.order_send returned None (Network Timeout)")
        return False
    if result.retcode != mt5.TRADE_RETCODE_DONE:
        print(f"[ERROR] Order Failed: {result.comment}")
        return False
    print(f"[TRADE] ORDER PLACED! Ticket: {result.order}")
    return True

def manage_open_positions():
    if DRY_RUN:
        tick = mt5.symbol_info_tick(SYMBOL)
        if not tick: return
        closed_tickets = []
        for v_id, pos in virtual_positions.items():
            curr_price = tick.bid if pos['type'] == mt5.ORDER_TYPE_BUY else tick.ask
            pos_profit_dist = (curr_price - pos['price_open']) if pos['type'] == mt5.ORDER_TYPE_BUY else (pos['price_open'] - curr_price)
            tp_dist = abs(pos['tp'] - pos['price_open'])
            
            if MOVE_BE and tp_dist > 0 and (pos_profit_dist / tp_dist) >= BE_TRIGGER_PCT:
                if pos.get('be_moved') is None:
                    pos['sl'] = pos['price_open']
                    pos['be_moved'] = True
                    if PARTIAL_CLOSE_PCT > 0:
                        closed_lots = pos['lots'] * PARTIAL_CLOSE_PCT
                        pos['lots'] -= closed_lots
                        profit = (pos_profit_dist) * closed_lots * 100
                        print(f"\n[PARTIAL] VIRTUAL CLOSE [Ticket {v_id}] | PnL: +{profit:.2f} | BE Hit")
                        
            if pos['type'] == mt5.ORDER_TYPE_BUY:
                if curr_price <= pos['sl']:
                    profit = (pos['sl'] - pos['price_open']) * pos['lots'] * 100
                    print(f"\n[CLOSED] VIRTUAL SL HIT [Ticket {v_id}] | PnL: {profit:.2f}")
                    closed_tickets.append(v_id); continue
                if curr_price >= pos['tp']:
                    profit = (pos['tp'] - pos['price_open']) * pos['lots'] * 100
                    print(f"\n[CLOSED] VIRTUAL TP HIT [Ticket {v_id}] | PnL: +{profit:.2f}")
                    closed_tickets.append(v_id); continue
                
                if USE_TRAILING_STOP and (curr_price - pos['price_open']) > TRAIL_START_DOLLARS:
                    new_sl = curr_price - TRAIL_DIST_DOLLARS
                    if new_sl > pos['sl'] and new_sl > pos['price_open']: pos['sl'] = new_sl
                if USE_TRAILING_TP and (pos['tp'] - curr_price) < TP_TRIGGER_DIST:
                     pos['tp'] += TP_EXTENSION_DIST

            elif pos['type'] == mt5.ORDER_TYPE_SELL:
                if curr_price >= pos['sl']:
                    profit = (pos['price_open'] - pos['sl']) * pos['lots'] * 100
                    print(f"\n[CLOSED] VIRTUAL SL HIT [Ticket {v_id}] | PnL: {profit:.2f}")
                    closed_tickets.append(v_id); continue
                if curr_price <= pos['tp']:
                    profit = (pos['price_open'] - pos['tp']) * pos['lots'] * 100
                    print(f"\n[CLOSED] VIRTUAL TP HIT [Ticket {v_id}] | PnL: +{profit:.2f}")
                    closed_tickets.append(v_id); continue

                if USE_TRAILING_STOP and (pos['price_open'] - curr_price) > TRAIL_START_DOLLARS:
                    new_sl = curr_price + TRAIL_DIST_DOLLARS
                    if (pos['sl'] == 0 or new_sl < pos['sl']) and new_sl < pos['price_open']: pos['sl'] = new_sl
                if USE_TRAILING_TP and (curr_price - pos['tp']) < TP_TRIGGER_DIST:
                     pos['tp'] -= TP_EXTENSION_DIST
                     
        for t in closed_tickets:
            del virtual_positions[t]
        return
        
    positions = mt5.positions_get(symbol=SYMBOL)
    if not positions: return
    for pos in positions:
        if pos.magic != MAGIC_NUMBER: continue
        curr_price = mt5.symbol_info_tick(SYMBOL).bid if pos.type == mt5.ORDER_TYPE_BUY else mt5.symbol_info_tick(SYMBOL).ask
        pos_profit_dist = (curr_price - pos.price_open) if pos.type == mt5.ORDER_TYPE_BUY else (pos.price_open - curr_price)
        tp_dist = abs(pos.tp - pos.price_open)
        
        if MOVE_BE and tp_dist > 0 and (pos_profit_dist / tp_dist) >= BE_TRIGGER_PCT:
            be_condition = pos.sl >= pos.price_open if pos.type == mt5.ORDER_TYPE_BUY else (pos.sl <= pos.price_open and pos.sl > 0)
            if not be_condition:
                request_sl = {
                    "action": mt5.TRADE_ACTION_SLTP, "position": pos.ticket, 
                    "sl": pos.price_open, "tp": pos.tp, "magic": pos.magic
                }
                res_sl = mt5.order_send(request_sl)
                if res_sl is None or res_sl.retcode != mt5.TRADE_RETCODE_DONE:
                    print(f"[WARNING] BE Move Request Failed for Ticket {pos.ticket}")
                
                if PARTIAL_CLOSE_PCT > 0:
                    action_type = mt5.ORDER_TYPE_SELL if pos.type == mt5.ORDER_TYPE_BUY else mt5.ORDER_TYPE_BUY
                    close_price = mt5.symbol_info_tick(SYMBOL).ask if action_type == mt5.ORDER_TYPE_BUY else mt5.symbol_info_tick(SYMBOL).bid
                    vol_to_close = pos.volume * PARTIAL_CLOSE_PCT
                    step = mt5.symbol_info(SYMBOL).volume_step
                    vol_to_close = round(vol_to_close / step) * step
                    if vol_to_close >= mt5.symbol_info(SYMBOL).volume_min:
                         req_partial = {
                            "action": mt5.TRADE_ACTION_DEAL, "symbol": pos.symbol, "volume": vol_to_close,
                            "type": action_type, "position": pos.ticket, "price": close_price, "deviation": DEVIATION, "magic": pos.magic,
                            "type_time": mt5.ORDER_TIME_GTC, "type_filling": mt5.ORDER_FILLING_IOC,
                         }
                         res = mt5.order_send(req_partial)
                         if res and res.retcode == mt5.TRADE_RETCODE_DONE:
                             print(f"\n[PARTIAL] LIVE CLOSE [Ticket {pos.ticket}] | BE Hit")
                             
        if USE_TRAILING_STOP:
            if pos.type == mt5.ORDER_TYPE_BUY and (curr_price - pos.price_open) > TRAIL_START_DOLLARS:
                new_sl = curr_price - TRAIL_DIST_DOLLARS
                if new_sl > pos.sl and new_sl > pos.price_open:
                    req = {"action": mt5.TRADE_ACTION_SLTP, "position": pos.ticket, "sl": new_sl, "tp": pos.tp, "magic": pos.magic}
                    mt5.order_send(req)
            elif pos.type == mt5.ORDER_TYPE_SELL and (pos.price_open - curr_price) > TRAIL_START_DOLLARS:
                new_sl = curr_price + TRAIL_DIST_DOLLARS
                if (pos.sl == 0.0 or new_sl < pos.sl) and new_sl < pos.price_open:
                    req = {"action": mt5.TRADE_ACTION_SLTP, "position": pos.ticket, "sl": new_sl, "tp": pos.tp, "magic": pos.magic}
                    mt5.order_send(req)

def close_all_positions(reason="FRIDAY_PROTOCOL"):
    if DRY_RUN:
        virtual_positions.clear()
        print("\n[LIQUIDATED] DRY_RUN: All Virtual Positions Liquidated.")
        return
        
    positions = mt5.positions_get(symbol=SYMBOL)
    if not positions: return
    for pos in positions:
        if pos.magic != MAGIC_NUMBER: continue
        tick = mt5.symbol_info_tick(SYMBOL)
        action_type = mt5.ORDER_TYPE_SELL if pos.type == mt5.ORDER_TYPE_BUY else mt5.ORDER_TYPE_BUY
        price = tick.bid if action_type == mt5.ORDER_TYPE_SELL else tick.ask
        request = {
            "action": mt5.TRADE_ACTION_DEAL, "symbol": SYMBOL, "volume": pos.volume, 
            "type": action_type, "position": pos.ticket, "price": price, 
            "deviation": DEVIATION, "magic": MAGIC_NUMBER, "comment": reason,
            "type_time": mt5.ORDER_TIME_GTC, "type_filling": mt5.ORDER_FILLING_IOC,
        }
        res = mt5.order_send(request)
        if res and res.retcode == mt5.TRADE_RETCODE_DONE:
            print(f"\n[CLOSED] Ticket #{pos.ticket} ({reason})")
