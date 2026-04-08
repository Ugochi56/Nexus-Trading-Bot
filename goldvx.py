import time
import sys
import joblib
from datetime import datetime
from core.config import *
from core.utils import is_us_dst, get_session_name
from core.indicators import calculate_atr_simple, calculate_rsi_simple, calculate_adx_simple
from engine.mt5_interface import connect_mt5, check_daily_drawdown, get_market_data, execute_trade, manage_open_positions, check_volatility_guard, close_all_positions
from engine.news_filter import fetch_economic_news, is_news_blackout
from strategies.smc_fvg import SMCStrategy
from strategies.rsi_reversion import RSIReversionStrategy
from strategies.smc_orderblock import SMCOrderBlockStrategy
from strategies.vwap_reversion import VWAPReversionStrategy
from strategies.bb_breakout import BBBreakoutStrategy
import MetaTrader5 as mt5

def load_ai_models():
    try:
        trend = joblib.load(TREND_MODEL_FILE)
        reversal = joblib.load(REVERSAL_MODEL_FILE)
        return trend, reversal
    except Exception as e:
        print(f"[ERROR] Error loading models: {e}")
        return None, None

def main():
    print("[START] Booting NEXUS Core...")
    connect_mt5()
    
    trend_model, reversal_model = load_ai_models()
    if trend_model is None:
        print("[CRITICAL] Engine shutting down due to Missing Intelligence (Joblib)")
        sys.exit()

    # Instantiate Strategies
    strategies = {
        "SMC_FVG": SMCStrategy(trend_model),
        "SMC_OB": SMCOrderBlockStrategy(trend_model),
        "RSI_REVERSION": RSIReversionStrategy(reversal_model),
        "VWAP_REVERSION": VWAPReversionStrategy(),
        "BB_BREAKOUT": BBBreakoutStrategy()
    }
    
    last_eval_time = 0 
    last_m5_time = None
    df_m5, df_h1, df_h4, df_adx = None, None, None, None
    curr_adx, current_atr, curr_rsi = 0, 0, 0
    vol_ratio = 1.0
    is_safe = True
    last_heartbeat_min = -1
    last_equity_check_time = 0
    cached_equity = 0.0
    market_closed_for_weekend = False
    
    prev_strats_str = ""
    
    print("\n" + "="*55)
    print("[NEXUS ONLINE]")
    print("[ENGINES: SMC_FVG, SMC_OB, RSI, VWAP, BB_BREAKOUT]")
    print("="*55 + "\n")

    while True:
        try:
            if not check_daily_drawdown(): time.sleep(60); continue

            tick = mt5.symbol_info_tick(SYMBOL)
            if tick is None: time.sleep(1); continue
            curr_price = tick.bid
            
            now_ts = time.time()
            if now_ts - last_equity_check_time > 60:
                acc_info = mt5.account_info()
                if acc_info: cached_equity = acc_info.equity
                last_equity_check_time = now_ts
            
            manage_open_positions()

            now_ts = time.time()
            if (now_ts - last_eval_time) > 3.0:
                rates = mt5.copy_rates_from_pos(SYMBOL, TIMEFRAME_ENTRY, 0, 1)
                fetch_economic_news()
                
                if rates is not None and rates[0]['time'] != last_m5_time:
                    df_m5 = get_market_data(SYMBOL, TIMEFRAME_ENTRY, n=1000)
                    df_h1 = get_market_data(SYMBOL, TIMEFRAME_TREND, n=100)
                    df_h4 = get_market_data(SYMBOL, mt5.TIMEFRAME_H4, n=100)
                    df_adx = get_market_data(SYMBOL, TIMEFRAME_ADX, n=100)
                    if df_m5 is not None: last_m5_time = df_m5.iloc[-1]['time']
                else:
                    df_m5_light = get_market_data(SYMBOL, TIMEFRAME_ENTRY, n=200)
                    if df_m5_light is not None: df_m5 = df_m5_light
                
                last_eval_time = now_ts
                
                if df_m5 is not None and df_h1 is not None and df_adx is not None:
                    is_safe, vol_ratio = check_volatility_guard(df_m5)
                    if not is_safe:
                        curr_min = datetime.now().minute
                        if curr_min != last_heartbeat_min:
                            print(f"\n[HALT] WAR MODE: Volatility {vol_ratio:.1f}x. HALTED.")
                            last_heartbeat_min = curr_min
                        continue

                    current_atr = calculate_atr_simple(df_m5, 14)
                    df_m5['rsi'] = calculate_rsi_simple(df_m5['close'], RSI_PERIOD)
                    curr_rsi = df_m5.iloc[-1]['rsi']
                    curr_adx = calculate_adx_simple(df_adx, ADX_PERIOD).iloc[-1]

            if df_m5 is None or df_h1 is None or df_h4 is None: time.sleep(1); continue
            if not is_safe: time.sleep(1); continue
            
            if is_news_blackout():
                print(f"\r[SLEEP] RED NEWS BLACKOUT ACTIVE".ljust(90), end='')
                time.sleep(10); continue
                
            server_time = datetime.fromtimestamp(tick.time)
            server_hour = server_time.hour
            
            if CLOSE_ALL_ON_FRIDAY and server_time.weekday() == 4 and server_hour >= FRIDAY_CLOSE_HOUR:
                if not market_closed_for_weekend:
                    close_all_positions()
                    market_closed_for_weekend = True
                print(f"\r[SLEEP] Market Closed for Weekend (Kill-Switch Active)".ljust(90), end='')
                time.sleep(60); continue
            
            if market_closed_for_weekend and server_time.weekday() not in [4, 5]:
                market_closed_for_weekend = False

            is_dst_active = False
            if AUTO_DST_ADJUST and is_us_dst(server_time):
                server_hour = (server_hour + DST_SHIFT_HOURS) % 24
                is_dst_active = True
            
            current_session = get_session_name(server_hour)
            sess_map = {"ASIAN": "ASIA", "LONDON": "LDN", "NY_LONDON_OVERLAP": "OLAP", "NEW_YORK": "NY", "ROLLOVER": "ROLL"}
            short_session = sess_map.get(current_session, current_session)
            current_risk = SESSION_RISK.get(current_session, 0.0)
            
            dst_tag = "[DST]" if is_dst_active else ""
            if server_hour < ASIAN_OPEN_HOUR or server_hour >= TRADING_END_HOUR:
                print(f"\r[SLEEP] Market Closed (Hour: {server_hour} {dst_tag})".ljust(90), end='')
                time.sleep(60); continue

            # --- Strategy Routing ---
            regime_icon = "[NEUTRAL]"
            active_strats = ["SMC_FVG", "SMC_OB"]
            if AUTO_SWITCH:
                if current_session == "NY_LONDON_OVERLAP":
                    active_strats = ["FLAT"]; regime_icon = "[FLAT]"
                elif current_session in ["LONDON", "NEW_YORK"]:
                     active_strats = ["SMC_FVG", "SMC_OB", "BB_BREAKOUT", "VWAP_REVERSION"]; regime_icon = "[HOT]" 
                else:
                    if curr_adx > ADX_TREND_START: active_strats = ["SMC_FVG", "SMC_OB", "BB_BREAKOUT"]; regime_icon = "[TREND]"
                    elif curr_adx < ADX_RANGE_START: active_strats = ["RSI_REVERSION", "VWAP_REVERSION"]; regime_icon = "[RANGE]"
                    else: active_strats = ["RSI_REVERSION", "VWAP_REVERSION", "BB_BREAKOUT"]; regime_icon = "[NEUTRAL]"

            curr_strats_str = ",".join(active_strats)
            if curr_strats_str != prev_strats_str:
                print(f"\n[CHANGE] Switched state to {regime_icon}. Active Engines: {' | '.join(active_strats)}")
                prev_strats_str = curr_strats_str

            curr_trend = "FLAT"
            if df_h1 is not None and len(df_h1) >= 50:
                ema_50 = df_h1['close'].ewm(span=50, adjust=False).mean().iloc[-1]
                close_h1 = df_h1.iloc[-1]['close']
                curr_trend = "UP" if close_h1 > ema_50 else "DOWN"

            status_base = f"\r[{short_session}{dst_tag}] {regime_icon} Trend:{curr_trend} | ADX:{curr_adx:.1f} | Bal: ${cached_equity:,.2f} | Prc: ${curr_price:.2f}"
            
            if "FLAT" not in active_strats:
                # ORCHESTRATOR: MAXIMUM CONFLUENCE EVALUATOR
                for strat_key in active_strats:
                    strategy = strategies.get(strat_key)
                    if strategy:
                        res = strategy.evaluate(
                            df_m5=df_m5, df_h1=df_h1, df_h4=df_h4, df_adx=df_adx, 
                            current_price=curr_price, current_risk=current_risk, atr=current_atr
                        )
                        payload = res.get('payload')
                        
                        if payload:
                            r_pct = payload.get('risk_override', current_risk)
                            execute_trade(
                                signal=payload['signal'], 
                                sl_price=payload['sl'], 
                                risk_pct=r_pct, 
                                magic_num=MAGIC_NUMBER, 
                                comment_text=payload['comment'], 
                                ai_conf=payload['confidence']
                            )

            print(f"{status_base.ljust(90)}", end='')
                
            time.sleep(0.5)

        except KeyboardInterrupt:
            mt5.shutdown(); break
        except Exception as e:
            print(f"\n❌ Error: {e}"); time.sleep(5)

if __name__ == "__main__":
    main()