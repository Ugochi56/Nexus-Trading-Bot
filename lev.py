import MetaTrader5 as mt5
import sys

def main():
    # 1. Connect to MT5
    if not mt5.initialize():
        print(f"❌ MT5 Init Failed: {mt5.last_error()}")
        sys.exit()
    
    # 2. Get Account Info
    account_info = mt5.account_info()
    
    if account_info is None:
        print("❌ Failed to retrieve account info")
        mt5.shutdown()
        sys.exit()

    # 3. Extract Leverage and Margin Details
    leverage = account_info.leverage
    balance = account_info.balance
    equity = account_info.equity
    margin_free = account_info.margin_free
    
    print("\n📊 ACCOUNT DETAILS")
    print(f"   Login:    {account_info.login}")
    print(f"   Server:   {account_info.server}")
    print(f"   Balance:  ${balance:,.2f}")
    print(f"   Equity:   ${equity:,.2f}")
    print("-" * 30)
    print(f"⚙️ ACCOUNT LEVERAGE: 1:{leverage}")
    print("-" * 30)
    
    # 4. Calculate Buying Power (Approximate)
    # Buying Power = Free Margin * Leverage
    buying_power = margin_free * leverage
    print(f"💪 Buying Power: ${buying_power:,.2f}")

    # 5. Calculate Real-Time Effective Leverage (Risk Exposure)
    positions = mt5.positions_get()
    if positions:
        total_notional_value = 0.0
        print("\nOPEN POSITIONS:")
        for pos in positions:
            symbol_info = mt5.symbol_info(pos.symbol)
            if symbol_info:
                # Value = Volume * ContractSize * CurrentPrice
                # For Forex/Metals, contract_size is usually 100,000 or 100.
                # Adjust calculation based on symbol type if needed, but standard is:
                notional = pos.volume * symbol_info.trade_contract_size * pos.price_current
                # If non-USD account, convert notional to account currency (omitted for simplicity)
                total_notional_value += notional
                print(f"   - {pos.symbol}: {pos.volume} lots (~${notional:,.2f})")
        
        effective_leverage = total_notional_value / equity
        print("-" * 30)
        print(f"⚠️ EFFECTIVE LEVERAGE: {effective_leverage:.2f}x")
        print(f"   (Total Exposure: ${total_notional_value:,.2f})")
    else:
        print("\n✅ No open positions. Effective Leverage: 0.0x")

    mt5.shutdown()

if __name__ == "__main__":
    main()