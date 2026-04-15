class BaseStrategy:
    def __init__(self, name="BASE"):
        self.name = name

    def evaluate(self, df_m5, df_h1, df_h4, df_adx, current_price, current_risk, **kwargs):
        """
        Evaluates market data and returns a trading signal dictionary.
        Must be implemented by child strategies.
        
        Returns:
            dict: {
                'signal': 'BUY' | 'SELL' | 'FLAT',
                'sl': float,
                'tp': float (optional, if calculated inside strategy),
                'confidence': float,
                'comment': str
            }
        """
        raise NotImplementedError("Strategy must implement evaluate() method")

    def reset(self):
        """
        Flushes all active algorithmic memory (strats/arrays).
        """
        pass
