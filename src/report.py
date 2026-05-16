import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import os

CSV_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "backtest_results.csv")

def generate_interactive_report():
    if not os.path.exists(CSV_PATH):
        print("[ERROR] No backtest results found. Run the backtester first.")
        return
    
    df = pd.read_csv(CSV_PATH)
    df = df[df['Entry Time'] != 'SUMMARY'].dropna(subset=['PnL'])
    df['PnL'] = pd.to_numeric(df['PnL'], errors='coerce')
    df['Balance'] = pd.to_numeric(df['Balance'], errors='coerce')
    df['RR'] = pd.to_numeric(df['RR'], errors='coerce')
    df['Lots'] = pd.to_numeric(df['Lots'], errors='coerce')
    df['Entry Time'] = pd.to_datetime(df['Entry Time'], errors='coerce')
    df['Exit Time'] = pd.to_datetime(df['Exit Time'], errors='coerce')
    df.dropna(subset=['PnL', 'Balance'], inplace=True)
    
    if len(df) == 0:
        print("[ERROR] No valid trades found.")
        return

    # Stats
    total = len(df)
    wins = len(df[df['PnL'] > 0])
    losses = len(df[df['PnL'] < 0])
    win_rate = (wins / total * 100) if total > 0 else 0
    avg_win = df[df['PnL'] > 0]['PnL'].mean() if wins > 0 else 0
    avg_loss = df[df['PnL'] < 0]['PnL'].mean() if losses > 0 else 0
    expectancy = (win_rate/100 * avg_win) + ((1 - win_rate/100) * avg_loss)
    initial_balance = df['Balance'].iloc[0] - df['PnL'].iloc[0]
    final_balance = df['Balance'].iloc[-1]
    total_profit = final_balance - initial_balance
    
    peak = df['Balance'].cummax()
    drawdown = (peak - df['Balance']) / peak * 100
    max_dd = drawdown.max()

    # Cumulative wins/losses for streak tracking
    df['Outcome'] = df['PnL'].apply(lambda x: 'WIN' if x > 0 else 'LOSS')
    df['Color'] = df['PnL'].apply(lambda x: '#3fb950' if x > 0 else '#f85149')
    
    # Build interactive dashboard
    fig = make_subplots(
        rows=4, cols=2,
        row_heights=[0.35, 0.25, 0.20, 0.20],
        column_widths=[0.6, 0.4],
        subplot_titles=[
            'EQUITY CURVE', 'PERFORMANCE SUMMARY',
            'DRAWDOWN', 'WIN/LOSS DISTRIBUTION',
            'PROFIT/LOSS PER TRADE', 'R:R DISTRIBUTION'
        ],
        specs=[
            [{"colspan": 1}, {"type": "table"}],
            [{"colspan": 1}, {"type": "pie"}],
            [{"colspan": 1}, {"colspan": 1}],
            [{"colspan": 2, "type": "bar"}, None]
        ],
        vertical_spacing=0.08,
        horizontal_spacing=0.08
    )

    # 1. EQUITY CURVE
    fig.add_trace(go.Scatter(
        x=df['Exit Time'], y=df['Balance'],
        mode='lines', name='Balance',
        line=dict(color='#58a6ff', width=2),
        fill='tozeroy', fillcolor='rgba(88,166,255,0.1)',
        hovertemplate='<b>Trade #%{pointNumber}</b><br>Balance: $%{y:.2f}<br>Date: %{x}<extra></extra>'
    ), row=1, col=1)
    
    fig.add_hline(y=initial_balance, line_dash="dash", line_color="#484f58", 
                  annotation_text=f"Start: ${initial_balance:.2f}", row=1, col=1)

    # 2. STATS TABLE
    fig.add_trace(go.Table(
        header=dict(
            values=['<b>METRIC</b>', '<b>VALUE</b>'],
            fill_color='#161b22',
            font=dict(color='#58a6ff', size=12),
            align='left', line_color='#30363d'
        ),
        cells=dict(
            values=[
                ['Total Trades', 'Wins', 'Losses', 'Win Rate', 'Avg Win', 'Avg Loss',
                 'Expectancy', 'Max Drawdown', 'Initial Balance', 'Final Balance', 'Total Profit'],
                [f'{total}', f'{wins}', f'{losses}', f'{win_rate:.1f}%',
                 f'${avg_win:.2f}', f'${avg_loss:.2f}', f'${expectancy:.2f}',
                 f'{max_dd:.1f}%', f'${initial_balance:.2f}', f'${final_balance:.2f}',
                 f'${total_profit:.2f}']
            ],
            fill_color=[['#0d1117'] * 11, ['#0d1117'] * 11],
            font=dict(color=[['#c9d1d9'] * 11, 
                            ['#c9d1d9', '#3fb950', '#f85149', '#c9d1d9', '#3fb950', '#f85149',
                             '#3fb950' if expectancy > 0 else '#f85149',
                             '#f85149', '#c9d1d9', '#c9d1d9',
                             '#3fb950' if total_profit > 0 else '#f85149']], size=11),
            align='left', line_color='#30363d', height=25
        )
    ), row=1, col=2)

    # 3. DRAWDOWN
    fig.add_trace(go.Scatter(
        x=df['Exit Time'], y=drawdown,
        mode='lines', name='Drawdown',
        line=dict(color='#f85149', width=1),
        fill='tozeroy', fillcolor='rgba(248,81,73,0.2)',
        hovertemplate='Drawdown: %{y:.1f}%<extra></extra>'
    ), row=2, col=1)

    # 4. PIE CHART
    fig.add_trace(go.Pie(
        labels=[f'Wins ({wins})', f'Losses ({losses})'],
        values=[wins, losses],
        marker=dict(colors=['#3fb950', '#f85149'], line=dict(color='#0d1117', width=2)),
        textinfo='percent+label',
        textfont=dict(color='white', size=12),
        hovertemplate='%{label}: %{value} trades<extra></extra>'
    ), row=2, col=2)

    # 5. PnL PER TRADE (bottom full width)
    fig.add_trace(go.Bar(
        x=df['Exit Time'], y=df['PnL'],
        marker_color=df['Color'].tolist(),
        name='P&L',
        hovertemplate='<b>Trade #%{pointNumber}</b><br>P&L: $%{y:.2f}<br>%{x}<extra></extra>'
    ), row=4, col=1)

    # 6. RR Distribution  
    rr_data = df['RR'].dropna()
    if len(rr_data) > 0:
        fig.add_trace(go.Histogram(
            x=rr_data, nbinsx=25,
            marker_color='#58a6ff', opacity=0.7,
            name='R:R',
            hovertemplate='R:R Range: %{x}<br>Count: %{y}<extra></extra>'
        ), row=3, col=2)

    # Cumulative PnL on same row as RR
    fig.add_trace(go.Scatter(
        x=df['Exit Time'], y=df['PnL'].cumsum(),
        mode='lines', name='Cumulative P&L',
        line=dict(color='#d2a8ff', width=2),
        hovertemplate='Cumulative P&L: $%{y:.2f}<extra></extra>'
    ), row=3, col=1)

    # Layout
    fig.update_layout(
        title=dict(
            text=f'NEXUS TRADING BOT — 6 MONTH BACKTEST | {total} Trades | {win_rate:.1f}% Win Rate | ${total_profit:.2f} Profit',
            font=dict(size=16, color='#58a6ff'),
            x=0.5
        ),
        template='plotly_dark',
        paper_bgcolor='#0d1117',
        plot_bgcolor='#161b22',
        font=dict(color='#c9d1d9'),
        showlegend=False,
        height=1200,
        width=1400
    )
    
    fig.update_yaxes(title_text="Balance ($)", row=1, col=1)
    fig.update_yaxes(title_text="Drawdown %", autorange="reversed", row=2, col=1)
    fig.update_yaxes(title_text="Cumulative P&L ($)", row=3, col=1)
    fig.update_yaxes(title_text="P&L ($)", row=4, col=1)

    output_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "backtest_report.html")
    fig.write_html(output_path)
    print(f"\n[REPORT] Interactive report saved to -> {output_path}")
    print(f"[STATS] {total} trades | Win Rate: {win_rate:.1f}% | Profit: ${total_profit:.2f} | Max DD: {max_dd:.1f}%")
    os.startfile(output_path)

if __name__ == "__main__":
    generate_interactive_report()
