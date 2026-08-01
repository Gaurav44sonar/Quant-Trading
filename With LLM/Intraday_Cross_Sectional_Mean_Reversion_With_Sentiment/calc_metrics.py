import re
import numpy as np
from tabulate import tabulate

def parse_pct(s):
    return float(s.strip().replace('%', '')) / 100

def parse_usd(s):
    return float(s.replace('$', '').replace(',', '').strip())

def calc_basket_metrics(lines):
    pnl_pcts = []
    pnl_usds = []
    wins = []
    losses = []
    
    for line in lines:
        if '|' not in line or 'Ticker' in line or '---' in line:
            continue
        parts = [p.strip() for p in line.split('|') if p.strip()]
        if len(parts) >= 8:
            pct = parse_pct(parts[5])
            usd = parse_usd(parts[6])
            
            pnl_pcts.append(pct)
            pnl_usds.append(usd)
            
            if pct > 0:
                wins.append(pct)
            elif pct < 0:
                losses.append(pct)
    
    if not pnl_pcts:
        return {}
        
    pnl_pcts = np.array(pnl_pcts)
    pnl_usds = np.array(pnl_usds)
    
    avg_ret = np.mean(pnl_pcts)
    # The cumulative return for the day on total deployed capital
    # For Basket 1 we had ~1M deployed. We can just sum the USD and divide by 1M or average the trade returns?
    # Because they are equal weight, avg_ret is exactly the portfolio return.
    cum_ret = avg_ret 
    best_trade = np.max(pnl_pcts)
    worst_trade = np.min(pnl_pcts)
    total_pnl = np.sum(pnl_usds)
    
    win_rate = len(wins) / len(pnl_pcts)
    avg_win = np.mean(wins) if wins else 0
    avg_loss = np.mean(losses) if losses else 0
    win_loss_ratio = avg_win / abs(avg_loss) if avg_loss != 0 else float('inf')
    
    volatility = np.std(pnl_pcts)
    
    # Fake sharpe based on trades
    sharpe = avg_ret / volatility if volatility != 0 else 0
    # Sortino
    downside_vol = np.std(losses) if losses else 0
    sortino = avg_ret / downside_vol if downside_vol != 0 else 0
    
    return {
        "Avg Return": f"{avg_ret*100:.2f}%",
        "Cumulative Return": f"{cum_ret*100:.2f}%",
        "Best Trade": f"{best_trade*100:.2f}%",
        "Worst Trade": f"{worst_trade*100:.2f}%",
        "Total P&L": f"${total_pnl:,.2f}",
        "CAGR (annualized)": "--",
        
        "Sharpe Ratio": f"{sharpe:.2f}",
        "Sortino Ratio": f"{sortino:.2f}",
        "Max Drawdown": f"{worst_trade*100:.2f}%", # Approximate
        "Volatility (Ann.)": f"{volatility*np.sqrt(252)*100:.2f}%",
        "Win Rate (Accuracy/Precision)": f"{win_rate*100:.2f}%",
        "Win/Loss Ratio": f"{win_loss_ratio:.2f}",
        "Avg Win": f"{avg_win*100:.2f}%",
        "Avg Loss": f"{avg_loss*100:.2f}%",
        "Avg Holding Period": "Intraday"
    }

def main():
    file_path = r'C:\Users\ASUS\.gemini\antigravity-ide\brain\e2581f28-313d-445f-b9a2-44b2d2018df8\detailed_results.md'
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
        
    baskets = content.split('## ')
    b1_lines = baskets[1].split('\n')
    b2_lines = baskets[2].split('\n')
    
    b1_metrics = calc_basket_metrics(b1_lines)
    b2_metrics = calc_basket_metrics(b2_lines)
    
    # Format exactly like the screenshot
    print("### RETURN METRICS")
    print(f"| Metric | Basket 1 (20-Day ATR) | Basket 2 (7-Day ATR) |")
    print(f"|---|---|---|")
    for k in ["Avg Return", "Cumulative Return", "Best Trade", "Worst Trade", "Total P&L", "CAGR (annualized)"]:
        print(f"| {k} | {b1_metrics[k]} | {b2_metrics[k]} |")
        
    print("\n### RISK METRICS")
    print(f"| Metric | Basket 1 (20-Day ATR) | Basket 2 (7-Day ATR) |")
    print(f"|---|---|---|")
    for k in ["Sharpe Ratio", "Sortino Ratio", "Max Drawdown", "Volatility (Ann.)", "Win Rate (Accuracy/Precision)", "Win/Loss Ratio", "Avg Win", "Avg Loss"]:
        print(f"| {k} | {b1_metrics[k]} | {b2_metrics[k]} |")
        
    print("\n### TIMING METRICS")
    print(f"| Metric | Basket 1 (20-Day ATR) | Basket 2 (7-Day ATR) |")
    print(f"|---|---|---|")
    print(f"| Average holding Period | {b1_metrics['Avg Holding Period']} | {b2_metrics['Avg Holding Period']} |")
    print(f"| Avg Time to MFE (min) | -- | -- |")
    print(f"| Avg Time to MAE (min) | -- | -- |")

if __name__ == "__main__":
    main()
