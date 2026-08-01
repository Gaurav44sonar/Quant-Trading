import os
import sys
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

def compute_metrics(trades, capital, portfolio_history):
    total_trades = len(trades)
    if total_trades == 0:
        return {
            "total_trades": 0, "wins": 0, "losses": 0, "win_rate": 0.0,
            "net_pnl": 0.0, "gross_profit": 0.0, "gross_loss": 0.0, "profit_factor": 0.0,
            "sharpe": 0.0, "sortino": 0.0, "max_drawdown": 0.0, "volatility": 0.0,
            "avg_holding_time": 0.0, "avg_trade_duration": 0.0, "exposure_pct": 0.0,
            "stock_breakdown": [], "cost": 0.0, "net_return_pct": 0.0,
            "gross_profit_pct": 0.0, "gross_loss_pct": 0.0
        }

    pnl_pcts = []
    holding_times = []
    trade_durations = []
    
    winning_trades_count = 0
    losing_trades_count = 0
    gross_profit = 0.0
    gross_loss = 0.0
    net_pnl = 0.0
    total_cost = 0.0
    
    stock_breakdown = []
    
    for t in trades:
        ticker = t["ticker"]
        entry_price = t["entry_price"]
        qty = t["initial_qty"]
        exits = t["exits"]
        entry_time = t["entry_time"]
        
        cost = qty * entry_price
        total_cost += cost
        revenue = sum(eqty * eprice for eqty, eprice, etime in exits)
        
        liquidated_qty = sum(eqty for eqty, eprice, etime in exits)
        unliquidated = qty - liquidated_qty
        if unliquidated > 0:
            last_price = exits[-1][1] if exits else entry_price
            revenue += unliquidated * last_price
            
        pnl = revenue - cost
        net_pnl += pnl
        
        pnl_pct = (pnl / cost) if cost > 0 else 0.0
        pnl_pcts.append(pnl_pct)
        
        if pnl >= 0.01:
            winning_trades_count += 1
            gross_profit += pnl
        elif pnl <= -0.01:
            losing_trades_count += 1
            gross_loss += abs(pnl)
        else:
            winning_trades_count += 1
            gross_profit += max(0.0, pnl)
            
        stock_holding_seconds = []
        for eqty, eprice, etime in exits:
            if etime and entry_time:
                diff = (etime - entry_time).total_seconds()
                stock_holding_seconds.append(eqty * diff)
        
        avg_hold_sec = sum(stock_holding_seconds) / qty if qty > 0 and stock_holding_seconds else 0.0
        holding_times.append(avg_hold_sec)
        
        if exits and exits[-1][2] and entry_time:
            duration = (exits[-1][2] - entry_time).total_seconds()
        else:
            duration = 0.0
        trade_durations.append(duration)
        
        stock_breakdown.append({
            "symbol": ticker,
            "trades": 1,
            "wins": 1 if pnl >= 0 else 0,
            "losses": 1 if pnl < 0 else 0,
            "win_rate": 100.0 if pnl >= 0 else 0.0,
            "net_pnl": pnl,
            "gross_profit": max(0.0, pnl),
            "gross_loss": abs(min(0.0, pnl)),
            "duration": duration,
            "cost": cost
        })

    win_rate = (winning_trades_count / total_trades) * 100.0 if total_trades > 0 else 0.0
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
    
    pnl_pcts = np.array(pnl_pcts)
    avg_ret = np.mean(pnl_pcts)
    volatility = np.std(pnl_pcts)
    
    sharpe = avg_ret / volatility * np.sqrt(252) if volatility != 0 else 0.0
    
    losses_pcts = pnl_pcts[pnl_pcts < 0]
    downside_vol = np.std(losses_pcts) if len(losses_pcts) > 0 else 0.0
    sortino = avg_ret / downside_vol * np.sqrt(252) if downside_vol != 0 else 0.0
    
    max_drawdown = 0.0
    if portfolio_history:
        peak = capital
        for t, val in portfolio_history:
            peak = max(peak, val)
            dd = (peak - val) / peak if peak > 0 else 0.0
            max_drawdown = max(max_drawdown, dd)
            
    avg_holding_sec = np.mean(holding_times) if holding_times else 0.0
    avg_trade_dur_sec = np.mean(trade_durations) if trade_durations else 0.0
    
    net_return_pct = (net_pnl / total_cost) * 100.0 if total_cost > 0 else 0.0
    gross_profit_pct = (gross_profit / total_cost) * 100.0 if total_cost > 0 else 0.0
    gross_loss_pct = (gross_loss / total_cost) * 100.0 if total_cost > 0 else 0.0
    
    return {
        "total_trades": total_trades,
        "wins": winning_trades_count,
        "losses": losing_trades_count,
        "win_rate": win_rate,
        "net_pnl": net_pnl,
        "gross_profit": gross_profit,
        "gross_loss": gross_loss,
        "profit_factor": profit_factor,
        "sharpe": sharpe,
        "sortino": sortino,
        "max_drawdown": max_drawdown,
        "volatility": volatility,
        "avg_holding_time": avg_holding_sec,
        "avg_trade_duration": avg_trade_dur_sec,
        "stock_breakdown": stock_breakdown,
        "cost": total_cost,
        "net_return_pct": net_return_pct,
        "gross_profit_pct": gross_profit_pct,
        "gross_loss_pct": gross_loss_pct
    }

def format_duration(seconds):
    if seconds <= 0:
        return "00:00:00"
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    return f"{h:02d}:{m:02d}:{s:02d}"

def compute_side_metrics(trades_list, capital, portfolio_history):
    metrics = compute_metrics(trades_list, capital, portfolio_history)
    
    pnl_pcts = []
    holding_times = []
    time_to_mfes = []
    time_to_maes = []
    wins = []
    losses = []
    
    for t in trades_list:
        entry_price = t["entry_price"]
        qty = t["initial_qty"]
        exits = t["exits"]
        entry_time = t.get("entry_time")
        
        cost = qty * entry_price
        revenue = sum(eqty * eprice for eqty, eprice, *etime in exits)
        liquidated_qty = sum(eqty for eqty, eprice, *etime in exits)
        unliquidated = qty - liquidated_qty
        if unliquidated > 0:
            last_price = exits[-1][1] if exits else entry_price
            revenue += unliquidated * last_price
            
        side = t.get("side", "long").lower()
        if side in ["short", "sell"]:
            pnl = cost - revenue
        else:
            pnl = revenue - cost
            
        pnl_pct = pnl / cost if cost > 0 else 0.0
        pnl_pcts.append(pnl_pct)
        
        if pnl >= 0.01:
            wins.append(pnl_pct)
        elif pnl <= -0.01:
            losses.append(pnl_pct)
        else:
            wins.append(max(0.0, pnl_pct))
            
        # MFE and MAE times
        if exits and entry_time:
            if side in ["short", "sell"]:
                mfe_exit = min(exits, key=lambda x: x[1])
                mae_exit = max(exits, key=lambda x: x[1])
            else:
                mfe_exit = max(exits, key=lambda x: x[1])
                mae_exit = min(exits, key=lambda x: x[1])
            
            mfe_time = mfe_exit[2] if len(mfe_exit) > 2 else None
            mae_time = mae_exit[2] if len(mae_exit) > 2 else None
            
            if mfe_time and entry_time:
                time_to_mfes.append((mfe_time - entry_time).total_seconds() / 60.0)
            if mae_time and entry_time:
                time_to_maes.append((mae_time - entry_time).total_seconds() / 60.0)
                
    pnl_pcts = np.array(pnl_pcts) if pnl_pcts else np.array([0.0])
    
    avg_ret = np.mean(pnl_pcts) if len(trades_list) > 0 else 0.0
    best_trade = np.max(pnl_pcts) if len(trades_list) > 0 else 0.0
    worst_trade = np.min(pnl_pcts) if len(trades_list) > 0 else 0.0
    
    cum_ret = metrics["net_return_pct"] / 100.0 if len(trades_list) > 0 else 0.0
    cagr = (1 + cum_ret) ** 252 - 1 if cum_ret > -1 else -1.0
    
    volatility = np.std(pnl_pcts) if len(trades_list) > 1 else 0.0
    volatility_ann = volatility * np.sqrt(252)
    
    sharpe = avg_ret / volatility * np.sqrt(252) if volatility != 0 else 0.0
    
    losses_pcts = pnl_pcts[pnl_pcts < 0]
    downside_vol = np.std(losses_pcts) if len(losses_pcts) > 0 else 0.0
    sortino = avg_ret / downside_vol * np.sqrt(252) if downside_vol != 0 else 0.0
    
    win_rate = metrics["win_rate"] / 100.0 if len(trades_list) > 0 else 0.0
    
    avg_win = np.mean(wins) if wins else 0.0
    avg_loss = np.mean(losses) if losses else 0.0
    win_loss_ratio = avg_win / abs(avg_loss) if avg_loss != 0 else 0.0
    
    avg_holding_time_sec = metrics["avg_holding_time"]
    
    avg_time_to_mfe = np.mean(time_to_mfes) if time_to_mfes else 0.0
    avg_time_to_mae = np.mean(time_to_maes) if time_to_maes else 0.0
    
    def get_exit_time(x):
        exits = x.get("exits", [])
        if not exits:
            return datetime.min
        last_exit = exits[-1]
        if len(last_exit) < 3 or last_exit[2] is None:
            return datetime.min
        val = last_exit[2]
        if isinstance(val, datetime):
            return val
        if isinstance(val, str):
            try:
                # Handle potential Z suffix or other ISO format variations
                clean_val = val.replace("Z", "+00:00")
                return datetime.fromisoformat(clean_val)
            except ValueError:
                pass
        return datetime.min

    sorted_trades = sorted(trades_list, key=get_exit_time)
    sorted_pnl_pcts = []
    for t in sorted_trades:
        cost = t["initial_qty"] * t["entry_price"]
        revenue = sum(eqty * eprice for eqty, eprice, *etime in t["exits"])
        liquidated_qty = sum(eqty for eqty, eprice, *etime in t["exits"])
        unliquidated = t["initial_qty"] - liquidated_qty
        if unliquidated > 0:
            last_price = t["exits"][-1][1] if t["exits"] else t["entry_price"]
            revenue += unliquidated * last_price
            
        side = t.get("side", "long").lower()
        if side in ["short", "sell"]:
            pnl = cost - revenue
        else:
            pnl = revenue - cost
            
        sorted_pnl_pcts.append(pnl / cost if cost > 0 else 0.0)
        
    if sorted_pnl_pcts:
        cum_returns = np.cumsum(sorted_pnl_pcts)
        peak = np.maximum.accumulate(np.insert(cum_returns, 0, 0.0))
        drawdowns = np.insert(cum_returns, 0, 0.0) - peak
        max_dd = np.min(drawdowns)
    else:
        max_dd = 0.0
        
    return {
        "avg_return": f"{avg_ret*100:.2f}%",
        "cum_return": f"{cum_ret*100:.2f}%",
        "best_trade": f"{best_trade*100:.2f}%",
        "worst_trade": f"{worst_trade*100:.2f}%",
        "total_pnl": metrics["net_pnl"],
        "cagr": f"{cagr*100:.2f}%",
        "sharpe": f"{sharpe:.2f}",
        "sortino": f"{sortino:.2f}",
        "max_drawdown": f"{max_dd*100:.2f}%",
        "volatility": f"{volatility_ann*100:.2f}%",
        "win_rate": f"{win_rate*100:.2f}%",
        "win_loss_ratio": f"{win_loss_ratio:.2f}",
        "avg_win": f"{avg_win*100:.2f}%",
        "avg_loss": f"{avg_loss*100:.2f}%",
        "avg_holding_period": format_duration(avg_holding_time_sec),
        "avg_time_to_mfe": f"{avg_time_to_mfe:.1f}",
        "avg_time_to_mae": f"{avg_time_to_mae:.1f}"
    }

def generate_comparison_table(trades_list, capital, portfolio_history, currency_symbol):
    long_trades = []
    short_trades = []
    for t in trades_list:
        side = t.get("side", "long").lower()
        if side in ["short", "sell"]:
            short_trades.append(t)
        else:
            long_trades.append(t)
            
    short_data = compute_side_metrics(short_trades, capital, [])
    long_data = compute_side_metrics(long_trades, capital, portfolio_history)
    combined_data = compute_side_metrics(trades_list, capital, portfolio_history)
    
    combined_data["cagr"] = "--"
    
    def fmt_pnl(val):
        if val >= 0:
            return f"{currency_symbol}{val:.2f}"
        else:
            return f"{currency_symbol}-{abs(val):.2f}"
            
    lines = []
    lines.append("💰 RETURN METRICS")
    lines.append("-" * 80)
    lines.append(f"{'Avg Return':<30}{short_data['avg_return']:>16}{long_data['avg_return']:>16}{combined_data['avg_return']:>16}")
    lines.append(f"{'Cumulative Return':<30}{short_data['cum_return']:>16}{long_data['cum_return']:>16}{combined_data['cum_return']:>16}")
    lines.append(f"{'Best Trade':<30}{short_data['best_trade']:>16}{long_data['best_trade']:>16}{combined_data['best_trade']:>16}")
    lines.append(f"{'Worst Trade':<30}{short_data['worst_trade']:>16}{long_data['worst_trade']:>16}{combined_data['worst_trade']:>16}")
    lines.append(f"{'Total P&L':<30}{fmt_pnl(short_data['total_pnl']):>16}{fmt_pnl(long_data['total_pnl']):>16}{fmt_pnl(combined_data['total_pnl']):>16}")
    lines.append(f"{'CAGR (annualized)':<30}{short_data['cagr']:>16}{long_data['cagr']:>16}{combined_data['cagr']:>16}")
    lines.append("")
    
    lines.append("⚠ RISK METRICS")
    lines.append("-" * 80)
    lines.append(f"{'Sharpe Ratio':<30}{short_data['sharpe']:>16}{long_data['sharpe']:>16}{combined_data['sharpe']:>16}")
    lines.append(f"{'Sortino Ratio':<30}{short_data['sortino']:>16}{long_data['sortino']:>16}{combined_data['sortino']:>16}")
    lines.append(f"{'Max Drawdown':<30}{short_data['max_drawdown']:>16}{long_data['max_drawdown']:>16}{combined_data['max_drawdown']:>16}")
    lines.append(f"{'Volatility (Ann.)':<30}{short_data['volatility']:>16}{long_data['volatility']:>16}{combined_data['volatility']:>16}")
    lines.append(f"{'Win Rate':<30}{short_data['win_rate']:>16}{long_data['win_rate']:>16}{combined_data['win_rate']:>16}")
    lines.append(f"{'Win/Loss Ratio':<30}{short_data['win_loss_ratio']:>16}{long_data['win_loss_ratio']:>16}{combined_data['win_loss_ratio']:>16}")
    lines.append(f"{'Avg Win':<30}{short_data['avg_win']:>16}{long_data['avg_win']:>16}{combined_data['avg_win']:>16}")
    lines.append(f"{'Avg Loss':<30}{short_data['avg_loss']:>16}{long_data['avg_loss']:>16}{combined_data['avg_loss']:>16}")
    lines.append("")
    
    lines.append("⏱ TIMING METRICS")
    lines.append("-" * 80)
    lines.append(f"{'Average holding Period':<30}{short_data['avg_holding_period']:>16}{long_data['avg_holding_period']:>16}{combined_data['avg_holding_period']:>16}")
    lines.append(f"{'Avg Time to MFE (min)':<30}{short_data['avg_time_to_mfe']:>16}{long_data['avg_time_to_mfe']:>16}{combined_data['avg_time_to_mfe']:>16}")
    lines.append(f"{'Avg Time to MAE (min)':<30}{short_data['avg_time_to_mae']:>16}{long_data['avg_time_to_mae']:>16}{combined_data['avg_time_to_mae']:>16}")
    
    return "\n".join(lines)

def generate_individual_report_text(
    market,
    universe_type,
    universe_size,
    volatile_list,
    nonvolatile_list,
    ranking_method,
    selection_timestamp,
    start_time,
    end_time,
    duration_seconds,
    termination_reason,
    duration_completed,
    manual_stop,
    runtime_errors,
    api_issues,
    trades,
    capital,
    portfolio_history,
    exposure_pct,
    save_path,
    sentiment_summary=None
):
    """
    Generate the text report for a single universe run and return the text.
    """
    overall = compute_metrics(trades, capital, portfolio_history)
    
    v_trades = [t for t in trades if t["group"] == "volatile"]
    volatile_perf = compute_metrics(v_trades, capital, [])
    
    nv_trades = [t for t in trades if t["group"] == "nonvolatile"]
    nonvolatile_perf = compute_metrics(nv_trades, capital, [])
    
    total_pnl = overall["net_pnl"]
    for sb in overall["stock_breakdown"]:
        pnl = sb["net_pnl"]
        sb["contribution_pct_capital"] = (pnl / capital) * 100.0
        sb["contribution_pct_overall"] = (pnl / total_pnl * 100.0) if abs(total_pnl) >= 0.01 else 0.0

    market = market.lower()
    currency_symbol = "₹" if (market == "india" or market == "ns") else "$"

    lines = []
    lines.append("=" * 80)
    lines.append(f"INTRADAY MEAN-REVERSION LIVE SESSION REPORT: {market.upper()} - {universe_type.upper()}")
    lines.append("=" * 80)
    lines.append("")
    
    # SECTION 1: OVERALL PERFORMANCE SUMMARY
    lines.append("==================================================")
    lines.append("SECTION 1: OVERALL PERFORMANCE SUMMARY")
    lines.append("==================================================")
    lines.append(f"Market                  : {market.upper()}")
    lines.append(f"Universe Type           : {universe_type.upper()}")
    lines.append(f"Total Universe Size     : {universe_size} Stocks")
    lines.append(f"Start Time              : {start_time.strftime('%Y-%m-%d %H:%M:%S') if start_time else 'N/A'}")
    lines.append(f"End Time                : {end_time.strftime('%Y-%m-%d %H:%M:%S') if end_time else 'N/A'}")
    lines.append(f"Duration                : {format_duration(duration_seconds)}")
    lines.append(f"Execution Status        : {termination_reason}")
    lines.append(f"Exposure %              : {exposure_pct:.2f}%")
    lines.append("")
    lines.append(generate_comparison_table(trades, capital, portfolio_history, currency_symbol))
    lines.append("")
    
    # SECTION 2: VOLATILE STOCKS PERFORMANCE
    lines.append("==================================================")
    lines.append("SECTION 2: VOLATILE STOCKS PERFORMANCE")
    lines.append("==================================================")
    lines.append(f"Selected Stocks         : {', '.join(sorted(list(set(t['ticker'] for t in v_trades)))) or 'None'}")
    lines.append("")
    lines.append(generate_comparison_table(v_trades, capital, [], currency_symbol))
    lines.append("")
    
    # SECTION 3: NON-VOLATILE STOCKS PERFORMANCE
    lines.append("==================================================")
    lines.append("SECTION 3: NON-VOLATILE STOCKS PERFORMANCE")
    lines.append("==================================================")
    lines.append(f"Selected Stocks         : {', '.join(sorted(list(set(t['ticker'] for t in nv_trades)))) or 'None'}")
    lines.append("")
    lines.append(generate_comparison_table(nv_trades, capital, [], currency_symbol))
    lines.append("")
    
    # SECTION 4: STOCK-WISE PERFORMANCE BREAKDOWN
    lines.append("==================================================")
    lines.append("SECTION 4: STOCK-WISE PERFORMANCE BREAKDOWN")
    lines.append("==================================================")
    headers = ["Symbol", "Trades", "Wins", "Losses", "Win Rate", "Net PnL", "Gross Profit", "Gross Loss", "Duration", "Contrib% Cap", "Contrib% PnL"]
    col_widths = [14, 8, 6, 8, 10, 12, 14, 14, 12, 12, 12]
    
    row_fmt = " | ".join(f"{{:<{w}}}" for w in col_widths)
    lines.append(row_fmt.format(*headers))
    lines.append("-" * (sum(col_widths) + len(col_widths)*3 - 3))
    
    for sb in overall["stock_breakdown"]:
        row_values = [
            sb["symbol"],
            str(sb["trades"]),
            str(sb["wins"]),
            str(sb["losses"]),
            f"{sb['win_rate']:.1f}%",
            f"{currency_symbol}{sb['net_pnl']:+,.2f}",
            f"{currency_symbol}{sb['gross_profit']:.2f}",
            f"{currency_symbol}{sb['gross_loss']:.2f}",
            format_duration(sb["duration"]),
            f"{sb['contribution_pct_capital']:+.2f}%",
            f"{sb['contribution_pct_overall']:+.1f}%"
        ]
        lines.append(row_fmt.format(*row_values))
    lines.append("")
    
    # SECTION 5: UNIVERSE INFORMATION
    lines.append("==================================================")
    lines.append("SECTION 5: UNIVERSE INFORMATION")
    lines.append("==================================================")
    lines.append(f"Universe Type           : {universe_type.upper()}")
    lines.append(f"Volatile Stocks List    : {', '.join(sorted(volatile_list))}")
    lines.append(f"Non-Volatile Stocks List: {', '.join(sorted(nonvolatile_list))}")
    lines.append(f"Ranking Method Used     : {ranking_method}")
    lines.append(f"Selection Timestamp     : {selection_timestamp.strftime('%Y-%m-%d %H:%M:%S') if selection_timestamp else 'N/A'}")
    lines.append("")
    
    # SECTION 6: SYSTEM INFORMATION
    lines.append("==================================================")
    lines.append("SECTION 6: SYSTEM INFORMATION")
    lines.append("==================================================")
    lines.append(f"Market                  : {market.upper()}")
    lines.append(f"Universe Type           : {universe_type.upper()}")
    lines.append(f"Termination Reason      : {termination_reason}")
    lines.append(f"Duration Completed Flag : {duration_completed}")
    lines.append(f"Manual Stop Flag        : {manual_stop}")
    lines.append(f"Runtime Errors          : {'; '.join(runtime_errors) if runtime_errors else 'None'}")
    lines.append(f"API/Data Issues         : {'; '.join(api_issues) if api_issues else 'None'}")
    lines.append(f"Report Generation Status: {'SUCCESS' if not runtime_errors else 'PARTIAL_DATA'}")
    lines.append(f"Report Save Location    : {os.path.abspath(save_path)}")
    lines.append("")
    
    # SECTION 7 & 8: News Sentiment Analysis (if available)
    if sentiment_summary is not None:
        stats = sentiment_summary.get("stats", {})
        details = sentiment_summary.get("stock_details", [])
        sources = sentiment_summary.get("sources_used", [])
        
        # SECTION 7: NEWS SENTIMENT ANALYSIS
        lines.append("==================================================")
        lines.append("SECTION 7: NEWS SENTIMENT ANALYSIS")
        lines.append("==================================================")
        lines.append(f"Overall Market Sentiment: {stats.get('market_sentiment', 0.0):+.2f}")
        lines.append(f"Average Sentiment Score : {stats.get('avg_sentiment', 0.0):+.2f}")
        lines.append(f"News Sources Used       : {', '.join(sources)}")
        lines.append(f"Articles Processed      : {stats.get('total_articles', 0)}")
        lines.append(f"Duplicate Articles Duped: {stats.get('dup_removed', 0)}")
        lines.append(f"Confirmed Trades        : {stats.get('confirmed', 0)}")
        lines.append(f"Rejected Trades (Negative): {stats.get('rejected', 0)}")
        lines.append(f"Held Trades (Low Conf)  : {stats.get('held', 0)}")
        lines.append("")
        
        # Sector sentiments
        lines.append("Sector Sentiments:")
        for sector, pol in stats.get("sector_sentiment", {}).items():
            lines.append(f"  - {sector:<15}: {pol:+.2f}")
        lines.append("")
        
        # SECTION 8: STOCK-WISE NEWS ANALYSIS
        lines.append("==================================================")
        lines.append("SECTION 8: STOCK-WISE NEWS ANALYSIS")
        lines.append("==================================================")
        
        headers_sent = ["Symbol", "Sector", "Signal(z)", "Ticker Sent", "Market Sent", "Final Conf", "Decision", "Reason"]
        widths_sent = [10, 15, 10, 12, 12, 11, 10, 35]
        row_fmt_sent = " | ".join(f"{{:<{w}}}" for w in widths_sent)
        
        lines.append(row_fmt_sent.format(*headers_sent))
        lines.append("-" * (sum(widths_sent) + len(widths_sent)*3 - 3))
        
        for d in details:
            lines.append(row_fmt_sent.format(
                d["ticker"],
                d["sector"],
                f"{d['z_score']:+.2f}",
                f"{d['company_sentiment']:+.2f}",
                f"{d['market_sentiment']:+.2f}",
                f"{d['confidence']:.2f}",
                d["decision"],
                d["reason"][:35]
            ))
        lines.append("")
        
    report_content = "\n".join(lines)
    return overall, report_content

def generate_master_summary_report(
    market,
    run1_type,
    run1_metrics,
    run1_text,
    run2_type,
    run2_metrics,
    run2_text,
    save_path,
    run_timestamp
):
    """
    Generate the Master Summary Report and write it along with the individual reports as sections.
    """
    lines = []
    lines.append("=" * 80)
    lines.append(f"CONSOLIDATED MASTER SUMMARY REPORT: {market.upper()} MARKET")
    lines.append("=" * 80)
    lines.append(f"Report Generated At: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"Shared Run Suffix: {run_timestamp}")
    lines.append("")
    
    def fmt_val(v, mode="usd"):
        if v is None:
            return "N/A"
        if mode == "usd":
            return f"${v:+,.2f}"
        elif mode == "pct":
            return f"{v:.2f}%"
        elif mode == "ratio":
            return f"{v:.2f}"
        elif mode == "int":
            return f"{v}"
        elif mode == "dur":
            return format_duration(v)
        return str(v)

    # 1. Performance & Risk Side-by-Side Comparison
    lines.append("================================================================================")
    lines.append("1. SIDE-BY-SIDE PERFORMANCE COMPARISON")
    lines.append("================================================================================")
    
    headers = ["Metric", run1_type.upper(), run2_type.upper()]
    col_widths = [30, 24, 24]
    row_fmt = " | ".join(f"{{:<{w}}}" for w in col_widths)
    
    lines.append(row_fmt.format(*headers))
    lines.append("-" * (sum(col_widths) + len(col_widths)*3 - 3))
    
    comparison_items = [
        ("Total Trades", "total_trades", "int"),
        ("Winning Trades", "wins", "int"),
        ("Losing Trades", "losses", "int"),
        ("Win Rate", "win_rate", "pct"),
        ("Profit Factor", "profit_factor", "ratio"),
        ("Net PnL", "net_pnl", "usd"),
        ("Net Return (%)", "net_return_pct", "pct"),
        ("Gross Profit", "gross_profit", "usd"),
        ("Gross Profit (%)", "gross_profit_pct", "pct"),
        ("Gross Loss", "gross_loss", "usd"),
        ("Gross Loss (%)", "gross_loss_pct", "pct"),
        ("Sharpe Ratio", "sharpe", "ratio"),
        ("Sortino Ratio", "sortino", "ratio"),
        ("Maximum Drawdown", "max_drawdown", "pct"),
        ("Volatility", "volatility", "pct"),
        ("Average Holding Time", "avg_holding_time", "dur"),
        ("Average Trade Duration", "avg_trade_duration", "dur"),
    ]
    
    for label, key, mode in comparison_items:
        v1 = run1_metrics.get(key) if run1_metrics else None
        v2 = run2_metrics.get(key) if run2_metrics else None
        if mode == "pct" and key in ["max_drawdown", "volatility"]:
            if v1 is not None: v1 *= 100.0
            if v2 is not None: v2 *= 100.0
        lines.append(row_fmt.format(label, fmt_val(v1, mode), fmt_val(v2, mode)))
    lines.append("")
    
    # 2. BEST/WORST PERFORMING UNIVERSE
    lines.append("================================================================================")
    lines.append("2. HIGHLIGHTS & BEST/WORST PERFORMERS")
    lines.append("================================================================================")
    
    pnl1 = run1_metrics.get("net_pnl", 0.0) if run1_metrics else 0.0
    pnl2 = run2_metrics.get("net_pnl", 0.0) if run2_metrics else 0.0
    best_pnl_uni = run1_type if pnl1 > pnl2 else run2_type
    worst_pnl_uni = run2_type if pnl1 > pnl2 else run1_type
    
    wr1 = run1_metrics.get("win_rate", 0.0) if run1_metrics else 0.0
    wr2 = run2_metrics.get("win_rate", 0.0) if run2_metrics else 0.0
    best_wr_uni = run1_type if wr1 > wr2 else run2_type
    worst_wr_uni = run2_type if wr1 > wr2 else run1_type
    
    sr1 = run1_metrics.get("sharpe", 0.0) if run1_metrics else 0.0
    sr2 = run2_metrics.get("sharpe", 0.0) if run2_metrics else 0.0
    best_sr_uni = run1_type if sr1 > sr2 else run2_type
    worst_sr_uni = run2_type if sr1 > sr2 else run1_type
    
    lines.append(f"Highest Net PnL        : {best_pnl_uni.upper()} ({fmt_val(max(pnl1, pnl2), 'usd')})")
    lines.append(f"Highest Win Rate       : {best_wr_uni.upper()} ({fmt_val(max(wr1, wr2), 'pct')})")
    lines.append(f"Highest Sharpe Ratio   : {best_sr_uni.upper()} ({fmt_val(max(sr1, sr2), 'ratio')})")
    lines.append("")
    lines.append(f"Lowest Net PnL         : {worst_pnl_uni.upper()} ({fmt_val(min(pnl1, pnl2), 'usd')})")
    lines.append(f"Lowest Win Rate        : {worst_wr_uni.upper()} ({fmt_val(min(wr1, wr2), 'pct')})")
    lines.append(f"Lowest Sharpe Ratio    : {worst_sr_uni.upper()} ({fmt_val(min(sr1, sr2), 'ratio')})")
    lines.append("")
    
    # 3. OVERALL UNIVERSE RANKING
    lines.append("================================================================================")
    lines.append("3. OVERALL UNIVERSE RANKING")
    lines.append("================================================================================")
    
    runs = []
    if run1_metrics:
        runs.append({"type": run1_type, "metrics": run1_metrics})
    if run2_metrics:
        runs.append({"type": run2_type, "metrics": run2_metrics})
        
    def rank_key(item):
        m = item["metrics"]
        return (m.get("net_pnl", 0.0), m.get("sharpe", 0.0), -m.get("max_drawdown", 0.0))
        
    sorted_runs = sorted(runs, key=rank_key, reverse=True)
    
    for idx, run_item in enumerate(sorted_runs):
        utype = run_item["type"].upper()
        metrics = run_item["metrics"]
        lines.append(
            f"Rank {idx+1} : {utype:<10} | "
            f"Net PnL: ${metrics.get('net_pnl', 0.0):+,.2f} | "
            f"Sharpe: {metrics.get('sharpe', 0.0):.2f} | "
            f"Max DD: {metrics.get('max_drawdown', 0.0)*100:.2f}%"
        )
    lines.append("")
    lines.append("\n" + "#" * 80 + "\n")
    
    # Append individual reports text
    if run1_text:
        lines.append(run1_text)
        lines.append("\n" + "#" * 80 + "\n")
    if run2_text:
        lines.append(run2_text)
        lines.append("\n" + "#" * 80 + "\n")
        
    master_content = "\n".join(lines)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, "w", encoding="utf-8") as f:
        f.write(master_content)
        
    return master_content
