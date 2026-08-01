import re
import os
import sys
import numpy as np

if sys.platform.startswith("win"):
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")


def format_row(row, widths):
    return " | ".join(f"{str(val):<{widths[i]}}" for i, val in enumerate(row))

def print_section_summary(title, count_label, entries_count, cost, revenue, wins, losses, trades=None):
    net_pnl = revenue - cost
    pct_return = (net_pnl / cost) * 100 if cost > 0 else 0.0
    win_rate = (wins / entries_count) * 100 if entries_count else 0.0
    
    print(f"\n{title} SUMMARY")
    print(f"  {count_label:<24} : {entries_count}")
    print(f"  {'Capital Deployed' if 'OVERALL' not in title else 'Total Deployed Capital':<24} : ${cost:,.2f}")
    print(f"  {'Capital Returned' if 'OVERALL' not in title else 'Total Returned Capital':<24} : ${revenue:,.2f}")
    print(f"  {'Net Session P&L ($)' :<24} : {net_pnl:+.2f}")
    print(f"  {'Session Return (%)' :<24} : {pct_return:+.2f}%")
    print(f"  {'Win Rate (Tickers)' :<24} : {win_rate:.1f}% ({wins} Win / {losses} Loss)")
    
    if trades:
        pnl_pcts = np.array([t['pnl_pct'] for t in trades]) / 100.0
        pnl_usds = np.array([t['pnl_usd'] for t in trades])
        
        avg_ret = np.mean(pnl_pcts)
        best_trade = np.max(pnl_pcts)
        worst_trade = np.min(pnl_pcts)
        
        cum_ret = net_pnl / cost if cost > 0 else 0.0
        cagr = (1 + cum_ret) ** 252 - 1
        
        volatility = np.std(pnl_pcts)
        volatility_ann = volatility * np.sqrt(252)
        
        sharpe = avg_ret / volatility * np.sqrt(252) if volatility != 0 else 0.0
        
        losses_pcts = pnl_pcts[pnl_pcts < 0]
        downside_vol = np.std(losses_pcts) if len(losses_pcts) > 0 else 0.0
        sortino = avg_ret / downside_vol * np.sqrt(252) if downside_vol != 0 else 0.0
        
        wins_pct = pnl_pcts[pnl_usds >= -0.0099]
        losses_pct = pnl_pcts[pnl_usds <= -0.01]
        
        avg_win = np.mean(wins_pct) if len(wins_pct) > 0 else 0.0
        avg_loss = np.mean(losses_pct) if len(losses_pct) > 0 else 0.0
        win_loss_ratio = avg_win / abs(avg_loss) if avg_loss != 0 else float('inf')
        
        def fmt_pct(val):
            sign = "+" if val >= 0 else ""
            return f"{sign}{val*100:.2f}%"
            
        print(f"  {'Avg Trade Return' :<24} : {fmt_pct(avg_ret)}")
        print(f"  {'Best Trade' :<24} : {fmt_pct(best_trade)}")
        print(f"  {'Worst Trade' :<24} : {fmt_pct(worst_trade)}")
        print(f"  {'CAGR (Annualized)' :<24} : {fmt_pct(cagr)}")
        print(f"  {'Sharpe Ratio' :<24} : {sharpe:.2f}")
        print(f"  {'Sortino Ratio' :<24} : {sortino:.2f}")
        print(f"  {'Volatility (Ann.)' :<24} : {fmt_pct(volatility_ann)}")
        print(f"  {'Win/Loss Ratio' :<24} : {win_loss_ratio:.2f}")
        print(f"  {'Avg Win' :<24} : {fmt_pct(avg_win)}")
        print(f"  {'Avg Loss' :<24} : {fmt_pct(avg_loss)}")

import contextlib

class DualWriter:
    def __init__(self, file_path):
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        self.file = open(file_path, "w", encoding="utf-8")
        self.stdout = sys.stdout

    def write(self, data):
        self.file.write(data)
        self.stdout.write(data)

    def flush(self):
        self.file.flush()
        self.stdout.flush()

    def close(self):
        self.file.close()

def parse_log(log_path="live_logs/pipeline.log", output_file=None):
    if output_file:
        writer = DualWriter(output_file)
        with contextlib.redirect_stdout(writer):
            try:
                _parse_log_inner(log_path)
            finally:
                writer.close()
    else:
        _parse_log_inner(log_path)

def _parse_log_inner(log_path="live_logs/pipeline.log"):
    if not os.path.exists(log_path):
        print(f"Error: Log file {log_path} not found.")
        return

    with open(log_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    # Find all startup indices
    startup_indicators = [
        "[PAPER TRADING MODE]",
        "!!! RUNNING IN LIVE TRADING MODE (REAL MONEY) !!!"
    ]
    startup_indices = []
    for idx, line in enumerate(lines):
        if any(indicator in line for indicator in startup_indicators):
            startup_indices.append(idx)

    # Walk backwards to find the original session start (ignoring crash recovery restarts)
    session_start_idx = 0
    if startup_indices:
        session_start_idx = startup_indices[-1]
        for i in range(len(startup_indices) - 1, -1, -1):
            idx = startup_indices[i]
            next_idx = startup_indices[i+1] if i + 1 < len(startup_indices) else len(lines)
            
            # Check if this segment loaded a saved state
            segment_lines = lines[idx:next_idx]
            is_recovery = any("Loaded execution state" in line for line in segment_lines)
            if not is_recovery:
                session_start_idx = idx
                break

    session_lines = lines[session_start_idx:]
    
    # Parse volatile / non-volatile ticker classification
    volatile_tickers = set()
    nonvolatile_tickers = set()
    has_classification = False
    
    for line in session_lines:
        if "[UNIVERSE] Volatile:" in line:
            has_classification = True
            parts = line.split("[UNIVERSE] Volatile:", 1)
            content = parts[1].strip()
            if "(" in content:
                content = content.split("(", 1)[0].strip()
            for t in content.split(","):
                ticker = t.strip()
                if ticker:
                    volatile_tickers.add(ticker)
        elif "[UNIVERSE] Non-Volatile:" in line:
            has_classification = True
            parts = line.split("[UNIVERSE] Non-Volatile:", 1)
            content = parts[1].strip()
            if "(" in content:
                content = content.split("(", 1)[0].strip()
            for t in content.split(","):
                ticker = t.strip()
                if ticker:
                    nonvolatile_tickers.add(ticker)
                    
    # 1. Parse Entries
    entries = {}
    # Find lines like: TEAM          score=+2.38z  entry=$85.84  shares=194
    entry_pattern = re.compile(r"(\w+)\s+score=.*entry=\$(\d+\.\d+)\s+shares=(\d+)")
    
    selected_tickers = set()
    for line in session_lines:
        if "Selected" in line and "stocks" in line:
            continue
        match = entry_pattern.search(line)
        if match:
            ticker = match.group(1)
            entry_price = float(match.group(2))
            shares = int(match.group(3))
            entries[ticker] = {
                "entry_price": entry_price,
                "shares": shares,
                "remaining_qty": shares,
                "exits": []
            }
            selected_tickers.add(ticker)

    # 2. Parse Exits and Price Updates
    current_bar_check_prices = {}
    
    for line in session_lines:
        if "[BAR CHECK]" in line:
            current_bar_check_prices = {}
            continue
            
        # Extract the text after the INFO prefix
        if "INFO" in line:
            content_part = line.split("INFO", 1)[1].strip()
        else:
            content_part = line.strip()
            
        parts = content_part.split()
        if len(parts) >= 6 and parts[0] in selected_tickers:
            try:
                ticker = parts[0]
                entry_price = float(parts[2])
                current_price = float(parts[3])
                current_bar_check_prices[ticker] = current_price
                if ticker in entries:
                    entries[ticker]["entry_price"] = entry_price
            except ValueError:
                pass

        # Parse BUY orders (initial entry overrides / duplicate submissions)
        buy_match = re.search(r"\[ORDER\] BUY (\d+) shares of (\w+)", line)
        if buy_match:
            qty = int(buy_match.group(1))
            ticker = buy_match.group(2)
            if ticker in entries:
                entries[ticker]["shares"] += qty
                entries[ticker]["remaining_qty"] += qty
            else:
                entries[ticker] = {
                    "entry_price": 0.0,
                    "shares": qty,
                    "remaining_qty": qty,
                    "exits": []
                }
                selected_tickers.add(ticker)

        # Parse SELL orders (Profit Takes / Trailing Stops)
        sell_match = re.search(r"\[ORDER\] SELL (\d+) shares of (\w+)", line)
        if sell_match:
            qty = int(sell_match.group(1))
            ticker = sell_match.group(2)
            
            # Better check: is the current context in the log part of the FLATTEN PHASE block?
            is_flatten = False
            try:
                line_idx = session_lines.index(line)
                for idx in range(max(0, line_idx - 25), line_idx):
                    if "[FLATTEN PHASE]" in session_lines[idx]:
                        is_flatten = True
                        break
            except ValueError:
                pass

            if ticker in entries:
                if is_flatten:
                    # In flatten phase, we liquidate all remaining tracking quantity of this session
                    qty_to_sell = entries[ticker]["remaining_qty"]
                else:
                    # Normal stop/target exit
                    qty_to_sell = min(qty, entries[ticker]["remaining_qty"])
                
                if qty_to_sell > 0:
                    sale_price = current_bar_check_prices.get(ticker)
                    if sale_price is None:
                        # search backwards for the last price of this ticker
                        try:
                            line_idx = session_lines.index(line)
                            for prev_line in reversed(session_lines[:line_idx]):
                                if "INFO" in prev_line:
                                    prev_content = prev_line.split("INFO", 1)[1].strip()
                                else:
                                    prev_content = prev_line.strip()
                                prev_parts = prev_content.split()
                                if len(prev_parts) >= 6 and prev_parts[0] == ticker:
                                    try:
                                        sale_price = float(prev_parts[3])
                                        break
                                    except ValueError:
                                        pass
                        except ValueError:
                            pass
                    
                    if sale_price is not None:
                        entries[ticker]["exits"].append((qty_to_sell, sale_price))
                        entries[ticker]["remaining_qty"] -= qty_to_sell

    # 3. Calculate metrics and tag groups
    for ticker, info in entries.items():
        # Tag group
        if ticker in volatile_tickers:
            info["group"] = "volatile"
        elif ticker in nonvolatile_tickers:
            info["group"] = "nonvolatile"
        else:
            info["group"] = "unclassified"
            
        unresolved_qty = info["remaining_qty"]
        if unresolved_qty > 0:
            last_price = info["entry_price"]
            for prev_line in reversed(session_lines):
                if "INFO" in prev_line:
                    prev_content = prev_line.split("INFO", 1)[1].strip()
                else:
                    prev_content = prev_line.strip()
                prev_parts = prev_content.split()
                if len(prev_parts) >= 6 and prev_parts[0] == ticker:
                    try:
                        last_price = float(prev_parts[3])
                        break
                    except ValueError:
                        pass
            info["exits"].append((unresolved_qty, last_price))
            info["remaining_qty"] = 0

    print("\n" + "="*80)
    print("                    LATEST SESSION TRADING REPORT")
    print("="*80)
    
    # === OVERALL REPORT =============================================================
    print("\n=== OVERALL REPORT =============================================================")
    
    if has_classification:
        headers = ["Ticker", "Shares", "Group", "Entry Px", "Total Cost", "Total Revenue", "P&L ($)", "P&L (%)"]
        widths = [8, 8, 12, 10, 14, 14, 12, 10]
    else:
        headers = ["Ticker", "Shares", "Entry Px", "Total Cost", "Total Revenue", "P&L ($)", "P&L (%)"]
        widths = [8, 8, 10, 14, 14, 12, 10]
        
    print(format_row(headers, widths))
    print("-" * (sum(widths) + len(widths)*3 - 3))
    
    overall_cost = 0
    overall_revenue = 0
    overall_wins = 0
    overall_losses = 0
    overall_trades_data = []
    
    for ticker, info in entries.items():
        qty = info["shares"]
        entry_price = info["entry_price"]
        total_cost = qty * entry_price
        total_revenue = sum(eqty * eprice for eqty, eprice in info["exits"])
        pnl_usd = total_revenue - total_cost
        pnl_pct = (pnl_usd / total_cost) * 100 if total_cost > 0 else 0.0
        
        overall_cost += total_cost
        overall_revenue += total_revenue
        
        if pnl_usd >= 0.01:
            overall_wins += 1
        elif pnl_usd <= -0.01:
            overall_losses += 1
        else:
            overall_wins += 1
            
        overall_trades_data.append({
            'ticker': ticker,
            'pnl_usd': pnl_usd,
            'pnl_pct': pnl_pct,
            'cost': total_cost,
            'revenue': total_revenue
        })
            
        row = [ticker, str(qty)]
        if has_classification:
            row.append(info["group"].upper())
        row.extend([
            f"${entry_price:.2f}",
            f"${total_cost:,.2f}",
            f"${total_revenue:,.2f}",
            f"${pnl_usd:+,.2f}",
            f"{pnl_pct:+.2f}%"
        ])
        print(format_row(row, widths))
        
    print_section_summary("OVERALL", "Total Selected Tickers", len(entries), overall_cost, overall_revenue, overall_wins, overall_losses, overall_trades_data)
    
    if has_classification:
        # === VOLATILE REPORT ============================================================
        volatile_entries = {t: info for t, info in entries.items() if info["group"] == "volatile"}
        if volatile_entries:
            print("\n=== VOLATILE REPORT ============================================================")
            sub_headers = ["Ticker", "Shares", "Entry Px", "Total Cost", "Total Revenue", "P&L ($)", "P&L (%)"]
            sub_widths = [8, 8, 10, 14, 14, 12, 10]
            print(format_row(sub_headers, sub_widths))
            print("-" * (sum(sub_widths) + len(sub_widths)*3 - 3))
            
            v_cost = 0
            v_revenue = 0
            v_wins = 0
            v_losses = 0
            for ticker, info in volatile_entries.items():
                qty = info["shares"]
                entry_price = info["entry_price"]
                total_cost = qty * entry_price
                total_revenue = sum(eqty * eprice for eqty, eprice in info["exits"])
                pnl_usd = total_revenue - total_cost
                pnl_pct = (pnl_usd / total_cost) * 100 if total_cost > 0 else 0.0
                
                v_cost += total_cost
                v_revenue += total_revenue
                if pnl_usd >= 0.01:
                    v_wins += 1
                elif pnl_usd <= -0.01:
                    v_losses += 1
                else:
                    v_wins += 1
                    
                row = [
                    ticker,
                    str(qty),
                    f"${entry_price:.2f}",
                    f"${total_cost:,.2f}",
                    f"${total_revenue:,.2f}",
                    f"${pnl_usd:+,.2f}",
                    f"{pnl_pct:+.2f}%"
                ]
                print(format_row(row, sub_widths))
            v_trades_data = [t for t in overall_trades_data if t['ticker'] in volatile_entries]
            print_section_summary("VOLATILE", "Volatile Tickers", len(volatile_entries), v_cost, v_revenue, v_wins, v_losses, v_trades_data)
            
        # === NON-VOLATILE REPORT ========================================================
        nonvolatile_entries = {t: info for t, info in entries.items() if info["group"] == "nonvolatile"}
        if nonvolatile_entries:
            print("\n=== NON-VOLATILE REPORT ========================================================")
            sub_headers = ["Ticker", "Shares", "Entry Px", "Total Cost", "Total Revenue", "P&L ($)", "P&L (%)"]
            sub_widths = [8, 8, 10, 14, 14, 12, 10]
            print(format_row(sub_headers, sub_widths))
            print("-" * (sum(sub_widths) + len(sub_widths)*3 - 3))
            
            nv_cost = 0
            nv_revenue = 0
            nv_wins = 0
            nv_losses = 0
            for ticker, info in nonvolatile_entries.items():
                qty = info["shares"]
                entry_price = info["entry_price"]
                total_cost = qty * entry_price
                total_revenue = sum(eqty * eprice for eqty, eprice in info["exits"])
                pnl_usd = total_revenue - total_cost
                pnl_pct = (pnl_usd / total_cost) * 100 if total_cost > 0 else 0.0
                
                nv_cost += total_cost
                nv_revenue += total_revenue
                if pnl_usd >= 0.01:
                    nv_wins += 1
                elif pnl_usd <= -0.01:
                    nv_losses += 1
                else:
                    nv_wins += 1
                    
                row = [
                    ticker,
                    str(qty),
                    f"${entry_price:.2f}",
                    f"${total_cost:,.2f}",
                    f"${total_revenue:,.2f}",
                    f"${pnl_usd:+,.2f}",
                    f"{pnl_pct:+.2f}%"
                ]
                print(format_row(row, sub_widths))
            nv_trades_data = [t for t in overall_trades_data if t['ticker'] in nonvolatile_entries]
            print_section_summary("NON-VOLATILE", "Non-Volatile Tickers", len(nonvolatile_entries), nv_cost, nv_revenue, nv_wins, nv_losses, nv_trades_data)
            
    print("\n" + "="*80 + "\n")

if __name__ == "__main__":
    log_path = sys.argv[1] if len(sys.argv) > 1 else "live_logs/pipeline.log"
    parse_log(log_path)
