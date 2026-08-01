import re
import sys
from pathlib import Path

def extract():
    log_file = r'C:\Users\ASUS\.gemini\antigravity-ide\brain\e2581f28-313d-445f-b9a2-44b2d2018df8\.system_generated\tasks\task-78.log'
    with open(log_file, 'r', encoding='utf-8') as f:
        content = f.read()

    blocks = re.findall(r'Ticker        Side   Shares         Entry        Exit      P&L%       P&L \$  Reason\n(.*?)Day P&L:', content, re.DOTALL)
    
    md = "# Detailed Backtesting Results: May 26, 2026\n\n"
    
    titles = ["## Basket 1 (20-Day ATR%)", "## Basket 2 (7-Day ATR%)"]
    
    for i, b in enumerate(blocks):
        md += f"{titles[i]}\n\n"
        md += "| Ticker | Side | Shares | Entry | Exit | P&L% | P&L $ | Reason |\n"
        md += "|---|---|---|---|---|---|---|---|\n"
        
        lines = b.split('\n')
        for line in lines:
            if 'LONG' in line:
                # e.g.: 2026-05-27 12:37:50  INFO        ARQQ          LONG   2537          $16.42      $16.09    -2.00%       $-833  STOP_LOSS
                parts = line.split('INFO')[1].strip().split()
                # parts should be: ['ARQQ', 'LONG', '2537', '$16.42', '$16.09', '-2.00%', '$-833', 'STOP_LOSS']
                if len(parts) >= 8:
                    md += f"| {parts[0]} | {parts[1]} | {parts[2]} | {parts[3]} | {parts[4]} | {parts[5]} | {parts[6]} | {parts[7]} |\n"
                    
        md += "\n"
        
    out_path = r'C:\Users\ASUS\.gemini\antigravity-ide\brain\e2581f28-313d-445f-b9a2-44b2d2018df8\detailed_results.md'
    with open(out_path, 'w', encoding='utf-8') as f:
        f.write(md)

if __name__ == '__main__':
    extract()
