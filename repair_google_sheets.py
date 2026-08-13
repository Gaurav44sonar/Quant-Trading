"""
repair_google_sheets.py
========================
One-time repair script: Reads all 4 tabs from Google Sheets.
If historical rows have data shifted (e.g. column C contains '-2.17%' instead of a Market Condition string like 'Positive'/'Negative'/'Neutral'/'N/A'),
it inserts 'N/A' into column C and shifts all subsequent values right so every row aligns perfectly under COLUMNS_ORDER.
"""

import os
import sys
import re
import gspread

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

from google_sheets_sync import (
    load_env_credentials,
    get_gspread_client,
    COLUMNS_ORDER,
    STRATEGY_MAP
)

sheet_id, creds_path = load_env_credentials()
if not sheet_id or not creds_path or not os.path.isfile(creds_path):
    print("❌ Error: Missing GOOGLE_SHEET_ID or google_credentials.json")
    sys.exit(1)

client = get_gspread_client(creds_path)
spreadsheet = client.open_by_key(sheet_id)

tabs = list(STRATEGY_MAP.values())

for tab_name in tabs:
    print(f"\n--- Checking tab: '{tab_name}' ---")
    try:
        ws = spreadsheet.worksheet(tab_name)
    except Exception as e:
        print(f"  [SKIP] Tab '{tab_name}' does not exist yet.")
        continue

    all_rows = ws.get_all_values()
    if not all_rows:
        print(f"  [SKIP] Tab '{tab_name}' is empty.")
        continue

    # Ensure header is COLUMNS_ORDER
    header = COLUMNS_ORDER
    fixed_rows = [header]
    modified_count = 0

    for row_idx, r in enumerate(all_rows[1:], start=2):
        if not r or not any(r):
            continue

        # Make sure row has at least 3 elements
        while len(r) < 3:
            r.append('')

        val_c = r[2].strip()

        # Check if val_c is a percentage or number/dollar (e.g. '-2.17%', '0.08%', '47.49%')
        # indicating an unshifted old row where Column C was Avg Return instead of Market Condition
        is_numeric_or_pct = bool(re.search(r'[+-]?\d+[\d,]*\.\d+%?|\$', val_c))
        is_known_condition = val_c in ['Positive', 'Negative', 'Neutral', 'N/A']

        if is_numeric_or_pct and not is_known_condition:
            # Old unshifted row! Insert 'N/A' at index 2 (column C), shifting rest right
            new_r = [r[0], r[1], 'N/A'] + r[2:]
            modified_count += 1
        else:
            new_r = r

        # Ensure exact length matching COLUMNS_ORDER (17 cols)
        if len(new_r) < len(COLUMNS_ORDER):
            new_r += [''] * (len(COLUMNS_ORDER) - len(new_r))
        else:
            new_r = new_r[:len(COLUMNS_ORDER)]

        fixed_rows.append(new_r)

    if modified_count > 0:
        print(f"  [FIXING] Found {modified_count} unaligned rows in '{tab_name}'. Rewriting sheet...")
        ws.clear()
        ws.update('A1', fixed_rows, value_input_option='USER_ENTERED')
        # Re-apply dark header format
        try:
            ws.format('A1:Q1', {
                "backgroundColor": {"red": 0.12, "green": 0.31, "blue": 0.47},
                "textFormat": {"bold": True, "foregroundColor": {"red": 1.0, "green": 1.0, "blue": 1.0}},
                "horizontalAlignment": "CENTER"
            })
        except Exception:
            pass
        print(f"  [SUCCESS] Tab '{tab_name}' updated cleanly! ({len(fixed_rows)-1} total rows)")
    else:
        print(f"  [OK] Tab '{tab_name}' is already perfectly aligned.")

print("\n============================================================")
print("  All Google Sheet tabs checked and repaired successfully!")
print("============================================================")
