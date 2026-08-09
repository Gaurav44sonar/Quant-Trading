"""
Google Sheets Integration Module for Quant-Trading
===================================================
Provides robust Google Sheets syncing for trading session metrics across 4 strategy tabs:
1. With LLM With Sentiment
2. With LLM Without Sentiment
3. Without LLM With Sentiment
4. Without LLM Without Sentiment
"""

import os
import re
import glob
import logging
import pandas as pd
from typing import List, Dict, Any, Optional

log = logging.getLogger(__name__)

# Standard column schema for all strategy tabs
COLUMNS_ORDER = [
    'date', 'Market', 'Avg Return', 'Cumulative Return', 'Best Trade', 'Worst Trade',
    'Total P&L', 'CAGR (annualized)', 'Sharpe Ratio', 'Sortino Ratio',
    'Max Drawdown', 'Volatility (Ann.)', 'Win Rate', 'Win/Loss Ratio',
    'Avg Win', 'Avg Loss'
]

STRATEGY_MAP = {
    'WITH_LLM_WITH_SENTIMENT': 'With LLM With Sentiment',
    'WITH_LLM_WITHOUT_SENTIMENT': 'With LLM Without Sentiment',
    'WITHOUT_LLM_WITH_SENTIMENT': 'Without LLM With Sentiment',
    'WITHOUT_LLM_WITHOUT_SENTIMENT': 'Without LLM Without Sentiment',
}

SCOPES = [
    'https://www.googleapis.com/auth/spreadsheets',
    'https://www.googleapis.com/auth/drive'
]


def load_env_credentials():
    """
    Attempts to locate Google Sheet ID and Service Account Credentials file path from environment variables,
    .env files, or default locations.
    """
    sheet_id = os.getenv('GOOGLE_SHEET_ID', '').strip()
    creds_path = os.getenv('GOOGLE_SERVICE_ACCOUNT_FILE', '').strip()

    # Search for .env files up to project root if env vars not set
    if not sheet_id or not creds_path:
        search_dirs = [
            os.getcwd(),
            os.path.dirname(os.path.abspath(__file__)),
            os.path.abspath(os.path.join(os.path.dirname(__file__), '..')),
            os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')),
        ]
        for d in search_dirs:
            env_file = os.path.join(d, '.env')
            if os.path.isfile(env_file):
                try:
                    with open(env_file, 'r', encoding='utf-8') as f:
                        for line in f:
                            line = line.strip()
                            if line.startswith('#') or '=' not in line:
                                continue
                            k, v = line.split('=', 1)
                            k = k.strip()
                            v = v.strip().strip('"\'')
                            if k == 'GOOGLE_SHEET_ID' and not sheet_id:
                                sheet_id = v
                            elif k == 'GOOGLE_SERVICE_ACCOUNT_FILE' and not creds_path:
                                creds_path = v
                except Exception:
                    pass

    # Default credentials file lookup if not specified
    if not creds_path:
        possible_creds = [
            'google_credentials.json',
            'credentials.json',
            'config/google_credentials.json',
            os.path.join(os.path.dirname(__file__), 'google_credentials.json'),
            os.path.join(os.path.dirname(__file__), 'credentials.json'),
        ]
        for p in possible_creds:
            if os.path.isfile(p):
                creds_path = os.path.abspath(p)
                break

    return sheet_id, creds_path


def get_gspread_client(creds_path: str):
    """
    Initializes a gspread client using the provided service account JSON key file.
    """
    import gspread
    from google.oauth2.service_account import Credentials

    if not creds_path or not os.path.isfile(creds_path):
        raise FileNotFoundError(
            f"Google Service Account key file not found at: '{creds_path}'. "
            f"Please place 'google_credentials.json' in the project root or set GOOGLE_SERVICE_ACCOUNT_FILE."
        )

    creds = Credentials.from_service_account_file(creds_path, scopes=SCOPES)
    client = gspread.authorize(creds)
    return client


def ensure_worksheet(spreadsheet, tab_name: str):
    """
    Ensures a worksheet tab exists in the Google Spreadsheet and has the standard header row.
    """
    try:
        worksheet = spreadsheet.worksheet(tab_name)
    except Exception:
        # Create worksheet if missing
        worksheet = spreadsheet.add_worksheet(title=tab_name, rows=500, cols=len(COLUMNS_ORDER))

    # Check header
    existing_headers = worksheet.row_values(1)
    if not existing_headers or existing_headers != COLUMNS_ORDER:
        worksheet.update('A1', [COLUMNS_ORDER])
        try:
            # Basic header styling (dark header format)
            worksheet.format('A1:P1', {
                "backgroundColor": {"red": 0.12, "green": 0.31, "blue": 0.47},
                "textFormat": {"bold": True, "foregroundColor": {"red": 1.0, "green": 1.0, "blue": 1.0}},
                "horizontalAlignment": "CENTER"
            })
        except Exception as ex_fmt:
            log.debug(f"Header formatting failed (non-critical): {ex_fmt}")

    return worksheet


def sync_dataframe_to_tab(tab_name: str, df: pd.DataFrame, sheet_id: str = None, creds_path: str = None) -> bool:
    """
    Uploads/updates a DataFrame of session records to the specified tab in Google Sheets.
    Prevents duplicates based on ('date', 'Market').
    """
    if df.empty:
        log.info(f"No records to sync for tab '{tab_name}'.")
        return True

    if not sheet_id or not creds_path:
        env_sheet_id, env_creds_path = load_env_credentials()
        sheet_id = sheet_id or env_sheet_id
        creds_path = creds_path or env_creds_path

    if not sheet_id:
        log.warning(
            f"[GOOGLE SHEETS NOTICE] GOOGLE_SHEET_ID not set. "
            f"Skipping sync for tab '{tab_name}'. To enable, set GOOGLE_SHEET_ID in your .env file."
        )
        return False

    if not creds_path or not os.path.isfile(creds_path):
        log.warning(
            f"[GOOGLE SHEETS NOTICE] Google service account credentials file not found. "
            f"Skipping sync for tab '{tab_name}'."
        )
        return False

    try:
        client = get_gspread_client(creds_path)
        spreadsheet = client.open_by_key(sheet_id)
        worksheet = ensure_worksheet(spreadsheet, tab_name)

        # Fetch existing rows
        all_values = worksheet.get_all_values()
        if not all_values:
            existing_headers = []
            existing_rows = []
        else:
            existing_headers = all_values[0]
            existing_rows = all_values[1:]

        # Prepare new DataFrame matching schema
        for col in COLUMNS_ORDER:
            if col not in df.columns:
                df[col] = ''

        if '_start_time' in df.columns:
            df = df.sort_values(by='_start_time', ascending=True)

        df = df[COLUMNS_ORDER]

        # Build existing keys set e.g. (date, market)
        existing_keys = set()
        for r in existing_rows:
            if len(r) >= 2:
                existing_keys.add((r[0].strip(), r[1].strip()))

        rows_to_append = []
        for _, row in df.iterrows():
            row_vals = [str(row[col]) if pd.notna(row[col]) else '' for col in COLUMNS_ORDER]
            key = (row_vals[0].strip(), row_vals[1].strip())
            if key not in existing_keys:
                rows_to_append.append(row_vals)
                existing_keys.add(key)

        if rows_to_append:
            worksheet.append_rows(rows_to_append, value_input_option='USER_ENTERED')
            log.info(f"Successfully synced {len(rows_to_append)} new records to Google Sheet tab '{tab_name}'.")
        else:
            log.info(f"All records for tab '{tab_name}' are already up to date in Google Sheet.")

        return True

    except Exception as e:
        err_msg = str(e.__cause__) if getattr(e, '__cause__', None) else str(e)
        if not err_msg:
            err_msg = repr(e)
        log.error(f"Failed to sync records to Google Sheet tab '{tab_name}': {err_msg}")
        return False


def parse_report_txt(filepath: str, market_label: str = "") -> Dict[str, str]:
    """
    Parses a live report .txt file into a dictionary matching COLUMNS_ORDER.
    """
    with open(filepath, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    start_time_str = ''
    for line in lines:
        if 'Start Time' in line and ':' in line:
            start_time_str = line.split(':', 1)[1].strip()
            break

    if start_time_str:
        date_part = start_time_str.split()[0]
    else:
        m = re.search(r'(\d{8})_\d{6}', os.path.basename(filepath))
        if m:
            d = m.group(1)
            date_part = f"{d[:4]}-{d[4:6]}-{d[6:8]}"
        else:
            date_part = ''

    if not market_label:
        filename = os.path.basename(filepath).upper()
        if 'INDIA' in filename or 'NSE' in filename:
            market_label = 'INDIA'
        else:
            market_label = 'US'

    target_keys = [
        'Avg Return', 'Cumulative Return', 'Best Trade', 'Worst Trade',
        'Total P&L', 'CAGR (annualized)', 'Sharpe Ratio', 'Sortino Ratio',
        'Max Drawdown', 'Volatility (Ann.)', 'Win Rate', 'Win/Loss Ratio',
        'Avg Win', 'Avg Loss'
    ]

    row_data = {
        'date': date_part,
        'Market': market_label,
        '_start_time': start_time_str
    }
    for key in target_keys:
        row_data[key] = ''

    section1_lines = lines[:45]
    for line in section1_lines:
        for key in target_keys:
            if line.startswith(key):
                rest = line[len(key):].strip()
                tokens = re.findall(r'[$₹]?[+-]?\d[\d,]*\.\d+%?|[$₹]?[+-]?\d+%?|--', rest)

                short_val = tokens[0] if len(tokens) >= 1 else ''
                long_val = tokens[1] if len(tokens) >= 2 else ''
                comb_val = tokens[2] if len(tokens) >= 3 else (tokens[-1] if tokens else '')

                val = comb_val
                if val == '--' or val == '':
                    if long_val != '--' and long_val != '':
                        val = long_val
                    elif short_val != '--' and short_val != '':
                        val = short_val

                row_data[key] = val
                break

    return row_data
