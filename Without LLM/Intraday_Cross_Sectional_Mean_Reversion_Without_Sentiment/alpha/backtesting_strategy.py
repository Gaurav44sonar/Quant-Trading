import pandas as pd
import numpy as np
from backtesting import Strategy, Backtest

class IntradayMaxProfitStrategy(Strategy):
    """
    Intraday Strategy using the established `backtesting` library.
    This replaces the custom IntradayExecutor.

    Stop-loss is tiered:
      - Tier 1: sell 50% of position at 2.5% loss
      - Tier 2: sell 25% of position at 5% loss
      - Tier 3: sell 25% of position at 10% loss
    """
    entry_bar_idx = 3
    # Tiered stop-loss levels (percentage drop from entry)
    sl_tier1_pct = 0.025   # 2.5% stop → sell 50% (Old: 0.03)
    sl_tier2_pct = 0.05    # 5%   stop → sell 25% (Old: 0.06)
    sl_tier3_pct = 0.10    # 10%  stop → sell 25% (Old: 0.12)
    # Weights: fraction of ORIGINAL position to exit at each tier
    sl_tier1_weight = 0.50
    sl_tier2_weight = 0.25
    sl_tier3_weight = 0.25
    trailing_stop_trigger = 0.04  # Old: 0.025
    trailing_stop_pct = 0.0075
    profit_take_1 = 0.015
    profit_take_2 = 0.03
    profit_take_3 = 0.045
    atr_pt_1 = 0.25         # Updated: same as stop loss (0.25 ATR ~ 2.5%)
    atr_pt_2 = 0.50         # Updated: same as stop loss (0.50 ATR ~ 5.0%)
    atr_pt_3 = 1.00         # Updated: same as stop loss (1.00 ATR ~ 10.0%)
    atr_value = None
    pt_weight_1 = 0.50     # Updated: same as stop loss weights
    pt_weight_2 = 0.25     # Updated: same as stop loss weights
    pt_weight_3 = 0.25     # Updated: same as stop loss weights
    exit_bar = 76

    def init(self):
        self.bar_count = 0
        self.entry_price_val = 0.0
        self.high_water = 0.0

        self.pt1_done = False
        self.pt2_done = False
        self.pt3_done = False

        # Track which stop-loss tiers have fired
        self.sl1_done = False
        self.sl2_done = False
        self.sl3_done = False
        # Running fraction of original position already exited via partial exits
        self._exited_fraction = 0.0

    def next(self):
        self.bar_count += 1
        current_bar_idx = self.bar_count - 1

        current_price = self.data.Close[-1]
        current_high = self.data.High[-1]
        current_low = self.data.Low[-1]

        # 1. Entry Logic
        if current_bar_idx == self.entry_bar_idx:
            # Buy with all available cash assigned to this ticker
            self.buy()
            self.entry_price_val = current_price
            self.high_water = current_price
            return

        # 2. Manage existing positions
        if self.position:
            self.high_water = max(self.high_water, current_high)

            # ── Tiered Stop-Loss (partial exits) ──────────────────────
            # Each tier checks if the low has breached the level and,
            # if so, closes the appropriate portion of the CURRENT
            # remaining position.

            # Tier 1: 2.5% loss → sell 50% of original position
            if not self.sl1_done and current_low <= self.entry_price_val * (1 - self.sl_tier1_pct):
                remaining = 1.0 - self._exited_fraction
                if remaining > 0:
                    portion = min(1.0, self.sl_tier1_weight / remaining)
                    self.position.close(portion=portion)
                self._exited_fraction += self.sl_tier1_weight
                self.sl1_done = True
                if not self.position:
                    return

            # Tier 2: 5% loss → sell 25% of original position
            if not self.sl2_done and current_low <= self.entry_price_val * (1 - self.sl_tier2_pct):
                remaining = 1.0 - self._exited_fraction
                if remaining > 0:
                    portion = min(1.0, self.sl_tier2_weight / remaining)
                    self.position.close(portion=portion)
                self._exited_fraction += self.sl_tier2_weight
                self.sl2_done = True
                if not self.position:
                    return

            # Tier 3: 10% loss → sell 25% of original position (full remaining)
            if not self.sl3_done and current_low <= self.entry_price_val * (1 - self.sl_tier3_pct):
                self.position.close()   # close whatever is left
                self.sl3_done = True
                self._exited_fraction = 1.0
                return

            # ── Trailing Stop ─────────────────────────────────────────
            if self.high_water >= self.entry_price_val * (1 + self.trailing_stop_trigger):
                trail_price = self.high_water * (1 - self.trailing_stop_pct)
                if current_low <= trail_price:
                    self.position.close()
                    return

            # Time Exit
            if current_bar_idx >= self.exit_bar:
                self.position.close()
                return

            # Profit Taking (Partial Exits)
            # portion represents % of CURRENT position to close
            if self.atr_pt_1 is not None and self.atr_value is not None:
                target_1 = self.entry_price_val + (self.atr_pt_1 * self.atr_value)
            else:
                target_1 = self.entry_price_val * (1 + self.profit_take_1)
                
            if self.atr_pt_2 is not None and self.atr_value is not None:
                target_2 = self.entry_price_val + (self.atr_pt_2 * self.atr_value)
            else:
                target_2 = self.entry_price_val * (1 + self.profit_take_2)
                
            if self.atr_pt_3 is not None and self.atr_value is not None:
                target_3 = self.entry_price_val + (self.atr_pt_3 * self.atr_value)
            else:
                target_3 = self.entry_price_val * (1 + self.profit_take_3)

            if not self.pt1_done and current_high >= target_1:
                remaining = 1.0 - self._exited_fraction
                if remaining > 0:
                    portion = min(1.0, self.pt_weight_1 / remaining)
                    self.position.close(portion=portion)
                self._exited_fraction += self.pt_weight_1
                self.pt1_done = True
                if not self.position:
                    return

            if not self.pt2_done and current_high >= target_2:
                remaining = 1.0 - self._exited_fraction
                if remaining > 0:
                    portion = min(1.0, self.pt_weight_2 / remaining)
                    self.position.close(portion=portion)
                self._exited_fraction += self.pt_weight_2
                self.pt2_done = True
                if not self.position:
                    return

            if not self.pt3_done and current_high >= target_3:
                self.position.close()   # close whatever is left
                self.pt3_done = True
                self._exited_fraction = 1.0
                return


def prep_ticker_data(ticker: str, panels: dict, date_mask: pd.Series) -> pd.DataFrame:
    """Extract standard OHLCV dataframe for backtesting.py from panels."""
    df = pd.DataFrame(index=panels["close"][date_mask].index)
    
    # Check if open exists in panels, otherwise use close as a fallback
    if "open" in panels: 
        df["Open"] = panels["open"][date_mask][ticker]
    else: 
        df["Open"] = panels["close"][date_mask][ticker]
        
    df["High"] = panels["high"][date_mask][ticker]
    df["Low"] = panels["low"][date_mask][ticker]
    df["Close"] = panels["close"][date_mask][ticker]
    
    if "volume" in panels: 
        df["Volume"] = panels["volume"][date_mask][ticker]
    else: 
        df["Volume"] = 0
        
    return df.dropna(subset=["Close"])
