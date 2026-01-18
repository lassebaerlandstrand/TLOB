import json
import sys
from backtesting import Backtest, Strategy
import torch.nn as nn
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import constants as cst

class Inventory(Strategy):
    """
    Grid-based strategy that pyramids positions.
    Gradually builds long/short positions by adding to existing positions,
    and gradually closes them by reducing position size.
    """
    stop_loss_pct = 0.0001
    min_hold_ticks = 9
    
    def init(self):
        self.entry_bar = None
        self.num_orders_raw = 1          # Number of orders in pyramid
        self.num_close_orders = 1        # Number of closing orders
        return

    def next(self):
        current_price = self.data.Close[-1]
        pred = self.data.Preds[-1]
        
        # Calculate price range over last 101 bars for volatility
        high = max(self.data.Close[-101:])
        low = min(self.data.Close[-101:])
        diff = high - low
        market_fee = 0
        
        # Position sizing with grid approach
        size = 100000 * self.num_orders_raw 
        close_size = 100000 * self.num_close_orders
        # Handle buy signal (pred == 0)
        if pred == 0:
            if not self.position and diff > market_fee:
                sl = current_price * (1 - self.stop_loss_pct)
                self.buy(size=size, sl=sl)
                self.num_orders_raw += 1
                self.entry_bar = len(self.data)
            elif self.position.is_short:
                # Gradually close short position by buying
                if abs(self.position.size) >= close_size:
                    self.buy(size=close_size)
                    self.num_close_orders += 1
                    self.num_orders_raw = 1
                else:
                    # Close remaining position
                    self.position.close()
                    self.num_close_orders = 1
                    self.num_orders_raw = 1
                self.entry_bar = len(self.data)
            elif self.position.is_long and self.position.size < 31 and diff > market_fee:
                sl = current_price * (1 - self.stop_loss_pct)
                self.buy(size=size, sl=sl)
                self.num_orders_raw += 1
                self.entry_bar = len(self.data)
                self.num_close_orders = 1
                
        # Handle sell signal (pred == 2)
        elif pred == 2:
            if not self.position and diff > market_fee:
                sl = current_price * (1 + self.stop_loss_pct)
                self.sell(size=size, sl=sl)
                self.entry_bar = len(self.data)
                self.num_orders_raw += 1
            elif self.position.is_long:
                # Gradually close long position by selling
                if abs(self.position.size) >= close_size:
                    self.sell(size=close_size)
                    self.num_close_orders += 1
                    self.num_orders_raw = 1
                else:
                    # Close remaining position
                    self.position.close()
                    self.num_close_orders = 1
                    self.num_orders_raw = 1
                self.entry_bar = len(self.data)
            elif self.position.is_short and self.position.size > -31 and diff > market_fee:
                sl = current_price * (1 + self.stop_loss_pct)
                self.sell(size=size, sl=sl)
                self.num_orders_raw += 1
                self.entry_bar = len(self.data)
                self.num_close_orders = 1
     
class ConservativeReversal(Strategy):
    """
    Closes opposing positions without immediately reversing.
    When receiving a signal to reverse, it closes the current position
    and waits for the next bar to enter a new position.
    """
    stop_loss_pct = 0.0001
    min_hold_ticks = 9
    
    def init(self):
        self.entry_bar = None
        return

    def next(self):
        bet_amount = 100000
        current_price = self.data.Close[-1]
        pred = self.data.Preds[-1]
        size = bet_amount // current_price
        
        # Calculate price range over last 21 bars as volatility proxy
        high = max(self.data.Close[-21:])
        low = min(self.data.Close[-21:])
        diff = high - low
        market_fee = 0

        # Handle buy signal (pred == 0)
        if pred == 0:
            if not self.position and diff > market_fee:
                # Enter new long position
                sl = current_price * (1 - self.stop_loss_pct)
                self.buy(size=size, sl=sl)
                self.entry_bar = len(self.data)
            elif self.position.is_short:
                # Close short position (will reverse on next bar if signal persists)
                self.position.close()
                self.entry_bar = None
                
        # Handle sell signal (pred == 2)
        elif pred == 2:
            if not self.position and diff > market_fee:
                # Enter new short position
                sl = current_price * (1 + self.stop_loss_pct)
                self.sell(size=size, sl=sl)
                self.entry_bar = len(self.data)
            elif self.position.is_long:
                # Close long position (will reverse on next bar if signal persists)
                self.position.close()
                self.entry_bar = None     
                
class AggressiveReversal(Strategy):
    """
    Immediately reverses positions without waiting.
    When receiving a signal to reverse, it closes the current position
    and immediately opens a new opposite position in the same bar.
    """
    stop_loss_pct = 0.0001
    min_hold_ticks = 9
    
    def init(self):
        self.entry_bar = None
        return

    def next(self):
        bet_amount = 100000
        current_price = self.data.Close[-1]
        pred = self.data.Preds[-1]
        size = bet_amount // current_price
        
        # Calculate price range over last 21 bars as volatility proxy
        high = max(self.data.Close[-21:])
        low = min(self.data.Close[-21:])
        diff = high - low
        market_fee = 0

        # Handle buy signal (pred == 0)
        if pred == 0:
            if not self.position and diff > market_fee:
                # Enter new long position
                sl = current_price * (1 - self.stop_loss_pct)
                self.buy(size=size, sl=sl)
                self.entry_bar = len(self.data)
            elif self.position.is_short:
                # Immediately reverse: close short and enter long in same bar
                self.position.close()
                self.entry_bar = None
                sl = current_price * (1 - self.stop_loss_pct)
                self.buy(size=size, sl=sl)
                
        # Handle sell signal (pred == 2)
        elif pred == 2:
            if not self.position and diff > market_fee:
                # Enter new short position
                sl = current_price * (1 + self.stop_loss_pct)
                self.sell(size=size, sl=sl)
                self.entry_bar = len(self.data)
            elif self.position.is_long:
                # Immediately reverse: close long and enter short in same bar
                self.position.close()
                self.entry_bar = None
                sl = current_price * (1 + self.stop_loss_pct)
                self.sell(size=size, sl=sl)    
    
                
def run_backtest(config, dir_path):
    """
    Run backtests for all strategy variants and collect performance metrics.
    
    Args:
        config: Configuration object with experiment parameters
        dir_path: Directory path containing result.csv with predictions
        
    Returns:
        Dictionary with performance metrics for each strategy
    """
    results = {
        "returns": [],
        "buy_and_hold_returns": [],
        "sharpe_ratios": [],
        "betas": [],
        "trades_count": [],
        "avg_return_per_trade": []
    }
    
    # Load prediction results from CSV
    data = pd.read_csv(os.path.join(dir_path, f"result.csv"))
    
    # Convert timestamp to datetime and handle duplicates
    data['timestamp'] = pd.to_datetime(data['timestamp'])
    duplicates_mask = data['timestamp'].duplicated(keep=False)
    if duplicates_mask.any():
        # Add millisecond offsets to duplicate timestamps
        duplicate_groups = data[duplicates_mask].groupby('timestamp').cumcount()
        data.loc[duplicates_mask, 'timestamp'] = (
            data.loc[duplicates_mask, 'timestamp'] + 
            pd.to_timedelta(duplicate_groups, unit='ms')
        )
    if data['timestamp'].duplicated().any():
        print("Warning: Duplicates still exist after offset. Removing duplicates...")
        data = data.drop_duplicates(subset='timestamp', keep='first')
    
    # Prepare OHLC data for backtesting (using price for all OHLC values)
    data.set_index('timestamp', inplace=True)
    data['Open'] = data['price']
    data['High'] = data['price']
    data['Low'] = data['price']
    data['Close'] = data['price']

    # Remove rows with missing predictions
    data = data.dropna(subset=["Preds"])    
    OHLC = data[["Preds", "Open", "High", "Low", "Close"]]
    # Perform backtesting with all three strategy variants
    Strategies = [ConservativeReversal, AggressiveReversal, Inventory]
    # Backtest each strategy variant
    for strategy in Strategies:
        # Initialize backtest with starting capital, commission, and other parameters
        bt = Backtest(OHLC, strategy, cash=100000000, commission=0.00005, margin=1, 
                      trade_on_close=False, finalize_trades=True)

        # Run backtest with strategy parameters
        stats = bt.run(min_hold_ticks=config.experiment.horizon, stop_loss_pct=config.experiment.stop_loss)
        # Extract equity curve metrics
        equity_curve = stats['_equity_curve']['Equity']
        min_equity = equity_curve.min()
        initial_cash = 100000000
        min_equity_pct = ((min_equity - initial_cash) / initial_cash) * 100
        
        results["min_equity"] = min_equity
        
        # Extract and save trade results
        trades_df = stats['_trades']
        trade_count = len(trades_df)
        
        # Save overall strategy statistics
        stats_df = pd.DataFrame([stats])
        stats_df.to_csv(os.path.join(dir_path, f"backtest_stats_{strategy.__name__}.csv"), index=False)
        results["trades_count"].append(trade_count)
        
        # ...existing code...
        
        # Print summary statistics
        print("stats for strategy:", strategy.__name__)
        print(f"Minimum Equity: {min_equity:,.2f} ({min_equity_pct:.2f}%)")
        print(f"Maximum Drawdown: {stats['Max. Drawdown [%]']:.2f}%")
        
        if not trades_df.empty:
            trades_log = []
            for index, trade in trades_df.iterrows():
                # Get price at decision time (one bar before entry)
                decision_bar_idx_entry = trade.EntryBar - 1
                price_at_decision = OHLC.iloc[decision_bar_idx_entry]['Close']
                
                # Classify trade and calculate stop loss levels
                if trade.Size > 0:  # Long trade
                    decision = 'BUY_LONG'
                    sl_calc = price_at_decision * (1 - stats._strategy.stop_loss_pct)
                else:  # Short trade
                    decision = 'SELL_SHORT'
                    sl_calc = price_at_decision * (1 + stats._strategy.stop_loss_pct)

                # Determine why trade was exited
                exit_reason = 'SIGNAL_CLOSE'
                if np.isclose(trade.ExitPrice, sl_calc):
                    exit_reason = 'STOP_LOSS'

                trade_info = {
                    'decision': decision,
                    'price_at_decision': price_at_decision,
                    'price_at_entry': trade.EntryPrice,
                    'price_at_exit': trade.ExitPrice,
                    'exit_reason': exit_reason,
                    'pnl': trade.PnL,
                    'entry_time': trade.EntryTime,
                    'exit_time': trade.ExitTime
                }
                trades_log.append(trade_info)
            
            # Save detailed trade log to CSV
            log_df = pd.DataFrame(trades_log)
            log_df.to_csv(os.path.join(dir_path, f"trades_{strategy.__name__}.csv"), index=False)

            # Collect performance metrics
            results["returns"].append(stats['Return [%]'])
            results["buy_and_hold_returns"].append(stats['Buy & Hold Return [%]'])
            results["betas"].append(stats['Beta'])
            results["sharpe_ratios"].append(stats['Sharpe Ratio'])
            
            # Calculate average return per trade
            avg_return_per_trade = stats['Return [%]'] / trade_count if trade_count > 0 else 0
            results["avg_return_per_trade"].append(avg_return_per_trade)
        else:
            # No trades executed - add zeros for all metrics
            results["returns"].append(0)
            results["buy_and_hold_returns"].append(0)
            results["betas"].append(0)
            results["sharpe_ratios"].append(0)
            results["avg_return_per_trade"].append(0)
        # Print final performance summary
        print("stats for strategy:", strategy.__name__)
        print(f"Average Return per Trade: {results['avg_return_per_trade'][-1]}%")
        print(stats)
    
    return results


def table_plot(results, tickers, dir_path):
    """
    Create and save a visual comparison table of backtest results.
    
    Args:
        results: Dictionary containing metrics for each strategy
        tickers: List of ticker symbols tested
        dir_path: Directory path to save the plot
    """
    # Define metrics and their display labels
    metrics = ["returns", "buy_and_hold_returns", "sharpe_ratios", "betas", "trades_count", "avg_return_per_trade"]
    metric_labels = ["Returns [%]", "Buy & Hold Returns [%]", "Sharpe Ratio", "Betas", "Trades Count", "Avg Return per Trade [%]"]

    # Calculate summary statistics for each metric
    averages = []
    sums = []
    for metric in metrics:
        avg = np.mean(results[metric])
        total = np.sum(results[metric])
        averages.append(avg)
        sums.append(total)
    
    # Build data matrix with individual and aggregate columns
    extended_tickers = tickers + ["Average", "Sum"]
    matrix = np.vstack([results[metric] for metric in metrics])
    Returns_with_stats = np.column_stack([matrix, averages, sums])
    
    # Create heatmap visualization
    fig, ax = plt.subplots(figsize=(35, 12))
    im = ax.matshow(Returns_with_stats, cmap=plt.cm.gray_r, alpha=0.3)
    
    # Configure axes labels
    ax.set_xticks(np.arange(len(extended_tickers)), minor=False)
    ax.set_yticks(np.arange(len(metrics)), minor=False)
    ax.set_xticklabels(extended_tickers)
    ax.set_yticklabels(metric_labels)
    
    # Add color-coded value annotations
    for i in range(len(metrics)):
        for j in range(len(extended_tickers)):
            value = Returns_with_stats[i, j]
            
            # Color code: red for negative, green for positive, black for zero
            if value < 0:
                text_color = 'red'
            elif value > 0:
                text_color = 'green'
            else:
                text_color = 'black'
            
            # Highlight aggregate columns with bold text
            if j >= len(tickers):  # Average or Sum column
                ax.text(j, i, round(value, 2), ha="center", va="center", 
                       fontweight='bold', fontsize=12, color=text_color)
            else:
                ax.text(j, i, round(value, 2), ha="center", va="center", 
                       color=text_color)
    
    # Add visual separators between tickers and aggregate columns
    ax.axvline(x=len(tickers)-0.5, color='red', linestyle='--', linewidth=2)
    ax.axvline(x=len(tickers)+0.5, color='blue', linestyle='--', linewidth=2)
    
    # Configure plot appearance and save
    plt.title("Backtesting Results by Ticker (with Average and Sum)")
    plt.xlabel("Tickers")
    plt.ylabel("Metrics")
    plt.colorbar(im)
    plt.tight_layout()
    plt.savefig(os.path.join(dir_path, "backtest", "table_plot.pdf"), dpi=300, bbox_inches='tight')