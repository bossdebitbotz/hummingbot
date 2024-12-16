from decimal import Decimal
from typing import List, Optional

import pandas as pd
from scipy.signal import find_peaks

from hummingbot.client.ui.interface_utils import format_df_for_printout
from hummingbot.core.data_type.common import TradeType
from hummingbot.data_feed.candles_feed.data_types import CandlesConfig
from hummingbot.strategy_v2.controllers import DirectionalTradingControllerBase, DirectionalTradingControllerConfigBase
from hummingbot.strategy_v2.executors.grid_executor.data_types import GridExecutorConfig
from hummingbot.strategy_v2.models.executor_actions import ExecutorAction, StopExecutorAction


class NGriTConfig(DirectionalTradingControllerConfigBase):
    """
    Configuration required to run the Dneitor strategy for one connector and trading pair.
    """

    controller_name: str = "ngrit"
    candles_config: List[CandlesConfig] = []
    connector_name: str = "binance_perpetual"
    trading_pair: str = "WLD-USDT"
    candles_connector: str = "binance_perpetual"
    candles_trading_pair: str = "WLD-USDT"
    interval: str = "1m"
    # MACD
    macd_fast: int = 21
    macd_slow: int = 42
    macd_signal: int = 9
    # EMAs
    ema_short: int = 8
    ema_medium: int = 29
    ema_long: int = 31

    # ATR
    atr_length: int = 11
    atr_multiplier: float = 1.5

    donchian_channel_length = 50
    prominence_pct_peaks = 0.05
    distance_between_peaks = 100

    close_signal_activated: bool = True
    grid_update_interval: Optional[int] = None
    activation_bounds: Decimal = Decimal("0.002")
    min_spread_between_orders: Decimal = Decimal("0.0002")
    min_order_amount_quote: Decimal = Decimal("10")
    max_open_orders: int = 3
    max_orders_per_batch: int = 1
    order_frequency: int = 9
    min_risk_reward_ratio: float = 3.0
    tp_default: float = 0.05
    min_distance_to_limit_price_pct: float = 0.05
    side_filter: int = 0


class NGriTController(DirectionalTradingControllerBase):
    def __init__(self, config: NGriTConfig, *args, **kwargs):
        self.config = config
        self.max_records = (
            max(
                config.macd_slow,
                config.macd_fast,
                config.macd_signal,
                config.atr_length,
                config.ema_short,
                config.ema_medium,
                config.ema_long,
                config.donchian_channel_length,
            )
            + 20
        )
        if len(self.config.candles_config) == 0:
            self.config.candles_config = [
                CandlesConfig(
                    connector=config.candles_connector,
                    trading_pair=config.candles_trading_pair,
                    interval=config.interval,
                    max_records=self.max_records,
                )
            ]
        super().__init__(config, *args, **kwargs)

    async def update_processed_data(self):
        df = self.market_data_provider.get_candles_df(
            connector_name=self.config.candles_connector,
            trading_pair=self.config.candles_trading_pair,
            interval=self.config.interval,
            max_records=self.max_records,
        )
        # Add indicators
        df.ta.macd(fast=self.config.macd_fast, slow=self.config.macd_slow, signal=self.config.macd_signal, append=True)
        df.ta.atr(length=self.config.atr_length, append=True)
        df.ta.ema(length=self.config.ema_short, append=True)
        df.ta.ema(length=self.config.ema_medium, append=True)
        df.ta.ema(length=self.config.ema_long, append=True)
        df["long_atr_support"] = (
            df["close"].shift(1) - df[f"ATRr_{self.config.atr_length}"] * self.config.atr_multiplier
        )
        df["short_atr_resistance"] = (
            df["close"].shift(1) + df[f"ATRr_{self.config.atr_length}"] * self.config.atr_multiplier
        )

        macdh = df[f"MACDh_{self.config.macd_fast}_{self.config.macd_slow}_{self.config.macd_signal}"]
        short_ema = df[f"EMA_{self.config.ema_short}"]
        medium_ema = df[f"EMA_{self.config.ema_medium}"]
        long_ema = df[f"EMA_{self.config.ema_long}"]
        close = df["close"]

        # Base long condition
        entry_long_condition = (
            (short_ema > medium_ema)
            & (medium_ema > long_ema)
            & (close > short_ema)
            & (close > df["long_atr_support"])
            & (macdh > 0)
            & (macdh > macdh.shift(1))  # MACD histogram increasing
        )

        # Base short condition
        entry_short_condition = (
            (short_ema < medium_ema)
            & (medium_ema < long_ema)
            & (close < short_ema)
            & (close < df["short_atr_resistance"])
            & (macdh < 0)
            & (macdh < macdh.shift(1))  # MACD histogram decreasing
        )

        # Entry Signal
        df["signal"] = 0
        df.loc[entry_long_condition, "signal"] = 1
        df.loc[entry_short_condition, "signal"] = -1

        # Close Signal
        # Add donchian channel
        df["donchian_high"] = df["high"].rolling(self.config.donchian_channel_length).max()
        df["donchian_low"] = df["low"].rolling(self.config.donchian_channel_length).min()

        close_long_condition = close < df["donchian_low"]
        close_short_condition = close > df["donchian_high"]
        df["close_signal"] = 0
        df.loc[close_long_condition, "close_signal"] = 1
        df.loc[close_short_condition, "close_signal"] = -1

        peaks = self.get_peaks(
            df,
            prominence_percentage=self.config.prominence_pct_peaks,
            distance=self.config.distance_between_peaks,
        )

        high_peaks = peaks["high_peaks"]
        low_peaks = peaks["low_peaks"]
        df.loc[high_peaks[0], "TP_LONG"] = high_peaks[1]
        df.loc[low_peaks[0], "TP_SHORT"] = low_peaks[1]
        df["TP_LONG"].ffill(inplace=True)
        df["TP_SHORT"].ffill(inplace=True)

        # Apply the function to create the TP_LONG column
        df["TP_LONG"] = df.apply(
            lambda x: x.TP_LONG
            if pd.notna(x.TP_LONG) and x.TP_LONG > x.high
            else self.get_unbounded_tp(x, self.config.tp_default, TradeType.BUY, high_peaks, low_peaks),
            axis=1,
        )
        df["TP_SHORT"] = df.apply(
            lambda x: x.TP_SHORT
            if pd.notna(x.TP_SHORT) and x.TP_SHORT < x.low
            else self.get_unbounded_tp(x, self.config.tp_default, TradeType.SELL, high_peaks, low_peaks),
            axis=1,
        )

        df["SL_LONG"] = df["donchian_low"]
        df["SL_SHORT"] = df["donchian_high"]
        df["LIMIT_LONG"] = df["SL_LONG"] - df[f"ATRr_{self.config.atr_length}"] * self.config.atr_multiplier
        df["LIMIT_SHORT"] = df["SL_SHORT"] + df[f"ATRr_{self.config.atr_length}"] * self.config.atr_multiplier
        # Update processed data
        self.processed_data.update(df.iloc[-1].to_dict())
        self.processed_data["features"] = df

    def can_create_executor(self, signal: int) -> bool:
        """
        Check if an executor can be created based on the signal, the quantity of active executors and the cooldown time.
        """
        active_executors_by_signal_side = self.filter_executors(
            executors=self.executors_info,
            filter_func=lambda x: x.is_active and (x.side == TradeType.BUY if signal > 0 else TradeType.SELL),
        )
        max_timestamp = max([executor.timestamp for executor in active_executors_by_signal_side], default=0)
        active_executors_condition = len(active_executors_by_signal_side) < self.config.max_executors_per_side
        cooldown_condition = self.market_data_provider.time() - max_timestamp > self.config.cooldown_time
        if signal == 1:
            risk_pct = (self.processed_data["close"] - self.processed_data["SL_LONG"]) / self.processed_data["close"]
            reward_pct = (self.processed_data["TP_LONG"] - self.processed_data["close"]) / self.processed_data["close"]
            distance_to_limit_price_pct = (
                self.processed_data["LIMIT_LONG"] - self.processed_data["close"]
            ) / self.processed_data["close"]
        else:
            risk_pct = (self.processed_data["SL_SHORT"] - self.processed_data["close"]) / self.processed_data["close"]
            reward_pct = (self.processed_data["TP_SHORT"] - self.processed_data["close"]) / self.processed_data["close"]
            distance_to_limit_price_pct = (
                self.processed_data["LIMIT_SHORT"] - self.processed_data["close"]
            ) / self.processed_data["close"]
        risk_reward_ratio = abs(reward_pct / risk_pct)
        risk_reward_condition = risk_reward_ratio > self.config.min_risk_reward_ratio
        min_risk_pct_condition = distance_to_limit_price_pct < self.config.min_distance_to_limit_price_pct
        side_filter_condition = self.config.side_filter == signal if self.config.side_filter != 0 else True
        self.processed_data["risk_pct"] = risk_pct
        self.processed_data["reward_pct"] = reward_pct
        self.processed_data["risk_reward_ratio"] = risk_reward_ratio
        self.processed_data["distance_to_limit_price_pct"] = distance_to_limit_price_pct
        return (
            active_executors_condition
            and cooldown_condition
            and risk_reward_condition
            and min_risk_pct_condition
            and side_filter_condition
        )

    def stop_actions_proposal(self) -> List[ExecutorAction]:
        """
        Stop actions based on the provided executor handler report.
        """
        stop_actions = []
        if self.config.close_signal_activated:
            signal = self.processed_data["close_signal"]
            if signal != 0:
                executors_to_stop = self.filter_executors(
                    executors=self.executors_info,
                    filter_func=lambda x: x.is_active and (x.side == TradeType.BUY if signal > 0 else TradeType.SELL),
                )
                stop_actions.extend(
                    StopExecutorAction(controller_id=self.config.id, executor_id=executor.id)
                    for executor in executors_to_stop
                )
        return stop_actions

    def get_executor_config(self, trade_type: TradeType, price: Decimal, amount: Decimal):
        """
        Get the executor config based on the trade_type, price and amount. This method can be overridden by the
        subclasses if required.
        """
        if trade_type == TradeType.BUY:
            start_price = self.processed_data["SL_LONG"]
            end_price = self.processed_data["TP_LONG"]
            limit_price = self.processed_data["LIMIT_LONG"]
        else:
            start_price = self.processed_data["TP_SHORT"]
            end_price = self.processed_data["SL_SHORT"]
            limit_price = self.processed_data["LIMIT_SHORT"]

        return GridExecutorConfig(
            timestamp=self.market_data_provider.time(),
            connector_name=self.config.connector_name,
            trading_pair=self.config.trading_pair,
            start_price=start_price,
            end_price=end_price,
            leverage=self.config.leverage,
            limit_price=limit_price,
            side=trade_type,
            total_amount_quote=self.config.total_amount_quote,
            min_spread_between_orders=self.config.min_spread_between_orders,
            min_order_amount_quote=self.config.min_order_amount_quote,
            max_open_orders=self.config.max_open_orders,
            max_orders_per_batch=self.config.max_orders_per_batch,
            order_frequency=self.config.order_frequency,
            activation_bounds=self.config.activation_bounds,
            triple_barrier_config=self.config.triple_barrier_config,
            level_id=None,
        )

    @staticmethod
    def get_unbounded_tp(row, tp_default, side, high_peaks, low_peaks, criteria="latest"):
        timestamp = row.name
        close = row["close"]
        if side == TradeType.BUY:
            previous_peaks_higher_than_price = [
                price_peak
                for price_timestamp, price_peak in zip(high_peaks[0], high_peaks[1])
                if price_timestamp < timestamp and price_peak > close
            ]
            if previous_peaks_higher_than_price:
                if criteria == "latest":
                    return previous_peaks_higher_than_price[-1]
                elif criteria == "closest":
                    return min(previous_peaks_higher_than_price, key=lambda x: abs(x - row["close"]))
            else:
                return close * (1 + tp_default)
        else:
            previous_peaks_lower_than_price = [
                price_peak
                for price_timestamp, price_peak in zip(low_peaks[0], low_peaks[1])
                if price_timestamp < timestamp and price_peak < close
            ]
            if previous_peaks_lower_than_price:
                if criteria == "latest":
                    return previous_peaks_lower_than_price[-1]
                elif criteria == "closest":
                    return min(previous_peaks_lower_than_price, key=lambda x: abs(x - row["close"]))
            else:
                return close * (1 - tp_default)

    def get_peaks(self, candles, prominence_percentage: float = 0.01, distance: int = 5):
        prominence_nominal = self._calculate_prominence(candles, prominence_percentage)
        high_peaks, low_peaks = self._find_price_peaks(candles, prominence_nominal, distance)
        high_peak_prices = candles["high"].iloc[high_peaks]
        low_peak_prices = candles["low"].iloc[low_peaks]
        high_peaks_index = candles.iloc[high_peaks].index
        low_peaks_index = candles.iloc[low_peaks].index
        return {
            "high_peaks": [high_peaks_index, high_peak_prices],
            "low_peaks": [low_peaks_index, low_peak_prices],
        }

    def _calculate_prominence(self, candles, prominence_percentage: float) -> float:
        price_range = candles["high"].max() - candles["low"].min()
        return price_range * prominence_percentage

    def _find_price_peaks(self, candles, prominence_nominal: float, distance: int):
        high_peaks, _ = find_peaks(candles["high"], prominence=prominence_nominal, distance=distance)
        low_peaks, _ = find_peaks(-candles["low"], prominence=prominence_nominal, distance=distance)
        return high_peaks, low_peaks

    def to_format_status(self) -> List[str]:
        df = self.processed_data.get("features", pd.DataFrame())
        if df.empty:
            return []
        columns_to_show = [
            "close",
            "signal",
            "close_signal",
            "TP_LONG",
            "TP_SHORT",
            "SL_LONG",
            "SL_SHORT",
            "LIMIT_LONG",
            "LIMIT_SHORT",
        ]
        lines = [
            format_df_for_printout(
                df[columns_to_show].tail(5),
                table_format="psql",
            )
        ]
        risk_reward_ratio = self.processed_data.get("risk_reward_ratio", None)
        distance_to_limit_price_pct = self.processed_data.get("distance_to_limit_price_pct", None)
        if risk_reward_ratio is not None:
            lines.append(f"Risk Reward Ratio: {risk_reward_ratio}")
        if distance_to_limit_price_pct is not None:
            lines.append(f"Distance to Limit Price: {distance_to_limit_price_pct}")
        return lines
