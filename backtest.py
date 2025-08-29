# -*- coding: utf-8 -*-
# EMA 回測（三策略整合）— 互動圖 + 明日預掛
# by ChatGPT

import io
import re
import time
from dataclasses import dataclass
from typing import List, Tuple, Optional

import numpy as np
import pandas as pd
import yfinance as yf
import streamlit as st
import plotly.graph_objects as go


# =========================
# 基本設定
# =========================
st.set_page_config(
    page_title="EMA 回測（三策略）",
    page_icon="📈",
    layout="wide",
)

DARK_ANNOT_BG = "rgba(255,255,255,0.96)"  # 手機深色背景可讀
DARK_ANNOT_FG = "#111"

# 67 策略的門檻（可依需求調整）
LONG67_CROSS_PCT = 0.002     # 0.2%
LONG67_CONSEC_PCT = 0.0015   # 0.15%
LONG67_CONSEC_N = 2

# 23 策略的門檻
SHORT23_CROSS_PCT = 0.03     # 3%
SHORT23_CONSEC_PCT = 0.01    # 1%
SHORT23_CONSEC_N = 2
SHORT23_TAKE_PROFIT = 0.30   # +30% 固定停利（簡化版）


# =========================
# 工具：代碼標準化 + .TW/.TWO fallback
# =========================
def normalize_ticker(raw: str) -> str:
    """使用者輸入的代碼先大寫；只含數字時預設補 .TW"""
    s = raw.strip().upper()
    if re.fullmatch(r"\d+[A-Z]?", s) and not s.endswith((".TW", ".TWO")):
        s = s + ".TW"  # 先假設上市
    return s


@st.cache_data(show_spinner=False, ttl=600)
def download_df_smart(user_input: str, lookback: str) -> Tuple[str, pd.DataFrame]:
    """
    下載資料（含 .TW/.TWO 自動 fallback）。
    回傳 (實際使用的代碼, DataFrame)
    """
    base = user_input.strip().upper()

    # 候選清單
    if base.endswith((".TW", ".TWO")):
        cands = [base, base[:-3] + (".TWO" if base.endswith(".TW") else ".TW")]
    elif re.fullmatch(r"\d+[A-Z]?", base):
        cands = [base + ".TW", base + ".TWO"]  # 先 TW，再 TWO
    else:
        cands = [base]  # 美股或指數

    last_err = None
    for tk in cands:
        try:
            df = yf.download(
                tk, period=lookback, interval="1d",
                auto_adjust=True, group_by="column", progress=False
            ).reset_index()

            if isinstance(df.columns, pd.MultiIndex):
                df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]

            if df.empty:
                last_err = "empty"
                continue

            # 欄位別名處理
            rename = {}
            for c in list(df.columns):
                low = str(c).lower()
                if "date" in low and "Date" not in df.columns: rename[c] = "Date"
                if "close" in low and "Close" not in df.columns: rename[c] = "Close"
                if "high" in low and "High" not in df.columns: rename[c] = "High"
                if "low"  in low and "Low"  not in df.columns: rename[c] = "Low"
                if "open" in low and "Open" not in df.columns: rename[c] = "Open"
                if "vol"  in low and "Volume" not in df.columns: rename[c] = "Volume"
            if rename:
                df = df.rename(columns=rename)

            need = {"Date", "Close", "High", "Low"}
            if need - set(df.columns):
                last_err = f"缺欄位：{need - set(df.columns)}"
                continue

            return tk, df

        except Exception as e:
            last_err = str(e)
            continue

    raise RuntimeError(
        "Yahoo 無法取得資料，請確認代碼是否為上市(.TW)或上櫃(.TWO)。" +
        ("" if last_err is None else f" 錯誤：{last_err}")
    )


# =========================
# 共用：EMA、統計、Excel
# =========================
def ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()


@dataclass
class Trade:
    buy_date: pd.Timestamp
    sell_date: Optional[pd.Timestamp]
    buy_px: float
    sell_px: Optional[float]
    ret_pct: Optional[float]


def trades_to_df(trs: List[Trade]) -> pd.DataFrame:
    if not trs:
        return pd.DataFrame(columns=["買進日", "賣出日", "進價", "賣價", "利潤%"])
    rows = []
    for t in trs:
        rows.append({
            "買進日":  None if t.buy_date is None else pd.to_datetime(t.buy_date).date(),
            "賣出日":  None if (t.sell_date is None) else pd.to_datetime(t.sell_date).date(),
            "進價":    None if t.buy_px is None else round(float(t.buy_px), 2),
            "賣價":    None if t.sell_px is None else round(float(t.sell_px), 2),
            "利潤%":   None if t.ret_pct is None else round(float(t.ret_pct), 2),
        })
    return pd.DataFrame(rows)


def summarize_df(trades_df: pd.DataFrame) -> pd.DataFrame:
    if trades_df.empty or not trades_df["利潤%"].notna().any():
        return pd.DataFrame({
            "項目": ["總交易數", "已平倉數", "勝率(%)", "總報酬(%)", "每筆平均(%)"],
            "數值": [len(trades_df), 0, 0.0, 0.0, 0.0]
        })
    profits = trades_df["利潤%"].dropna()
    return pd.DataFrame({
        "項目": ["總交易數", "已平倉數", "勝率(%)", "總報酬(%)", "每筆平均(%)"],
        "數值": [
            len(trades_df),
            len(profits),
            round((profits > 0).mean() * 100, 2),
            round(profits.sum(), 2),
            round(profits.mean(), 2)
        ]
    })


def to_excel_bytes(trades_df: pd.DataFrame, summary_df: pd.DataFrame) -> bytes:
    bio = io.BytesIO()
    with pd.ExcelWriter(bio, engine="xlsxwriter") as writer:
        trades_df.to_excel(writer, index=False, sheet_name="交易明細")
        summary_df.to_excel(writer, index=False, sheet_name="摘要統計")
    bio.seek(0)
    return bio.read()


# =========================
# 策略 1：EMA23/EMA67 均線交叉
# =========================
def backtest_cross(df: pd.DataFrame, fast=23, slow=67) -> Tuple[List[Trade], pd.DataFrame]:
    d = df.copy()
    d["Date"] = pd.to_datetime(d["Date"])
    d[f"EMA{fast}"] = ema(d["Close"], fast)
    d[f"EMA{slow}"] = ema(d["Close"], slow)
    d["Signal"] = np.where(d[f"EMA{fast}"] > d[f"EMA{slow}"], 1, -1)
    d["Cross"] = d["Signal"].diff()

    trs: List[Trade] = []
    entry_px = None
    entry_date = None
    for i in range(1, len(d)):
        if d["Cross"].iat[i] == 2:        # 上穿
            entry_px = float(d["Close"].iat[i])
            entry_date = d["Date"].iat[i]
        elif d["Cross"].iat[i] == -2 and entry_px is not None:  # 下破
            sell_px = float(d["Close"].iat[i])
            sell_date = d["Date"].iat[i]
            ret_pct = (sell_px - entry_px) / entry_px * 100
            trs.append(Trade(entry_date, sell_date, entry_px, sell_px, ret_pct))
            entry_px = None
            entry_date = None

    if entry_px is not None:
        trs.append(Trade(entry_date, None, entry_px, None, None))

    return trs, d


def nextday_cross_price(Ef: float, Es: float, fast: int, slow: int) -> float:
    """
    明日收盤使 EMA_fast == EMA_slow 的臨界價。
    推導：Ef' = af*C + (1-af)Ef；Es' = as*C + (1-as)Es；解 Ef' = Es'
    """
    af = 2.0 / (fast + 1.0)
    as_ = 2.0 / (slow + 1.0)
    denom = (af - as_)
    return ((1 - as_) * Es - (1 - af) * Ef) / denom if denom != 0 else np.nan


# =========================
# 策略 2：長線 67EMA（0.2% / 0.15% 連續≥2）
# =========================
def backtest_long67(df: pd.DataFrame,
                    span=67,
                    cross_pct=LONG67_CROSS_PCT,
                    consec_pct=LONG67_CONSEC_PCT,
                    consec_n=LONG67_CONSEC_N) -> Tuple[List[Trade], pd.DataFrame]:
    d = df.copy()
    d["Date"] = pd.to_datetime(d["Date"])
    d["EMA"] = ema(d["Close"], span)

    trs: List[Trade] = []
    entry_px = None
    entry_date = None
    up_streak = 0
    dn_streak = 0

    for i in range(1, len(d)):
        pc = float(d["Close"].iat[i-1])
        pe = float(d["EMA"].iat[i-1])
        cc = float(d["Close"].iat[i])
        ce = float(d["EMA"].iat[i])
        ch = float(d["High"].iat[i])
        cl = float(d["Low"].iat[i])
        dt = d["Date"].iat[i]

        # 連續統計（相對 EMA 偏離）
        up_streak = up_streak + 1 if (cc - ce) / ce >= consec_pct else 0
        dn_streak = dn_streak + 1 if (ce - cc) / ce >= consec_pct else 0

        buy_A = (pc <= pe) and (cc > ce) and (ch >= ce * (1.0 + cross_pct))
        sell_A = (pc >= pe) and (cc < ce) and (cl <= ce * (1.0 - cross_pct))
        buy_B = (up_streak >= consec_n)
        sell_B = (dn_streak >= consec_n)

        if entry_px is None:
            if buy_A or buy_B:
                entry_px = cc
                entry_date = dt
                up_streak = dn_streak = 0
        else:
            if sell_A or sell_B:
                ret_pct = (cc - entry_px) / entry_px * 100.0
                trs.append(Trade(entry_date, dt, entry_px, cc, ret_pct))
                entry_px = None
                entry_date = None
                up_streak = dn_streak = 0

    if entry_px is not None:
        trs.append(Trade(entry_date, None, entry_px, None, None))

    return trs, d


# =========================
# 策略 3：短線 23EMA（3% / 1% + 30% 停利）
# =========================
def backtest_short23(df: pd.DataFrame,
                     span=23,
                     cross_pct=SHORT23_CROSS_PCT,
                     consec_pct=SHORT23_CONSEC_PCT,
                     consec_n=SHORT23_CONSEC_N,
                     take_profit=SHORT23_TAKE_PROFIT) -> Tuple[List[Trade], pd.DataFrame]:
    d = df.copy()
    d["Date"] = pd.to_datetime(d["Date"])
    d["EMA"] = ema(d["Close"], span)

    trs: List[Trade] = []
    entry_px = None
    entry_date = None
    up_streak = 0
    dn_streak = 0
    peak_px = None  # 進場後的最高價（用於停利）

    for i in range(1, len(d)):
        pc = float(d["Close"].iat[i-1])
        pe = float(d["EMA"].iat[i-1])
        cc = float(d["Close"].iat[i])
        ce = float(d["EMA"].iat[i])
        ch = float(d["High"].iat[i])
        cl = float(d["Low"].iat[i])
        dt = d["Date"].iat[i]

        up_streak = up_streak + 1 if (cc - ce) / ce >= consec_pct else 0
        dn_streak = dn_streak + 1 if (ce - cc) / ce >= consec_pct else 0

        buy_A = (pc <= pe) and (cc > ce) and (ch >= ce * (1.0 + cross_pct))
        sell_A = (pc >= pe) and (cc < ce) and (cl <= ce * (1.0 - cross_pct))
        buy_B = (up_streak >= consec_n)
        sell_B = (dn_streak >= consec_n)

        if entry_px is None:
            if buy_A or buy_B:
                entry_px = cc
                entry_date = dt
                peak_px = cc
                up_streak = dn_streak = 0
        else:
            # 固定 +30% 停利（簡化版）
            peak_px = max(peak_px, ch)
            if (peak_px - entry_px) / entry_px >= take_profit:
                ret_pct = (cc - entry_px) / entry_px * 100.0
                trs.append(Trade(entry_date, dt, entry_px, cc, ret_pct))
                entry_px = None
                entry_date = None
                peak_px = None
                up_streak = dn_streak = 0
                continue

            # 其他賣出條件
            if sell_A or sell_B:
                ret_pct = (cc - entry_px) / entry_px * 100.0
                trs.append(Trade(entry_date, dt, entry_px, cc, ret_pct))
                entry_px = None
                entry_date = None
                peak_px = None
                up_streak = dn_streak = 0

    if entry_px is not None:
        trs.append(Trade(entry_date, None, entry_px, None, None))

    return trs, d


# =========================
# 預掛觸發價（解 EMA 明日價）
# =========================
def _p_for_ratio_nextday(E_t: float, span: int, frac: float, side: str) -> float:
    """
    求明日收盤 P，使得「明日收盤相對明日 EMA 的偏離」達到 frac。
    - side='buy'  : P >= (1+frac) * EMA_{t+1}
    - side='sell' : P <= (1-frac) * EMA_{t+1}
    EMA_{t+1} = K*P + (1-K)*E_t,  K=2/(span+1)
    """
    K = 2.0 / (span + 1.0)
    if side == "buy":
        den = 1.0 - K*(1.0 + frac)
        return ((1.0 + frac) * (1.0 - K) / den) * E_t if den > 0 else np.nan
    else:
        den = 1.0 - K + K*frac
        return ((1.0 - frac) * (1.0 - K) / den) * E_t if den > 0 else np.nan


def triggers_for_single_ema(today_close: float, today_ema: float,
                             span: int,
                             cross_frac: float,
                             consec_frac: float,
                             consec_n: int,
                             streak_up: int,
                             streak_dn: int) -> dict:
    """
    回傳 67/23 單 EMA 規則明日觸發價：
      - 突破買 >=, 跌破賣 <=
      - 連續買/賣：須今天已連續 (n-1) 根才顯示門檻
    以及前置條件（昨收需在 EMA 下/上）
    """
    E = float(today_ema)
    C = float(today_close)

    cross_buy = _p_for_ratio_nextday(E, span, cross_frac, "buy")
    cross_sel = _p_for_ratio_nextday(E, span, cross_frac, "sell")

    consec_buy = (_p_for_ratio_nextday(E, span, consec_frac, "buy")
                  if streak_up >= max(0, consec_n - 1) else None)
    consec_sel = (_p_for_ratio_nextday(E, span, consec_frac, "sell")
                  if streak_dn >= max(0, consec_n - 1) else None)

    return dict(
        crossover_buy=None if np.isnan(cross_buy) else round(float(cross_buy), 2),
        crossover_sell=None if np.isnan(cross_sel) else round(float(cross_sel), 2),
        consec_buy=None if consec_buy is None or np.isnan(consec_buy) else round(float(consec_buy), 2),
        consec_sell=None if consec_sel is None or np.isnan(consec_sel) else round(float(consec_sel), 2),
        prereq_buy_ok=(C <= E),
        prereq_sell_ok=(C >= E),
    )


# =========================
# 畫圖（共用）
# =========================
def add_summary_box(fig: go.Figure, text: str):
    """左下角白底黑字摘要框（手機深色主題可讀）"""
    fig.add_annotation(
        xref="paper", yref="paper",
        x=0.01, y=0.02,
        xanchor="left", yanchor="bottom",
        showarrow=False, align="left",
        bgcolor=DARK_ANNOT_BG, bordercolor="#222", borderwidth=1,
        font=dict(color=DARK_ANNOT_FG, size=12),
        text=text,
    )


def draw_cross_chart(d: pd.DataFrame, fast=23, slow=67,
                     trades_df: pd.DataFrame = pd.DataFrame(),
                     title: str = "") -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=d["Date"], y=d["Close"], name="Close", mode="lines",
                             line=dict(color="#4c78ff", width=2)))
    fig.add_trace(go.Scatter(x=d["Date"], y=d[f"EMA{fast}"], name=f"EMA{fast}",
                             mode="lines", line=dict(color="#ff8c00", width=2, dash="dot")))
    fig.add_trace(go.Scatter(x=d["Date"], y=d[f"EMA{slow}"], name=f"EMA{slow}",
                             mode="lines", line=dict(color="#27b09c", width=2, dash="dash")))

    if not trades_df.empty:
        closed = trades_df[trades_df["賣出日"].notna()]
        openpos = trades_df[trades_df["賣出日"].isna()]
        if not closed.empty:
            fig.add_trace(go.Scatter(
                x=pd.to_datetime(closed["買進日"]), y=closed["進價"], name="Buy",
                mode="markers", marker=dict(symbol="triangle-up", color="green", size=12)))
            fig.add_trace(go.Scatter(
                x=pd.to_datetime(closed["賣出日"]), y=closed["賣價"], name="Sell",
                mode="markers", marker=dict(symbol="triangle-down", color="red", size=12)))
        if not openpos.empty:
            fig.add_trace(go.Scatter(
                x=pd.to_datetime(openpos["買進日"]), y=openpos["進價"], name="Open",
                mode="markers", marker=dict(symbol="triangle-up", color="gold", size=13,
                                            line=dict(color="#222", width=1))))
    fig.update_layout(
        title=title, hovermode="x unified",
        xaxis_title="Date", yaxis_title="Price",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        margin=dict(l=10, r=10, t=60, b=10),
    )
    fig.update_xaxes(rangeslider_visible=False)
    return fig


def draw_single_ema_chart(d: pd.DataFrame, span: int,
                          trades_df: pd.DataFrame = pd.DataFrame(),
                          title: str = "") -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=d["Date"], y=d["Close"], name="Close", mode="lines",
                             line=dict(color="#4c78ff", width=2)))
    fig.add_trace(go.Scatter(x=d["Date"], y=d["EMA"], name=f"EMA{span}",
                             mode="lines", line=dict(color="#27b09c", width=2, dash="dot")))

    if not trades_df.empty:
        closed = trades_df[trades_df["賣出日"].notna()]
        openpos = trades_df[trades_df["賣出日"].isna()]
        if not closed.empty:
            fig.add_trace(go.Scatter(
                x=pd.to_datetime(closed["買進日"]), y=closed["進價"], name="Buy",
                mode="markers", marker=dict(symbol="triangle-up", color="green", size=12)))
            fig.add_trace(go.Scatter(
                x=pd.to_datetime(closed["賣出日"]), y=closed["賣價"], name="Sell",
                mode="markers", marker=dict(symbol="triangle-down", color="red", size=12)))
        if not openpos.empty:
            fig.add_trace(go.Scatter(
                x=pd.to_datetime(openpos["買進日"]), y=openpos["進價"], name="Open",
                mode="markers", marker=dict(symbol="triangle-up", color="gold", size=13,
                                            line=dict(color="#222", width=1))))
    fig.update_layout(
        title=title, hovermode="x unified",
        xaxis_title="Date", yaxis_title="Price",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        margin=dict(l=10, r=10, t=60, b=10),
    )
    fig.update_xaxes(rangeslider_visible=False)
    return fig


# =========================
# UI
# =========================
st.title("📈 EMA 回測（三策略整合）— 互動圖＋明日預掛")

with st.sidebar:
    st.header("參數設定")
    ticker_in = st.text_input("股票代碼（例：2330 / 6227 / 0050.TW / NVDA）", value="2330").strip()
    lookback = st.selectbox("歷史期間", ["30d", "0.5y", "1y", "2y"], index=2)

    strat = st.radio(
        "策略",
        ["均線交叉 (EMA23/67)", "長線 67EMA (0.2%/0.15%)", "短線 23EMA (3%/1% +30%停利)"],
        index=0
    )

    fast_span = st.number_input("快速 EMA 天數", min_value=5, max_value=200, value=23, step=1)
    slow_span = st.number_input("慢速 EMA 天數", min_value=5, max_value=400, value=67, step=1)

    show_trig = st.checkbox("顯示明日『預掛』", value=True)
    run = st.button("開始回測 🚀", use_container_width=True)


if run:
    try:
        tk_norm = normalize_ticker(ticker_in)
        with st.spinner("下載資料與回測中…"):
            actual_tk, df = download_df_smart(tk_norm, lookback)
            st.write(f"**目前代碼：** `{actual_tk}`")

            # 共有欄位
            df["Date"] = pd.to_datetime(df["Date"])
            df = df.sort_values("Date").reset_index(drop=True)

            if strat.startswith("均線交叉"):
                trs, d = backtest_cross(df, fast=int(fast_span), slow=int(slow_span))
                trades_df = trades_to_df(trs)
                summary_df = summarize_df(trades_df)

                # 明日交叉臨界價
                Ef = float(ema(df["Close"], int(fast_span)).iloc[-1])
                Es = float(ema(df["Close"], int(slow_span)).iloc[-1])
                Cx = nextday_cross_price(Ef, Es, int(fast_span), int(slow_span))

                state = "多頭中" if Ef > Es else ("空頭中" if Ef < Es else "臨界點")
                hint = (f"若明日收盤 ≤ {Cx:.2f} 可能翻空" if Ef > Es
                        else (f"若明日收盤 ≥ {Cx:.2f} 可能翻多" if Ef < Es else "今日就在臨界，觀察明日收盤。"))

                fig = draw_cross_chart(d, int(fast_span), int(slow_span), trades_df,
                                       f"{actual_tk} - EMA{int(fast_span)}/EMA{int(slow_span)} 回測（{lookback.upper()}）")
                if show_trig:
                    add_summary_box(fig, (
                        f"【摘要】<br>"
                        f"- 狀態：{state}<br>"
                        f"- 交叉臨界價：{Cx:.2f}<br>"
                        f"- 提示：{hint}"
                    ))
                st.plotly_chart(fig, use_container_width=True, config={"displaylogo": False})

            elif strat.startswith("長線 67EMA"):
                trs, d = backtest_long67(df, span=int(slow_span))
                trades_df = trades_to_df(trs)
                summary_df = summarize_df(trades_df)

                # 預掛（67）
                d["EMA"] = ema(d["Close"], int(slow_span))
                d = d.dropna().reset_index(drop=True)

                # 計算最後一天連續狀態
                up_st = dn_st = 0
                for i in range(len(d)):
                    c = float(d["Close"].iat[i]); e = float(d["EMA"].iat[i])
                    if (c - e) / e >= LONG67_CONSEC_PCT:
                        up_st += 1; dn_st = 0
                    elif (e - c) / e >= LONG67_CONSEC_PCT:
                        dn_st += 1; up_st = 0
                    else:
                        up_st = dn_st = 0

                last = d.iloc[-1]
                trig = triggers_for_single_ema(
                    today_close=float(last["Close"]),
                    today_ema=float(last["EMA"]),
                    span=int(slow_span),
                    cross_frac=LONG67_CROSS_PCT,
                    consec_frac=LONG67_CONSEC_PCT,
                    consec_n=LONG67_CONSEC_N,
                    streak_up=up_st, streak_dn=dn_st
                )

                fig = draw_single_ema_chart(d, int(slow_span), trades_df,
                                            f"{actual_tk} - EMA{int(slow_span)}（0.2%/0.15%）回測（{lookback.upper()}）")

                if show_trig:
                    t1 = f"{'—' if trig['crossover_buy'] is None else trig['crossover_buy']}"
                    t2 = f"{'—' if trig['crossover_sell'] is None else trig['crossover_sell']}"
                    t3 = f"{'—' if trig['consec_buy'] is None else trig['consec_buy']}（需已連續）"
                    t4 = f"{'—' if trig['consec_sell'] is None else trig['consec_sell']}（需已連續）"
                    pre = []
                    if not trig["prereq_buy_ok"]:
                        pre.append("⚠️ 昨收未在 EMA 下，嚴格規則下『突破買』前置不成立")
                    if not trig["prereq_sell_ok"]:
                        pre.append("⚠️ 昨收未在 EMA 上，嚴格規則下『跌破賣』前置不成立")
                    add_summary_box(fig, (
                        f"【預掛(EMA{int(slow_span)})】<br>"
                        f"- 突破買 ≥ {t1}<br>"
                        f"- 跌破賣 ≤ {t2}<br>"
                        f"- 連續買 ≥ {t3}<br>"
                        f"- 連續賣 ≤ {t4}<br>"
                        + ("<br>".join(pre) if pre else "")
                    ))
                st.plotly_chart(fig, use_container_width=True, config={"displaylogo": False})

            else:  # 短線 23EMA
                trs, d = backtest_short23(df, span=int(fast_span))
                trades_df = trades_to_df(trs)
                summary_df = summarize_df(trades_df)

                # 預掛（23）
                d["EMA"] = ema(d["Close"], int(fast_span))
                d = d.dropna().reset_index(drop=True)
                up_st = dn_st = 0
                for i in range(len(d)):
                    c = float(d["Close"].iat[i]); e = float(d["EMA"].iat[i])
                    if (c - e) / e >= SHORT23_CONSEC_PCT:
                        up_st += 1; dn_st = 0
                    elif (e - c) / e >= SHORT23_CONSEC_PCT:
                        dn_st += 1; up_st = 0
                    else:
                        up_st = dn_st = 0

                last = d.iloc[-1]
                trig = triggers_for_single_ema(
                    today_close=float(last["Close"]),
                    today_ema=float(last["EMA"]),
                    span=int(fast_span),
                    cross_frac=SHORT23_CROSS_PCT,
                    consec_frac=SHORT23_CONSEC_PCT,
                    consec_n=SHORT23_CONSEC_N,
                    streak_up=up_st, streak_dn=dn_st
                )

                fig = draw_single_ema_chart(d, int(fast_span), trades_df,
                                            f"{actual_tk} - EMA{int(fast_span)}（3%/1% +30%停利）回測（{lookback.upper()}）")
                if show_trig:
                    t1 = f"{'—' if trig['crossover_buy'] is None else trig['crossover_buy']}"
                    t2 = f"{'—' if trig['crossover_sell'] is None else trig['crossover_sell']}"
                    t3 = f"{'—' if trig['consec_buy'] is None else trig['consec_buy']}（需已連續）"
                    t4 = f"{'—' if trig['consec_sell'] is None else trig['consec_sell']}（需已連續）"
                    pre = []
                    if not trig["prereq_buy_ok"]:
                        pre.append("⚠️ 昨收未在 EMA 下，嚴格規則下『突破買』前置不成立")
                    if not trig["prereq_sell_ok"]:
                        pre.append("⚠️ 昨收未在 EMA 上，嚴格規則下『跌破賣』前置不成立")
                    add_summary_box(fig, (
                        f"【預掛(EMA{int(fast_span)})】<br>"
                        f"- 突破買 ≥ {t1}<br>"
                        f"- 跌破賣 ≤ {t2}<br>"
                        f"- 連續買 ≥ {t3}<br>"
                        f"- 連續賣 ≤ {t4}<br>"
                        + ("<br>".join(pre) if pre else "")
                    ))
                st.plotly_chart(fig, use_container_width=True, config={"displaylogo": False})

            # ===== 表格區 =====
            c1, c2 = st.columns([2, 1])
            with c1:
                st.subheader("交易明細")
                if trades_df.empty:
                    st.info("本次參數下尚無交易訊號。")
                else:
                    st.dataframe(trades_df, use_container_width=True)
            with c2:
                st.subheader("摘要統計")
                st.table(summary_df)

            # 下載
            st.subheader("下載")
            xbytes = to_excel_bytes(trades_df, summary_df)
            st.download_button(
                "📄 下載 Excel（交易明細＋摘要）",
                data=xbytes,
                file_name=f"trades_{actual_tk.replace('.','_')}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True
            )

            with st.expander("🧭 策略與計算說明（點我展開）", expanded=False):
                st.markdown(rf"""
**策略 1：EMA 交叉（{int(fast_span)}/{int(slow_span)}）**  
- 買：EMA{int(fast_span)} 上穿 EMA{int(slow_span)}；賣：下穿  
- 明日交叉臨界價：由 \( E_f' = a_f P + (1-a_f)E_f \)、\( E_s' = a_s P + (1-a_s)E_s \) 解 \( E_f' = E_s' \)

**策略 2：長線 67EMA（0.2% / 0.15%）**  
- 買：向上穿越且盤中高點 ≥ EMA×(1+0.2%)；或連續 ≥2 根、每根與 EMA 偏離 ≥0.15%  
- 賣：向下跌破且盤中低點 ≤ EMA×(1−0.2%)；或連續 ≥2 根、每根偏離 ≥0.15%  
- 預掛門檻（明日）：解 \( P \ge (1+f)E_{t+1} \) 與 \( P \le (1-f)E_{t+1} \)，其中 \( E_{t+1}=K P + (1-K)E_t \)、\(K=2/(N+1)\)

**策略 3：短線 23EMA（3% / 1% +30%停利）**  
- 買/賣規則同上，只是門檻改為 3% / 1%  
- 另外加上「固定 +30% 停利」簡化版

> 本範例僅供教學示範，非投資建議。資料來源 Yahoo Finance（yfinance），可能有延遲或調整差異。
""")

        st.success("完成 ✅")

    except Exception as e:
        st.error(f"發生錯誤：{e}")
else:
    st.info("輸入代碼與參數後，按下 **開始回測 🚀**。")
