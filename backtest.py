# -*- coding: utf-8 -*-
# EMA 回測（三策略整合）— 互動圖 + 明日預掛（支援 30D/0.5Y/1Y/2Y；手機黑底可讀；Streamlit Cloud 可直接部署）

import io
import re
from typing import Dict, List

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
import yfinance as yf

# ==============================
# 基本設定
# ==============================
st.set_page_config(page_title="EMA 回測（三策略）", page_icon="📈", layout="wide")
st.title("📈 **EMA 回測（三策略整合）— 互動圖 + 明日預掛**")

# ==============================
# 側邊欄
# ==============================
with st.sidebar:
    st.header("參數設定")
    raw_ticker = st.text_input("股票/指數代碼（例：0050 / 2330.TW / ^TWII / NVDA）", value="0050")
    period_choice = st.selectbox("歷史期間", ["30D", "0.5Y", "1Y", "2Y"], index=2)

    strategy = st.radio(
        "策略",
        [
            "均線交叉（EMA23/67）",
            "長線 67EMA（0.2% / 0.15%）",
            "短線 23EMA（3% / 1%，+30% 停利）",
        ],
        index=0,
    )

    fast_span = st.number_input("快速 EMA 天數（短線用 23）", 5, 200, 23, step=1)
    slow_span = st.number_input("慢速 EMA 天數（長線用 67）", 5, 400, 67, step=1)

    show_triggers = st.checkbox("顯示明日『預掛』", value=True)
    run_btn = st.button("開始回測 🚀", use_container_width=True)

# ==============================
# 小工具
# ==============================
PERIOD_MAP = {"30D": "30d", "0.5Y": "6mo", "1Y": "1y", "2Y": "2y"}


def normalize_ticker(s: str) -> str:
    """台股只輸入數字時自動補 .TW；保留 ^ 指數與其它國際代碼。"""
    s = (s or "").strip()
    if not s:
        return s
    if s.startswith("^"):
        return s  # 指數
    up = s.upper()
    if re.fullmatch(r"\d{3,6}[A-Z]?", up) and not up.endswith(".TW") and not up.endswith(".TWO"):
        up = up.rstrip("T") + ".TW"
    return up


@st.cache_data(show_spinner=False, ttl=900)
def _raw_download(ticker: str, period: str) -> pd.DataFrame:
    return yf.download(
        ticker,
        period=period,
        interval="1d",
        auto_adjust=True,
        group_by="column",
        progress=False,
    ).reset_index()


@st.cache_data(show_spinner=False, ttl=900)
def download_df(ticker: str, period: str) -> pd.DataFrame:
    """下載價量資料並清欄位；若 .TW 無資料且代碼像台股數字，嘗試改 .TWO 重試一次。"""
    def _clean(df: pd.DataFrame) -> pd.DataFrame:
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]

        rename = {}
        for c in list(df.columns):
            lc = str(c).lower()
            if "date" in lc and "Date" not in df.columns: rename[c] = "Date"
            if "close" in lc and "Close" not in df.columns: rename[c] = "Close"
            if "high" in lc and "High" not in df.columns: rename[c] = "High"
            if "low" in lc and "Low" not in df.columns: rename[c] = "Low"
            if "open" in lc and "Open" not in df.columns: rename[c] = "Open"
            if "volume" in lc and "Volume" not in df.columns: rename[c] = "Volume"
        if rename: df = df.rename(columns=rename)

        need = {"Date", "Close", "High", "Low", "Open"}
        lack = need - set(df.columns)
        if df.empty or lack:
            raise RuntimeError(f"資料不足或缺欄位：{lack}，實際欄位：{list(df.columns)}")

        df["Date"] = pd.to_datetime(df["Date"])
        for c in ["Close", "High", "Low", "Open", "Volume"]:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors="coerce")
        df = df.dropna(subset=["Close", "High", "Low", "Open"]).reset_index(drop=True)
        return df

    df = _raw_download(ticker, period)
    try:
        return _clean(df)
    except Exception:
        if ticker.endswith(".TW") and re.fullmatch(r"\d{3,6}[A-Z]?\.(TW)", ticker):
            alt = ticker[:-3] + "TWO"
            df2 = _raw_download(alt, period)
            return _clean(df2)
        raise


def ema(s: pd.Series, span: int) -> pd.Series:
    return s.ewm(span=int(span), adjust=False).mean()


def next_cross_price(Ef: float, Es: float, af: float, as_: float) -> float:
    den = (af - as_)
    if abs(den) < 1e-12:
        return np.nan
    return ((1 - as_) * Es - (1 - af) * Ef) / den


def price_for_offset(E: float, span: int, pct: float, side: str) -> float:
    K = 2.0 / (span + 1.0)
    if side == "buy":
        den = 1.0 - K * (1.0 + pct)
        return ((1.0 + pct) * (1.0 - K) / den) * E if den > 0 else np.nan
    else:
        den = 1.0 - K + K * pct
        return ((1.0 - pct) * (1.0 - K) / den) * E if den > 0 else np.nan


def build_trades_cross(df: pd.DataFrame, f: int, s: int) -> pd.DataFrame:
    d = df.copy()
    d["EMAf"] = ema(d["Close"], f)
    d["EMAs"] = ema(d["Close"], s)
    sig = np.where(d["EMAf"] > d["EMAs"], 1, -1)
    cross = pd.Series(sig, index=d.index).diff().fillna(0)

    trades: List[Dict] = []
    entry = None
    for i in range(1, len(d)):
        if cross.iat[i] == 2 and entry is None:
            entry = (d["Date"].iat[i], float(d["Close"].iat[i]))
        elif cross.iat[i] == -2 and entry is not None:
            bdt, bp = entry
            sdt, sp = d["Date"].iat[i], float(d["Close"].iat[i])
            trades.append(
                {"買進日": bdt.date(), "賣出日": sdt.date(), "進價": round(bp, 2), "賣價": round(sp, 2),
                 "利潤%": round((sp - bp) / bp * 100.0, 2)}
            )
            entry = None
    if entry is not None:
        bdt, bp = entry
        trades.append({"買進日": bdt.date(), "賣出日": None, "進價": round(bp, 2), "賣價": None, "利潤%": None})
    return pd.DataFrame(trades)


def build_trades_band(
    df: pd.DataFrame,
    span: int,
    cross_pct: float,
    consec_pct: float,
    consec_n: int,
    take_profit: float | None = None,
) -> pd.DataFrame:
    d = df.copy()
    d["EMA"] = ema(d["Close"], span)
    trades: List[Dict] = []
    position = None
    up_cnt = 0
    dn_cnt = 0

    for i in range(1, len(d)):
        c = float(d["Close"].iat[i]); e = float(d["EMA"].iat[i])
        h = float(d["High"].iat[i]);  l = float(d["Low"].iat[i])
        prev_c = float(d["Close"].iat[i - 1]); prev_e = float(d["EMA"].iat[i - 1])

        up_cnt = up_cnt + 1 if (c - e) / e >= consec_pct else 0
        dn_cnt = dn_cnt + 1 if (e - c) / e >= consec_pct else 0

        buy_A  = (prev_c <= prev_e) and (c > e) and (h >= e * (1.0 + cross_pct))
        sell_A = (prev_c >= prev_e) and (c < e) and (l <= e * (1.0 - cross_pct))
        buy_B  = up_cnt >= consec_n
        sell_B = dn_cnt >= consec_n

        if position is None:
            if buy_A or buy_B:
                position = (d["Date"].iat[i], float(c))
                up_cnt = dn_cnt = 0
        else:
            bdt, bp = position
            tp_hit = (take_profit is not None) and ((c - bp) / bp >= take_profit)
            if sell_A or sell_B or tp_hit:
                sdt, sp = d["Date"].iat[i], float(c)
                trades.append(
                    {"買進日": bdt.date(), "賣出日": sdt.date(), "進價": round(bp, 2), "賣價": round(sp, 2),
                     "利潤%": round((sp - bp) / bp * 100.0, 2)}
                )
                position = None
                up_cnt = dn_cnt = 0

    if position is not None:
        bdt, bp = position
        trades.append({"買進日": bdt.date(), "賣出日": None, "進價": round(bp, 2), "賣價": None, "利潤%": None})
    return pd.DataFrame(trades)


def summarize(trades_df: pd.DataFrame) -> pd.DataFrame:
    if trades_df.empty:
        return pd.DataFrame({"項目": ["總交易數", "已平倉數", "勝率(%)", "總報酬(%)", "每筆平均(%)"],
                             "數值": [0, 0, 0.0, 0.0, 0.0]})
    closed = trades_df[trades_df["利潤%"].notna()]
    if closed.empty:
        win = total = avg = 0.0
    else:
        win = round((closed["利潤%"] > 0).mean() * 100.0, 2)
        total = round(closed["利潤%"].sum(), 2)
        avg = round(closed["利潤%"].mean(), 2)
    return pd.DataFrame({"項目": ["總交易數", "已平倉數", "勝率(%)", "總報酬(%)", "每筆平均(%)"],
                         "數值": [len(trades_df), len(closed), win, total, avg]})


def to_excel_bytes(trades_df: pd.DataFrame, summary_df: pd.DataFrame) -> bytes:
    bio = io.BytesIO()
    with pd.ExcelWriter(bio, engine="xlsxwriter") as writer:
        trades_df.to_excel(writer, index=False, sheet_name="交易明細")
        summary_df.to_excel(writer, index=False, sheet_name="摘要統計")
    bio.seek(0)
    return bio.read()


def _format_volume_numbers(v):
    try:
        v = float(v)
    except Exception:
        return "-"
    if v >= 1_000_000_000: return f"{v/1_000_000_000:.1f}B"
    if v >= 1_000_000:     return f"{v/1_000_000:.1f}M"
    if v >= 1_000:         return f"{v/1_000:.1f}K"
    return f"{v:.0f}"


def make_plot(
    df: pd.DataFrame,
    trades_df: pd.DataFrame,
    strategy: str,
    fast_span: int,
    slow_span: int,
    trigger_box: str | None = None,
    title: str = "",
) -> go.Figure:
    d = df.copy()
    d["Date"] = pd.to_datetime(d["Date"])

    # ===== 準備 EMA 與最後一筆 =====
    show_cross = "均線交叉" in strategy
    if show_cross:
        d["EMAf"] = ema(d["Close"], fast_span)
        d["EMAs"] = ema(d["Close"], slow_span)
    else:
        span = slow_span if "長線" in strategy else fast_span
        d["EMA"] = ema(d["Close"], span)

    last = d.dropna().iloc[-1]
    last_date = pd.to_datetime(last["Date"])
    last_close = float(last["Close"])
    last_open = float(last["Open"])
    last_high = float(last["High"])
    last_low = float(last["Low"])

    # ===== 子圖：上=價格(K棒/Close/EMA)，下=成交量 =====
    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True,
        row_heights=[0.73, 0.27], vertical_spacing=0.06,
        specs=[[{"secondary_y": False}], [{"secondary_y": False}]],
    )

    # K 棒（紅漲綠跌）
    fig.add_trace(
        go.Candlestick(
            x=d["Date"], open=d["Open"], high=d["High"], low=d["Low"], close=d["Close"],
            name="Candles",
            increasing_line_color="#FF3B30", increasing_fillcolor="#FF3B30",
            decreasing_line_color="#00C853", decreasing_fillcolor="#00C853",
            showlegend=True,
        ),
        row=1, col=1,
    )

    # Close 線：高對比亮黃
    fig.add_trace(
        go.Scatter(
            x=d["Date"], y=d["Close"], mode="lines", name="Close",
            line=dict(width=3, color="#FFD166"), opacity=0.98,
        ),
        row=1, col=1,
    )

    # EMA 線（強化對比）
    if show_cross:
        fig.add_trace(
            go.Scatter(
                x=d["Date"], y=d["EMAf"], mode="lines", name=f"EMA{fast_span}",
                line=dict(width=2.6, dash="dot", color="#FF1744")  # 亮紅
            ),
            row=1, col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=d["Date"], y=d["EMAs"], mode="lines", name=f"EMA{slow_span}",
                line=dict(width=2.6, dash="dash", color="#1E90FF")  # 亮藍
            ),
            row=1, col=1,
        )
    else:
        span = slow_span if "長線" in strategy else fast_span
        fig.add_trace(
            go.Scatter(
                x=d["Date"], y=d["EMA"], mode="lines", name=f"EMA{span}",
                line=dict(width=2.6, dash="dot", color="#1E90FF")
            ),
            row=1, col=1,
        )

    # ===== 成交量（紅漲綠跌）=====
    if "Volume" in d.columns:
        up = d["Close"] >= d["Open"]
        colors = np.where(up, "#FF3B30", "#00C853")
        fig.add_trace(
            go.Bar(
                x=d["Date"], y=d["Volume"], name="Volume",
                marker=dict(color=colors, line=dict(width=0)),
                hovertemplate="Vol: %{y}<extra></extra>",
            ),
            row=2, col=1,
        )

    # ===== 買賣點 =====
    if not trades_df.empty:
        closed = trades_df[trades_df["賣出日"].notna()]
        openpos = trades_df[trades_df["賣出日"].isna()]

        if not closed.empty:
            fig.add_trace(
                go.Scatter(
                    x=pd.to_datetime(closed["買進日"]), y=closed["進價"],
                    mode="markers", name="Buy",
                    marker=dict(symbol="triangle-up", size=12, color="#FFC400"),
                ),
                row=1, col=1,
            )
            fig.add_trace(
                go.Scatter(
                    x=pd.to_datetime(closed["賣出日"]), y=closed["賣價"],
                    mode="markers", name="Sell",
                    marker=dict(symbol="triangle-down", size=12, color="#00E5FF"),
                ),
                row=1, col=1,
            )
        if not openpos.empty:
            fig.add_trace(
                go.Scatter(
                    x=pd.to_datetime(openpos["買進日"]), y=openpos["進價"],
                    mode="markers", name="Open",
                    marker=dict(symbol="triangle-up", size=14, color="orange",
                                line=dict(color="black", width=1)),
                ),
                row=1, col=1,
            )

    # ===== 最後一筆標示（價格 + 參考線 + 角落數值）=====
    fig.add_hline(y=last_close, line_dash="dot", line_color="#9E9E9E",
                  line_width=1, opacity=0.7, row=1, col=1)

    fig.add_trace(
        go.Scatter(
            x=[last_date], y=[last_close],
            mode="markers+text", name="Last",
            marker=dict(size=11, color="#FFD166", line=dict(color="black", width=1)),
            text=[f"{last_close:.2f}"], textposition="top center",
            textfont=dict(color="#333", size=11),
            showlegend=False,
        ),
        row=1, col=1,
    )

    # 左下角（避免擋線）
    ema_text = ""
    if show_cross and not np.isnan(last.get("EMAf", np.nan)) and not np.isnan(last.get("EMAs", np.nan)):
        ema_text = f"EMA{fast_span}: {float(last['EMAf']):.2f}｜EMA{slow_span}: {float(last['EMAs']):.2f}"
    elif "EMA" in d.columns and not np.isnan(last.get("EMA", np.nan)):
        ema_text = f"{'67' if '長線' in strategy else str(fast_span)}EMA: {float(last['EMA']):.2f}"

    vol_text = _format_volume_numbers(last.get("Volume", np.nan))
    hi_lo = f"H:{last_high:.2f} L:{last_low:.2f} O:{last_open:.2f} C:{last_close:.2f}"

    fig.add_annotation(
        xref="paper", yref="paper", x=0.02, y=0.02, xanchor="left", yanchor="bottom",
        showarrow=False, align="left",
        font=dict(color="#111"),
        bordercolor="#ddd", borderwidth=1,
        bgcolor="rgba(255,255,255,0.90)",
        text=f"{hi_lo}<br>{ema_text}<br>Vol: {vol_text}",
    )

    # 預掛/摘要方塊（左下稍上，避免與資訊框重疊）
    if trigger_box:
        fig.add_annotation(
            xref="paper", yref="paper", x=0.02, y=0.15, showarrow=False, align="left",
            font=dict(color="#111"),
            bordercolor="#ddd", borderwidth=1,
            bgcolor="rgba(255,255,255,0.92)",
            text=trigger_box,
        )

    # ===== 版面 =====
    fig.update_layout(
        title=title,
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        margin=dict(l=10, r=10, t=60, b=10),
        xaxis_rangeslider_visible=False,
    )
    fig.update_xaxes(showspikes=True, spikemode="across", spikesnap="cursor")
    fig.update_yaxes(fixedrange=False)
    fig.update_yaxes(title_text="Price", row=1, col=1)
    fig.update_yaxes(title_text="Volume", row=2, col=1)

    return fig


# ==============================
# 主流程
# ==============================
if run_btn:
    try:
        ticker = normalize_ticker(raw_ticker)
        st.write(f"**目前代碼：** `{ticker}`")

        with st.spinner("下載資料與回測中…"):
            df = download_df(ticker, PERIOD_MAP[period_choice])

            # 依策略建交易
            if "均線交叉" in strategy:
                trades = build_trades_cross(df, int(fast_span), int(slow_span))
            elif "長線" in strategy:
                trades = build_trades_band(
                    df, span=int(slow_span), cross_pct=0.002, consec_pct=0.0015, consec_n=2, take_profit=None
                )
            else:
                trades = build_trades_band(
                    df, span=int(fast_span), cross_pct=0.03, consec_pct=0.01, consec_n=2, take_profit=0.30
                )

            summary = summarize(trades)

        # ===== 即時數值卡（圖上方）=====
        # 依策略決定顯示哪一條 EMA 的最新值
        d_for_ema = df.copy()
        if "均線交叉" in strategy:
            d_for_ema["EMA_show"] = ema(d_for_ema["Close"], int(fast_span))
            ema_label = f"EMA{int(fast_span)}"
        elif "長線" in strategy:
            d_for_ema["EMA_show"] = ema(d_for_ema["Close"], int(slow_span))
            ema_label = f"EMA{int(slow_span)}"
        else:
            d_for_ema["EMA_show"] = ema(d_for_ema["Close"], int(fast_span))
            ema_label = f"EMA{int(fast_span)}"

        last_row = d_for_ema.dropna().iloc[-1]
        last_close = float(last_row["Close"])
        last_ema = float(last_row["EMA_show"])
        last_vol = last_row["Volume"] if "Volume" in d_for_ema.columns else np.nan

        k1, k2, k3 = st.columns(3)
        k1.metric("最新收盤 Close", f"{last_close:,.2f}")
        k2.metric(f"最新 {ema_label}", f"{last_ema:,.2f}")
        k3.metric("最新成交量 Volume", f"{int(last_vol):,}" if pd.notna(last_vol) else "-")

        # ===== 視覺化 =====
        title = f"{ticker} - {strategy} 回測（{period_choice}）"
        fig = make_plot(
            df, trades,
            strategy=strategy,
            fast_span=int(fast_span),
            slow_span=int(slow_span),
            trigger_box=None,  # 預掛方塊稍後組
            title=title,
        )

        # ===== 明日預掛（中文方塊） =====
        if show_triggers:
            if "均線交叉" in strategy:
                f, s = int(fast_span), int(slow_span)
                d = df.copy()
                d["EMAf"] = ema(d["Close"], f)
                d["EMAs"] = ema(d["Close"], s)
                last = d.dropna().iloc[-1]
                Ef, Es = float(last["EMAf"]), float(last["EMAs"])
                af, as_ = 2.0 / (f + 1.0), 2.0 / (s + 1.0)
                p_cross = next_cross_price(Ef, Es, af, as_)
                if Ef > Es:
                    state = "多頭中"; tip = f"若明日收盤 ≤ {p_cross:.2f} 可能翻空。"
                elif Ef < Es:
                    state = "空頭中"; tip = f"若明日收盤 ≥ {p_cross:.2f} 可能翻多。"
                else:
                    state = "臨界點"; tip = "目前就在臨界，觀察明日收盤。"
                trig_text = "【摘要】<br>" + f"- 狀態：{state}<br>- 交叉臨界價：{p_cross:.2f}<br>- 提示：{tip}"
            elif "長線" in strategy:
                span = int(slow_span)
                d = df.copy(); d["EMA"] = ema(d["Close"], span)
                last = d.dropna().iloc[-1]; E = float(last["EMA"])
                cross_buy = price_for_offset(E, span, 0.002, "buy")
                cross_sell = price_for_offset(E, span, 0.002, "sell")
                consec_buy = price_for_offset(E, span, 0.0015, "buy")
                consec_sell = price_for_offset(E, span, 0.0015, "sell")
                state = "多頭中" if last["Close"] > E else "空頭中" if last["Close"] < E else "臨界點"
                trig_text = (
                    "【預掛(67EMA)】<br>"
                    f"- 目前：{state}<br>- 突破買 ≥ {cross_buy:.2f}<br>- 跌破賣 ≤ {cross_sell:.2f}<br>"
                    f"- 連續買 ≥ {consec_buy:.2f}<br>- 連續賣 ≤ {consec_sell:.2f}"
                )
            else:
                span = int(fast_span)
                d = df.copy(); d["EMA"] = ema(d["Close"], span)
                last = d.dropna().iloc[-1]; E = float(last["EMA"])
                cross_buy = price_for_offset(E, span, 0.03, "buy")
                cross_sell = price_for_offset(E, span, 0.03, "sell")
                consec_buy = price_for_offset(E, span, 0.01, "buy")
                consec_sell = price_for_offset(E, span, 0.01, "sell")
                state = "多頭中" if last["Close"] > E else "空頭中" if last["Close"] < E else "臨界點"
                trig_text = (
                    "【預掛(23EMA)】<br>"
                    f"- 目前：{state}<br>- 突破買 ≥ {cross_buy:.2f}<br>- 跌破賣 ≤ {cross_sell:.2f}<br>"
                    f"- 連續買 ≥ {consec_buy:.2f}<br>- 連續賣 ≤ {consec_sell:.2f}<br>- 停利：+30%（示範）"
                )

            # 放到左下角偏上，避免擋線
            fig.add_annotation(
                xref="paper", yref="paper", x=0.02, y=0.18, showarrow=False, align="left",
                font=dict(color="#111"),
                bordercolor="#ddd", borderwidth=1,
                bgcolor="rgba(255,255,255,0.92)",
                text=trig_text,
            )

        st.plotly_chart(fig, use_container_width=True, config={"displaylogo": False})

        # ===== 交易明細與摘要 =====
        c1, c2 = st.columns([2, 1])
        with c1:
            st.subheader("交易明細")
            if trades.empty:
                st.info("本次參數下尚無交易訊號。")
            else:
                st.dataframe(trades, use_container_width=True)
        with c2:
            st.subheader("摘要統計")
            st.table(summary)

        # ===== 下載 =====
        st.subheader("下載")
        xlsx = to_excel_bytes(trades, summary)
        st.download_button(
            "📄 下載 Excel（交易明細＋摘要）",
            data=xlsx,
            file_name=f"trades_{ticker.replace('.','_')}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True,
        )

        # ===== 說明 =====
        with st.expander("🧭 策略與計算說明（點我展開）", expanded=False):
            fs, ss = int(fast_span), int(slow_span)
            md = (
                f"**策略 1：EMA 交叉（{fs}/{ss}）**  \n"
                f"- 買：EMA{fs} 上穿 EMA{ss}；賣：下穿  \n"
                r"- 明日交叉臨界價：由 \( E_f' = a_f P + (1-a_f)E_f \)、\( E_s' = a_s P + (1-a_s)E_s \) 解 \( P = \frac{(1-a_s)E_s-(1-a_f)E_f}{a_f-a_s} \)"
                "\n\n"
                r"**策略 2：長線 67EMA（0.2% / 0.15%）**  \n"
                r"- 買：向上穿越且盤中高點 ≥ EMA×(1+0.2%)；或連續 ≥2 根、每根與 EMA 偏離 ≥0.15%  \n"
                r"- 賣：向下跌破且盤中低點 ≤ EMA×(1−0.2%)；或連續 ≥2 根、每根偏離 ≥0.15%  \n"
                r"- 預掛門檻（明日）：解 \( P \ge (1+f)E_{t+1} \) 與 \( P \le (1-f)E_{t+1} \)，其中 \( E_{t+1}=K P + (1-K)E_t \)、\(K=2/(N+1)\)"
                "\n\n"
                r"**策略 3：短線 23EMA（3% / 1% ＋ 30%停利）**  \n"
                r"- 買/賣規則與上同，只是門檻改為 3% / 1%，另加固定 +30% 停利（示範）。  \n"
                r"> 本範例僅供教學示範，非投資建議。資料來源：Yahoo Finance（yfinance），可能有延遲或調整差異。"
            )
            st.markdown(md)

    except Exception as e:
        st.error(f"發生錯誤：{e}")

else:
    st.info("輸入代碼與參數後，按下 **開始回測 🚀**。圖表圖例維持英文，其餘介面為繁體中文。")
