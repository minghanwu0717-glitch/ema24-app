# -*- coding: utf-8 -*-
# 三合一：EMA23/67 交叉、長線 67EMA(0.2%/0.15%)、短線 23EMA(3%/1% +30%停利)
# 圖例(legend)為英文，其餘 UI 為繁中。摘要方塊固定左下角。

import io
import re
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf

st.set_page_config(page_title="EMA 回測（三策略）", page_icon="📈", layout="wide")

# ================== 小工具 ==================
def normalize_ticker(raw: str) -> str:
    s = raw.strip().upper()
    # 台股只輸入數字或數字+一碼字母就補 .TW
    if re.fullmatch(r"\d+[A-Z]?", s) and not s.endswith(".TW"):
        s = s + ".TW"
    return s

def period_to_dates(tag: str):
    today = datetime.today()
    if tag == "30D":
        start = today - timedelta(days=30)
    elif tag == "0.5Y":
        start = today - timedelta(days=182)
    elif tag == "1Y":
        start = today - timedelta(days=365)
    else:  # 2Y
        start = today - timedelta(days=730)
    return start, today

@st.cache_data(show_spinner=False, ttl=1200)
def download_df(ticker: str, start: datetime, end: datetime) -> pd.DataFrame:
    df = yf.download(
        ticker, start=start, end=end, interval="1d",
        auto_adjust=True, group_by="column", progress=False
    )
    # 扁平欄位 ('Close','0050.TW')->'Close'
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0] for c in df.columns]
    df = df.reset_index()

    # 欄位對應
    ren = {}
    for c in list(df.columns):
        low = str(c).lower()
        if "date" in low and "Date" not in df.columns: ren[c] = "Date"
        if "close" in low and "Close" not in df.columns: ren[c] = "Close"
        if "high" in low and "High" not in df.columns: ren[c] = "High"
        if "low"  in low and "Low"  not in df.columns: ren[c] = "Low"
        if "open" in low and "Open" not in df.columns: ren[c] = "Open"
        if "volume" in low and "Volume" not in df.columns: ren[c] = "Volume"
    if ren: df = df.rename(columns=ren)

    need = {"Date","Close","High","Low"}
    if df.empty or (need - set(df.columns)):
        raise RuntimeError(f"資料不足或缺欄位：{need - set(df.columns)}，實際欄位：{list(df.columns)}")
    return df

# ================== 共用計算 ==================
def summarize(tr: pd.DataFrame) -> pd.DataFrame:
    if tr.empty:
        return pd.DataFrame({"項目":["交易次數","勝率(%)","總報酬(%)","平均報酬(%)","平均持有(天)"],
                             "數值":[0,0.0,0.0,0.0,0.0]})
    wins = (tr["報酬率(%)"]>0).mean()*100 if "報酬率(%)" in tr else 0.0
    avgd = 0.0
    if "出場日" in tr and "進場日" in tr and tr["出場日"].notna().any():
        avgd = (pd.to_datetime(tr["出場日"]) - pd.to_datetime(tr["進場日"])).dt.days.mean()
    total = tr["報酬率(%)"].sum() if "報酬率(%)" in tr else 0.0
    avg = tr["報酬率(%)"].mean() if "報酬率(%)" in tr else 0.0
    return pd.DataFrame({"項目":["交易次數","勝率(%)","總報酬(%)","平均報酬(%)","平均持有(天)"],
                         "數值":[len(tr), round(wins,2), round(total,2), round(avg,2), round(float(avgd),1)]})

def to_excel_bytes(trades_df: pd.DataFrame, summary_df: pd.DataFrame) -> bytes:
    bio = io.BytesIO()
    with pd.ExcelWriter(bio, engine="xlsxwriter") as w:
        trades_df.to_excel(w, index=False, sheet_name="交易明細")
        summary_df.to_excel(w, index=False, sheet_name="摘要統計")
    bio.seek(0)
    return bio.read()

def add_trade_markers(fig: go.Figure, trades_df: pd.DataFrame):
    if trades_df.empty:
        return
    closed = trades_df[trades_df["出場日"].notna()].copy()
    openpos = trades_df[trades_df["出場日"].isna()].copy()
    if not closed.empty:
        fig.add_trace(go.Scatter(
            x=pd.to_datetime(closed["進場日"]), y=closed["進場價"],
            mode="markers", name="Buy",
            marker=dict(symbol="triangle-up", size=12, color="green")
        ))
        fig.add_trace(go.Scatter(
            x=pd.to_datetime(closed["出場日"]), y=closed["出場價"],
            mode="markers", name="Sell",
            marker=dict(symbol="triangle-down", size=12, color="red")
        ))
    if not openpos.empty:
        fig.add_trace(go.Scatter(
            x=pd.to_datetime(openpos["進場日"]), y=openpos["進場價"],
            mode="markers", name="Open",
            marker=dict(symbol="triangle-up", size=14, color="gold",
                        line=dict(color="black", width=1))
        ))

# ================== 策略 ①：EMA 交叉 (23/67) ==================
def add_ema_and_signals(df: pd.DataFrame, fast=23, slow=67):
    d = df.copy()
    d["Date"] = pd.to_datetime(d["Date"])
    d["EMA23"] = d["Close"].ewm(span=fast, adjust=False).mean()
    d["EMA67"] = d["Close"].ewm(span=slow, adjust=False).mean()
    d["Signal"] = np.where(d["EMA23"] > d["EMA67"], 1, -1)   # 多/空
    d["Cross"]  = d["Signal"].diff()                         # +2/-2
    return d

def build_trades_cross(d: pd.DataFrame) -> pd.DataFrame:
    trades = []
    entry_px = entry_dt = None
    for i in range(len(d)):
        r = d.iloc[i]
        if r["Cross"] == 2:              # 買
            entry_px = float(r["Close"]); entry_dt = r["Date"].date()
        elif r["Cross"] == -2 and entry_px is not None:  # 賣
            exit_px = float(r["Close"]); exit_dt = r["Date"].date()
            ret = (exit_px - entry_px) / entry_px * 100
            trades.append({"進場日": entry_dt, "進場價": round(entry_px,2),
                           "出場日": exit_dt, "出場價": round(exit_px,2),
                           "報酬率(%)": round(ret,2)})
            entry_px = entry_dt = None
    return pd.DataFrame(trades)

def trigger_cross_critical_price(E23, E67, fast=23, slow=67):
    """反解明日收盤使 EMA23_next == EMA67_next 的臨界價"""
    a23 = 2/(fast+1); a67 = 2/(slow+1)
    den = (a23 - a67)
    if den == 0: return None
    return float(((1-a67)*E67 - (1-a23)*E23) / den)

# ================== 策略 ②：長線 67EMA (0.2% / 0.15%) ==================
def backtest_long_67(df: pd.DataFrame,
                     span=67, cross_eps=0.002, consec_eps=0.0015, consec_n=2):
    d = df.copy()
    d["Date"] = pd.to_datetime(d["Date"])
    d["EMA"]  = d["Close"].ewm(span=span, adjust=False).mean()

    trades = []
    pos_px = pos_dt = None
    above_cnt = below_cnt = 0

    for i in range(1, len(d)):
        pc, pe = float(d["Close"].iat[i-1]), float(d["EMA"].iat[i-1])
        cc, ce = float(d["Close"].iat[i]),   float(d["EMA"].iat[i])
        ch = float(d.get("High", d["Close"]).iat[i])
        cl = float(d.get("Low",  d["Close"]).iat[i])
        dt = d["Date"].iat[i].date()

        above_cnt = above_cnt + 1 if (cc - ce)/ce >  consec_eps else 0
        below_cnt = below_cnt + 1 if (ce - cc)/ce >  consec_eps else 0

        buy_A  = (pc <= pe) and (cc > ce) and (ch >= ce * (1 + cross_eps))
        sell_A = (pc >= pe) and (cc < ce) and (cl <= ce * (1 - cross_eps))
        buy_B  = (above_cnt >= consec_n)
        sell_B = (below_cnt >= consec_n)

        if pos_px is None:
            if buy_A or buy_B:
                pos_px, pos_dt = cc, dt
                above_cnt = below_cnt = 0
        else:
            if sell_A or sell_B:
                ret = (cc - pos_px) / pos_px * 100
                trades.append({"進場日": pos_dt, "進場價": round(pos_px,2),
                               "出場日": dt,   "出場價": round(cc,2),
                               "報酬率(%)": round(ret,2)})
                pos_px = pos_dt = None
                above_cnt = below_cnt = 0

    # 可選：保留未平倉
    if pos_px is not None:
        trades.append({"進場日": pos_dt, "進場價": round(pos_px,2),
                       "出場日": None, "出場價": None, "報酬率(%)": None})
    return d, pd.DataFrame(trades)

# ================== 策略 ③：短線 23EMA (3% / 1% +30% 停利) ==================
def backtest_short_23(df: pd.DataFrame,
                      span=23, cross_eps=0.03, consec_eps=0.01, consec_n=2, take_profit=0.30):
    d = df.copy()
    d["Date"] = pd.to_datetime(d["Date"])
    d["EMA"]  = d["Close"].ewm(span=span, adjust=False).mean()

    trades = []
    pos_px = pos_dt = None
    above_cnt = below_cnt = 0

    for i in range(1, len(d)):
        pc, pe = float(d["Close"].iat[i-1]), float(d["EMA"].iat[i-1])
        cc, ce = float(d["Close"].iat[i]),   float(d["EMA"].iat[i])
        ch = float(d.get("High", d["Close"]).iat[i])
        cl = float(d.get("Low",  d["Close"]).iat[i])
        dt = d["Date"].iat[i].date()

        above_cnt = above_cnt + 1 if (cc - ce)/ce >  consec_eps else 0
        below_cnt = below_cnt + 1 if (ce - cc)/ce >  consec_eps else 0

        buy_A  = (pc <= pe) and (cc > ce) and (ch >= ce * (1 + cross_eps))
        sell_A = (pc >= pe) and (cc < ce) and (cl <= ce * (1 - cross_eps))
        buy_B  = (above_cnt >= consec_n)
        sell_B = (below_cnt >= consec_n)

        if pos_px is None:
            if buy_A or buy_B:
                pos_px, pos_dt = cc, dt
                above_cnt = below_cnt = 0
        else:
            # 停利 +30%
            if (cc - pos_px) / pos_px >= take_profit:
                ret = (cc - pos_px) / pos_px * 100
                trades.append({"進場日": pos_dt, "進場價": round(pos_px,2),
                               "出場日": dt,   "出場價": round(cc,2),
                               "報酬率(%)": round(ret,2)})
                pos_px = pos_dt = None
                above_cnt = below_cnt = 0
                continue
            if sell_A or sell_B:
                ret = (cc - pos_px) / pos_px * 100
                trades.append({"進場日": pos_dt, "進場價": round(pos_px,2),
                               "出場日": dt,   "出場價": round(cc,2),
                               "報酬率(%)": round(ret,2)})
                pos_px = pos_dt = None
                above_cnt = below_cnt = 0

    if pos_px is not None:
        trades.append({"進場日": pos_dt, "進場價": round(pos_px,2),
                       "出場日": None, "出場價": None, "報酬率(%)": None})
    return d, pd.DataFrame(trades)

# ================== 預掛（單一 EMA 規則） ==================
def compute_nextday_triggers(today_close: float, today_ema: float, ema_span: int,
                              cross_filter: float, consec_filter: float, consec_n: int,
                              streak_up: int, streak_dn: int):
    """
    反解明日「達標價」：
      - 突破/跌破：±cross_filter (例 0.2% / 3%)
      - 連續：≥ consec_n 根、每根 ≥ consec_filter (例 0.15% / 1%)
    """
    K = 2.0 / (ema_span + 1.0)
    E = float(today_ema)

    # 突破/跌破達標價（只用今日 EMA 估，明日一價即成立）
    den_buy = 1.0 - K * (1.0 + cross_filter)
    den_sel = 1.0 - K + K * cross_filter
    crossover_buy  = ((1.0 + cross_filter) * (1.0 - K) / den_buy) * E if den_buy > 0 else np.nan
    crossover_sell = ((1.0 - cross_filter) * (1.0 - K) / den_sel) * E if den_sel > 0 else np.nan

    # 連續達標（需今天已連續 consec_n-1 根）
    consec_buy = consec_sell = np.nan
    if streak_up >= max(0, consec_n - 1):
        den_cb = 1.0 - K * (1.0 + consec_filter)
        consec_buy = ((1.0 + consec_filter) * (1.0 - K) / den_cb) * E if den_cb > 0 else np.nan
    if streak_dn >= max(0, consec_n - 1):
        den_cs = 1.0 - K + K * consec_filter
        consec_sell = ((1.0 - consec_filter) * (1.0 - K) / den_cs) * E if den_cs > 0 else np.nan

    return dict(
        crossover_buy = None if not np.isfinite(crossover_buy) else round(float(crossover_buy),2),
        crossover_sell= None if not np.isfinite(crossover_sell) else round(float(crossover_sell),2),
        consec_buy    = None if not np.isfinite(consec_buy) else round(float(consec_buy),2),
        consec_sell   = None if not np.isfinite(consec_sell) else round(float(consec_sell),2),
        prereq_buy_ok  = (today_close <= today_ema),
        prereq_sell_ok = (today_close >= today_ema)
    )

def current_streaks(d: pd.DataFrame, ema_col="EMA", consec_eps=0.01):
    """回傳最後一日已連續的 up/dn 根數"""
    up = dn = 0
    for i in range(len(d)):
        c = float(d["Close"].iat[i]); e = float(d[ema_col].iat[i])
        if (c - e)/e > consec_eps:
            up += 1; dn = 0
        elif (e - c)/e > consec_eps:
            dn += 1; up = 0
        else:
            up = dn = 0
    return up, dn

# ================== 畫圖 ==================
def make_figure_cross(d: pd.DataFrame, title: str, trades: pd.DataFrame, info_box: str|None):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=d["Date"], y=d["Close"], mode="lines", name="Close"))
    fig.add_trace(go.Scatter(x=d["Date"], y=d["EMA23"], mode="lines", name="EMA23",
                             line=dict(dash="dot")))
    fig.add_trace(go.Scatter(x=d["Date"], y=d["EMA67"], mode="lines", name="EMA67",
                             line=dict(dash="dash")))
    add_trade_markers(fig, trades)
    fig.update_layout(title=title, hovermode="x unified",
                      legend=dict(orientation="h", y=1.03, x=0),
                      xaxis_title="Date", yaxis_title="Price",
                      margin=dict(l=10, r=10, t=60, b=80))
    if info_box:
        fig.add_annotation(xref="paper", yref="paper",
                           x=0.01, y=0.02, xanchor="left", yanchor="bottom",
                           showarrow=False, align="left",
                           bordercolor="#ccc", borderwidth=1,
                           bgcolor="rgba(255,255,255,0.85)", text=info_box)
    return fig

def make_figure_single_ema(d: pd.DataFrame, ema_span: int, title: str,
                           trades: pd.DataFrame, info_box: str|None):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=d["Date"], y=d["Close"], mode="lines", name="Close"))
    fig.add_trace(go.Scatter(x=d["Date"], y=d["EMA"], mode="lines", name=f"EMA{ema_span}",
                             line=dict(dash="dot")))
    add_trade_markers(fig, trades)
    fig.update_layout(title=title, hovermode="x unified",
                      legend=dict(orientation="h", y=1.03, x=0),
                      xaxis_title="Date", yaxis_title="Price",
                      margin=dict(l=10, r=10, t=60, b=80))
    if info_box:
        fig.add_annotation(xref="paper", yref="paper",
                           x=0.01, y=0.02, xanchor="left", yanchor="bottom",
                           showarrow=False, align="left",
                           bordercolor="#ccc", borderwidth=1,
                           bgcolor="rgba(255,255,255,0.85)", text=info_box)
    return fig

# ================== UI ==================
st.title("📈 EMA 回測（三策略整合）— 互動圖＋明日預掛")

with st.sidebar:
    st.header("設定")
    raw = st.text_input("股票/指數代碼（^TWII / 0050.TW / 2330.TW / NVDA）", value="^TWII")
    period_tag = st.selectbox("歷史期間", ["30D","0.5Y","1Y","2Y"], index=2)
    strategy = st.radio("策略", ["均線交叉(EMA23/67)", "長線 67EMA (0.2%/0.15%)", "短線 23EMA (3%/1% +30%停利)"])

    if strategy == "均線交叉(EMA23/67)":
        span_fast = st.number_input("快速 EMA 天數", 5, 100, 23, 1)
        span_slow = st.number_input("慢速 EMA 天數", 10, 300, 67, 1)
    elif strategy.startswith("長線"):
        span_fast = None
        span_slow = None
    else:
        span_fast = None
        span_slow = None

    show_trigger = st.checkbox("顯示明日『預掛』", value=True)
    run_btn = st.button("開始回測 🚀", use_container_width=True)

if run_btn:
    try:
        ticker = normalize_ticker(raw)
        start, end = period_to_dates(period_tag)
        st.write(f"**目前代碼：** `{ticker}`")

        with st.spinner("下載資料…"):
            df = download_df(ticker, start, end)
            if df.empty and ticker.upper()=="^TWII":
                df = download_df("0050.TW", start, end)
                st.info("`^TWII` 下載失敗，已自動改用 `0050.TW`。")

        # ======== 依策略回測 ========
        if strategy == "均線交叉(EMA23/67)":
            d = add_ema_and_signals(df, int(span_fast), int(span_slow))
            trades = build_trades_cross(d)
            summary = summarize(trades)

            # info box：交叉臨界價
            info = None
            if show_trigger and d[["EMA23","EMA67"]].notna().all(axis=None):
                last = d.iloc[-1]
                crit = trigger_cross_critical_price(last["EMA23"], last["EMA67"],
                                                    int(span_fast), int(span_slow))
                state = "多頭中" if last["EMA23"] > last["EMA67"] else "空頭中" if last["EMA23"] < last["EMA67"] else "臨界點"
                tip = (f"若明日收盤 ≤ {crit:.2f} 可能翻空" if state=="多頭中"
                       else f"若明日收盤 ≥ {crit:.2f} 可能翻多" if state=="空頭中"
                       else "觀察明日收盤變化")
                info = (f"【摘要】<br>"
                        f"- 狀態：{state}<br>"
                        f"- 交叉臨界價：{None if crit is None else round(crit,2)}<br>"
                        f"- 提示：{tip}")

            title = f"{ticker} - EMA{int(span_fast)}/EMA{int(span_slow)} Strategy ({period_tag})"
            fig = make_figure_cross(d, title, trades, info)

        elif strategy.startswith("長線"):
            # 67EMA 0.2% / 0.15% / N=2
            d, trades = backtest_long_67(df, span=67, cross_eps=0.002, consec_eps=0.0015, consec_n=2)
            summary = summarize(trades)

            info = None
            if show_trigger and d["EMA"].notna().all():
                up, dn = current_streaks(d, ema_col="EMA", consec_eps=0.0015)
                last = d.iloc[-1]
                trig = compute_nextday_triggers(
                    today_close=float(last["Close"]),
                    today_ema=float(last["EMA"]),
                    ema_span=67,
                    cross_filter=0.002,
                    consec_filter=0.0015,
                    consec_n=2,
                    streak_up=up, streak_dn=dn
                )
                info = (f"【預掛(67EMA)】<br>"
                        f"- 突破買 ≥ {trig['crossover_buy']}<br>"
                        f"- 跌破賣 ≤ {trig['crossover_sell']}<br>"
                        f"- 連續買 ≥ {trig['consec_buy']}（需已連續）<br>"
                        f"- 連續賣 ≤ {trig['consec_sell']}（需已連續）")

            title = f"{ticker} - EMA67 Strategy (0.2% / 0.15%, {period_tag})"
            fig = make_figure_single_ema(d, 67, title, trades, info)

        else:  # 短線 23EMA
            d, trades = backtest_short_23(df, span=23, cross_eps=0.03, consec_eps=0.01, consec_n=2, take_profit=0.30)
            summary = summarize(trades)

            info = None
            if show_trigger and d["EMA"].notna().all():
                up, dn = current_streaks(d, ema_col="EMA", consec_eps=0.01)
                last = d.iloc[-1]
                trig = compute_nextday_triggers(
                    today_close=float(last["Close"]),
                    today_ema=float(last["EMA"]),
                    ema_span=23,
                    cross_filter=0.03,
                    consec_filter=0.01,
                    consec_n=2,
                    streak_up=up, streak_dn=dn
                )
                info = (f"【預掛(23EMA)】<br>"
                        f"- 突破買 ≥ {trig['crossover_buy']}<br>"
                        f"- 跌破賣 ≤ {trig['crossover_sell']}<br>"
                        f"- 連續買 ≥ {trig['consec_buy']}（需已連續）<br>"
                        f"- 連續賣 ≤ {trig['consec_sell']}（需已連續）")

            title = f"{ticker} - EMA23 Strategy (3% / 1% + TP30%, {period_tag})"
            fig = make_figure_single_ema(d, 23, title, trades, info)

        # ======== 顯示表格/圖表/下載 ========
        c1, c2 = st.columns([2,1])
        with c1:
            st.subheader("交易明細")
            if trades.empty:
                st.info("本次參數下尚無完成的交易。")
            st.dataframe(trades, use_container_width=True)
        with c2:
            st.subheader("摘要統計")
            st.table(summary)

        st.subheader("走勢圖（手機可雙指縮放）")
        st.plotly_chart(fig, use_container_width=True, config={"displaylogo": False})

        st.subheader("下載")
        xbytes = to_excel_bytes(trades, summary)
        st.download_button(
            "📄 下載 Excel（交易明細＋摘要）",
            data=xbytes,
            file_name=f"trades_{ticker.replace('.','_')}_{strategy.split()[0]}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True
        )

        with st.expander("🧭 策略與計算說明", expanded=False):
            st.markdown(r"""
**三策略：**

1) **均線交叉(EMA23/67)**  
   - 買：EMA23 由下穿越 EMA67（黃金交叉）  
   - 賣：EMA23 由上跌破 EMA67（死亡交叉）  
   - 預掛：反解明日收盤，使 EMA23_next == EMA67_next 的臨界價

2) **長線 67EMA（0.2% / 0.15%）**  
   - 買：昨收在下且今上穿，且當日**高點 ≥ EMA ×(1+0.2%)**；或 **連續 ≥2 根** 每根 ≥0.15%  
   - 賣：昨收在上且今下破，且當日**低點 ≤ EMA ×(1−0.2%)**；或 **連續 ≥2 根** 每根 ≤−0.15%  

3) **短線 23EMA（3% / 1%；停利 +30%）**  
   - 規則同上，把 0.2%/0.15% 改為 3%/1%；加上**+30% 停利**  
   - 出場後若再達進場條件即可再進場

> 教學示範，非投資建議；實務尚需考量滑價、成本、跳空與流動性。
""")
    except Exception as e:
        st.error(f"發生錯誤：{e}")
else:
    st.info("輸入代碼、選期間與策略後，按 **開始回測 🚀**。")
