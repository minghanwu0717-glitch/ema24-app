# -*- coding: utf-8 -*-
# EMA 回測（三策略整合）— 互動圖 + 明日預掛（支援 30D/0.5Y/1Y/2Y；手機黑底可讀；可直接部署到 Streamlit Cloud）

import io
import re
from typing import Dict, List

import numpy as np
import pandas as pd
import plotly.graph_objects as go
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
    if re.fullmatch(r"\d{3,6}[A-Z]?", up) and not up.endswith(".TW"):
        up = up.rstrip("T") + ".TW"
    return up


@st.cache_data(show_spinner=False, ttl=900)
def download_df(ticker: str, period: str) -> pd.DataFrame:
    df = yf.download(
        ticker,
        period=period,
        interval="1d",
        auto_adjust=True,
        group_by="column",
        progress=False,
    ).reset_index()

    # 扁平欄名（避免 ('Close','0050.TW')）
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0] if isinstance(c, tuple) else c for c in df.columns]

    # 別名對應
    rename = {}
    for c in list(df.columns):
        lc = str(c).lower()
        if "date" in lc and "Date" not in df.columns:
            rename[c] = "Date"
        if "close" in lc and "Close" not in df.columns:
            rename[c] = "Close"
        if "high" in lc and "High" not in df.columns:
            rename[c] = "High"
        if "low" in lc and "Low" not in df.columns:
            rename[c] = "Low"
        if "open" in lc and "Open" not in df.columns:
            rename[c] = "Open"
        if "volume" in lc and "Volume" not in df.columns:
            rename[c] = "Volume"
    if rename:
        df = df.rename(columns=rename)

    need = {"Date", "Close", "High", "Low"}
    lack = need - set(df.columns)
    if df.empty or lack:
        raise RuntimeError(f"資料不足或缺欄位：{lack}，實際欄位：{list(df.columns)}")

    # 型別清理
    df["Date"] = pd.to_datetime(df["Date"])
    for c in ["Close", "High", "Low", "Open"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["Close", "High", "Low"]).reset_index(drop=True)
    return df


def ema(s: pd.Series, span: int) -> pd.Series:
    return s.ewm(span=int(span), adjust=False).mean()


def next_cross_price(Ef: float, Es: float, af: float, as_: float) -> float:
    """
    求明日收盤 P，使得明日 EMAf == EMAs：
    Ef' = af P + (1-af)Ef
    Es' = as P + (1-as)Es
    解 P = ((1-as)Es - (1-af)Ef) / (af - as)
    """
    den = (af - as_)
    if abs(den) < 1e-12:
        return np.nan
    return ((1 - as_) * Es - (1 - af) * Ef) / den


def price_for_offset(E: float, span: int, pct: float, side: str) -> float:
    """
    明日若要與 EMA 偏離指定百分比的觸價
      E' = K P + (1-K)E, K=2/(span+1)
      要求 P >= (1+pct)*E'（多）或 P <= (1-pct)*E'（空）
    推得：
      多：P = ((1+p)(1-K))/(1-K(1+p)) * E
      空：P = ((1-p)(1-K))/(1-K+Kp)   * E
    """
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
        if cross.iat[i] == 2 and entry is None:  # 上穿 → 買
            entry = (d["Date"].iat[i], float(d["Close"].iat[i]))
        elif cross.iat[i] == -2 and entry is not None:  # 下穿 → 賣
            bdt, bp = entry
            sdt, sp = d["Date"].iat[i], float(d["Close"].iat[i])
            trades.append(
                {
                    "買進日": bdt.date(),
                    "賣出日": sdt.date(),
                    "進價": round(bp, 2),
                    "賣價": round(sp, 2),
                    "利潤%": round((sp - bp) / bp * 100.0, 2),
                }
            )
            entry = None
    if entry is not None:  # 未平倉
        bdt, bp = entry
        trades.append(
            {"買進日": bdt.date(), "賣出日": None, "進價": round(bp, 2), "賣價": None, "利潤%": None}
        )
    return pd.DataFrame(trades)


def build_trades_band(
    df: pd.DataFrame,
    span: int,
    cross_pct: float,
    consec_pct: float,
    consec_n: int,
    take_profit: float | None = None,
) -> pd.DataFrame:
    """
    依「突破/跌破 cross_pct」與「連續 consec_pct、連續根數 ≥ consec_n」建構交易。
    若 take_profit 設定（如 0.30），遇到達標即平倉（短線策略用）。
    """
    d = df.copy()
    d["EMA"] = ema(d["Close"], span)
    trades: List[Dict] = []
    position = None
    up_cnt = 0
    dn_cnt = 0

    for i in range(1, len(d)):
        c = float(d["Close"].iat[i])
        e = float(d["EMA"].iat[i])
        h = float(d["High"].iat[i])
        l = float(d["Low"].iat[i])
        prev_c = float(d["Close"].iat[i - 1])
        prev_e = float(d["EMA"].iat[i - 1])

        # 連續計數
        up_cnt = up_cnt + 1 if (c - e) / e >= consec_pct else 0
        dn_cnt = dn_cnt + 1 if (e - c) / e >= consec_pct else 0

        # 規則 A：突破/跌破
        buy_A = (prev_c <= prev_e) and (c > e) and (h >= e * (1.0 + cross_pct))
        sell_A = (prev_c >= prev_e) and (c < e) and (l <= e * (1.0 - cross_pct))

        # 規則 B：連續 N 根
        buy_B = up_cnt >= consec_n
        sell_B = dn_cnt >= consec_n

        if position is None:
            if buy_A or buy_B:
                position = (d["Date"].iat[i], float(c))
                up_cnt = dn_cnt = 0
        else:
            bdt, bp = position

            # 停利判斷
            tp_hit = False
            if take_profit is not None:
                tp_hit = (c - bp) / bp >= take_profit

            if sell_A or sell_B or tp_hit:
                sdt, sp = d["Date"].iat[i], float(c)
                trades.append(
                    {
                        "買進日": bdt.date(),
                        "賣出日": sdt.date(),
                        "進價": round(bp, 2),
                        "賣價": round(sp, 2),
                        "利潤%": round((sp - bp) / bp * 100.0, 2),
                    }
                )
                position = None
                up_cnt = dn_cnt = 0

    if position is not None:
        bdt, bp = position
        trades.append(
            {"買進日": bdt.date(), "賣出日": None, "進價": round(bp, 2), "賣價": None, "利潤%": None}
        )
    return pd.DataFrame(trades)


def summarize(trades_df: pd.DataFrame) -> pd.DataFrame:
    if trades_df.empty:
        return pd.DataFrame(
            {
                "項目": ["總交易數", "已平倉數", "勝率(%)", "總報酬(%)", "每筆平均(%)"],
                "數值": [0, 0, 0.0, 0.0, 0.0],
            }
        )
    closed = trades_df[trades_df["利潤%"].notna()]
    if closed.empty:
        win = 0.0
        total = 0.0
        avg = 0.0
    else:
        win = round((closed["利潤%"] > 0).mean() * 100.0, 2)
        total = round(closed["利潤%"].sum(), 2)
        avg = round(closed["利潤%"].mean(), 2)
    return pd.DataFrame(
        {
            "項目": ["總交易數", "已平倉數", "勝率(%)", "總報酬(%)", "每筆平均(%)"],
            "數值": [len(trades_df), len(closed), win, total, avg],
        }
    )


def to_excel_bytes(trades_df: pd.DataFrame, summary_df: pd.DataFrame) -> bytes:
    bio = io.BytesIO()
    with pd.ExcelWriter(bio, engine="xlsxwriter") as writer:
        trades_df.to_excel(writer, index=False, sheet_name="交易明細")
        summary_df.to_excel(writer, index=False, sheet_name="摘要統計")
    bio.seek(0)
    return bio.read()


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

    fig = go.Figure()
    # Close
    fig.add_trace(
        go.Scatter(
            x=d["Date"],
            y=d["Close"],
            mode="lines",
            name="Close",
            line=dict(width=2),
        )
    )

    # EMA 線
    if "均線交叉" in strategy:
        d["EMAf"] = ema(d["Close"], fast_span)
        d["EMAs"] = ema(d["Close"], slow_span)
        fig.add_trace(
            go.Scatter(
                x=d["Date"],
                y=d["EMAf"],
                mode="lines",
                name=f"EMA{fast_span}",
                line=dict(width=2, dash="dot"),
            )
        )
        fig.add_trace(
            go.Scatter(
                x=d["Date"],
                y=d["EMAs"],
                mode="lines",
                name=f"EMA{slow_span}",
                line=dict(width=2, dash="dash"),
            )
        )
    else:
        span = slow_span if "長線" in strategy else fast_span
        d["EMA"] = ema(d["Close"], span)
        fig.add_trace(
            go.Scatter(
                x=d["Date"],
                y=d["EMA"],
                mode="lines",
                name=f"EMA{span}",
                line=dict(width=2, dash="dot"),
            )
        )

    # 買賣點
    if not trades_df.empty:
        closed = trades_df[trades_df["賣出日"].notna()]
        openpos = trades_df[trades_df["賣出日"].isna()]

        if not closed.empty:
            fig.add_trace(
                go.Scatter(
                    x=pd.to_datetime(closed["買進日"]),
                    y=closed["進價"],
                    mode="markers",
                    name="Buy",
                    marker=dict(symbol="triangle-up", size=12, color="green"),
                )
            )
            fig.add_trace(
                go.Scatter(
                    x=pd.to_datetime(closed["賣出日"]),
                    y=closed["賣價"],
                    mode="markers",
                    name="Sell",
                    marker=dict(symbol="triangle-down", size=12, color="red"),
                )
            )
        if not openpos.empty:
            fig.add_trace(
                go.Scatter(
                    x=pd.to_datetime(openpos["買進日"]),
                    y=openpos["進價"],
                    mode="markers",
                    name="Open",
                    marker=dict(
                        symbol="triangle-up",
                        size=14,
                        color="gold",
                        line=dict(color="black", width=1),
                    ),
                )
            )

    # 版面
    fig.update_layout(
        title=title,
        xaxis_title="Date",
        yaxis_title="Price",
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        margin=dict(l=10, r=10, t=60, b=10),
    )
    fig.update_xaxes(rangeslider_visible=False)

    # 加入左下角摘要（白底黑字，手機深色模式可讀）
    if trigger_box:
        fig.add_annotation(
            xref="paper",
            yref="paper",
            x=0.02,
            y=0.12,  # 左下
            showarrow=False,
            align="left",
            font=dict(color="#111"),
            bordercolor="#ddd",
            borderwidth=1,
            bgcolor="rgba(255,255,255,0.92)",
            text=trigger_box,
        )

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
                    df,
                    span=int(slow_span),          # 長線用慢速（預設 67）
                    cross_pct=0.002,              # 0.2%
                    consec_pct=0.0015,            # 0.15%
                    consec_n=2,
                    take_profit=None,
                )
            else:  # 短線
                trades = build_trades_band(
                    df,
                    span=int(fast_span),          # 短線用快速（預設 23）
                    cross_pct=0.03,               # 3%
                    consec_pct=0.01,              # 1%
                    consec_n=2,
                    take_profit=0.30,             # +30% 停利示範
                )

            summary = summarize(trades)

        # ===== 明日預掛（中文方塊） =====
        box_text = None
        if show_triggers:
            if "均線交叉" in strategy:
                # 狀態與臨界價
                f, s = int(fast_span), int(slow_span)
                d = df.copy()
                d["EMAf"] = ema(d["Close"], f)
                d["EMAs"] = ema(d["Close"], s)
                last = d.dropna().iloc[-1]
                Ef, Es = float(last["EMAf"]), float(last["EMAs"])
                af, as_ = 2.0 / (f + 1.0), 2.0 / (s + 1.0)
                p_cross = next_cross_price(Ef, Es, af, as_)
                if Ef > Es:
                    state = "多頭中"
                    tip = f"若明日收盤 ≤ {p_cross:.2f} 可能翻空。"
                elif Ef < Es:
                    state = "空頭中"
                    tip = f"若明日收盤 ≥ {p_cross:.2f} 可能翻多。"
                else:
                    state = "臨界點"
                    tip = "目前就在臨界，觀察明日收盤。"

                box_text = (
                    "【摘要】<br>"
                    f"- 狀態：{state}<br>"
                    f"- 交叉臨界價：{p_cross:.2f}<br>"
                    f"- 提示：{tip}"
                )

            elif "長線" in strategy:
                span = int(slow_span)
                d = df.copy()
                d["EMA"] = ema(d["Close"], span)
                last = d.dropna().iloc[-1]
                E = float(last["EMA"])

                cross_buy = price_for_offset(E, span, 0.002, "buy")
                cross_sell = price_for_offset(E, span, 0.002, "sell")
                consec_buy = price_for_offset(E, span, 0.0015, "buy")
                consec_sell = price_for_offset(E, span, 0.0015, "sell")

                # 狀態
                state = "多頭中" if last["Close"] > E else "空頭中" if last["Close"] < E else "臨界點"

                box_text = (
                    "【預掛(67EMA)】<br>"
                    f"- 目前：{state}<br>"
                    f"- 突破買 ≥ {cross_buy:.2f}<br>"
                    f"- 跌破賣 ≤ {cross_sell:.2f}<br>"
                    f"- 連續買 ≥ {consec_buy:.2f}<br>"
                    f"- 連續賣 ≤ {consec_sell:.2f}"
                )

            else:  # 短線
                span = int(fast_span)
                d = df.copy()
                d["EMA"] = ema(d["Close"], span)
                last = d.dropna().iloc[-1]
                E = float(last["EMA"])

                cross_buy = price_for_offset(E, span, 0.03, "buy")
                cross_sell = price_for_offset(E, span, 0.03, "sell")
                consec_buy = price_for_offset(E, span, 0.01, "buy")
                consec_sell = price_for_offset(E, span, 0.01, "sell")

                state = "多頭中" if last["Close"] > E else "空頭中" if last["Close"] < E else "臨界點"

                box_text = (
                    "【預掛(23EMA)】<br>"
                    f"- 目前：{state}<br>"
                    f"- 突破買 ≥ {cross_buy:.2f}<br>"
                    f"- 跌破賣 ≤ {cross_sell:.2f}<br>"
                    f"- 連續買 ≥ {consec_buy:.2f}<br>"
                    f"- 連續賣 ≤ {consec_sell:.2f}<br>"
                    "- 停利：+30%（示範）"
                )

        # ===== 視覺化 =====
        title = f"{ticker} - {strategy} 回測（{period_choice}）"
        fig = make_plot(
            df,
            trades,
            strategy=strategy,
            fast_span=int(fast_span),
            slow_span=int(slow_span),
            trigger_box=box_text,
            title=title,
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
