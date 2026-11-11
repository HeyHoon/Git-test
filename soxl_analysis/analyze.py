#!/usr/bin/env python3
"""SOXL 폭락장의 기술적 공통점 분석.

사용법:
    python3 analyze.py SOXL.json [SOXX.json]

입력은 Yahoo Finance v8 chart API 의 일봉 JSON 이다.
    https://query1.finance.yahoo.com/v8/finance/chart/SOXL?period1=1262304000&period2=<now>&interval=1d&events=div%2Csplit

폭락 정의: 종가 기준 고점 대비 -40% 이상 하락 (저점은 +50% 반등으로 확정) 하는 지그재그 구간.
주봉/월봉 지표는 "그 날짜까지의 진행 중인 봉"으로 계산해 미래 데이터를 쓰지 않는다.
"""
import json
import sys

import numpy as np
import pandas as pd

DROP = 0.40      # 폭락으로 보는 고점 대비 하락률
REBOUND = 0.50   # 저점 확정에 필요한 반등률

pd.set_option("display.width", 250)
pd.set_option("display.max_columns", 50)
pd.set_option("display.max_rows", 500)


def load(path):
    r = json.load(open(path))["chart"]["result"][0]
    q = r["indicators"]["quote"][0]
    idx = pd.to_datetime(r["timestamp"], unit="s", utc=True).tz_convert("America/New_York").normalize().tz_localize(None)
    df = pd.DataFrame({k: q[k] for k in ("open", "high", "low", "close", "volume")}, index=idx)
    return df.dropna(subset=["close"])


def wilder_rsi(close, n=14):
    d = close.diff()
    up = d.clip(lower=0).ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
    return 100 - 100 / (1 + up / dn)


def htf_asof(close, rule, n=14, ma=None):
    """상위 봉(주/월) RSI 와 이격도를 일 단위로, 진행 중인 봉 기준으로 계산."""
    period = close.index.to_period(rule)
    bars = close.groupby(period).last()                      # 완성된 봉의 종가
    d = bars.diff()
    up = d.clip(lower=0).ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
    prev = pd.DataFrame({"c": bars, "up": up, "dn": dn}).shift(1)   # 직전 완성봉 상태
    p = prev.reindex(period)
    p.index = close.index
    ch = close - p["c"]
    u = (p["up"] * (n - 1) + ch.clip(lower=0)) / n
    v = (p["dn"] * (n - 1) + (-ch.clip(upper=0))) / n
    rsi = 100 - 100 / (1 + u / v)
    out = {"rsi": rsi}
    if ma:
        s = bars.rolling(ma - 1).sum().shift(1).reindex(period)  # 직전 ma-1 개 완성봉 합
        s.index = close.index
        out["disp"] = close / ((s + close) / ma) * 100
    return pd.DataFrame(out)


def indicators(df):
    c = df["close"]
    x = pd.DataFrame(index=df.index)
    x["close"] = c
    x["rsi_d"] = wilder_rsi(c)
    w = htf_asof(c, "W-FRI", ma=20)
    m = htf_asof(c, "M", ma=10)
    x["rsi_w"], x["disp_w20"] = w["rsi"], w["disp"]
    x["rsi_m"], x["disp_m10"] = m["rsi"], m["disp"]
    for n in (20, 60, 120, 200):
        x[f"disp{n}"] = c / c.rolling(n).mean() * 100
    ma20, sd20 = c.rolling(20).mean(), c.rolling(20).std(ddof=0)
    x["bb_pctb"] = (c - (ma20 - 2 * sd20)) / (4 * sd20)
    macd = c.ewm(span=12, adjust=False).mean() - c.ewm(span=26, adjust=False).mean()
    x["macd_hist"] = macd - macd.ewm(span=9, adjust=False).mean()
    lo14, hi14 = df["low"].rolling(14).min(), df["high"].rolling(14).max()
    x["stoch_k"] = ((c - lo14) / (hi14 - lo14) * 100).rolling(3).mean()
    tr = pd.concat([df["high"] - df["low"], (df["high"] - c.shift()).abs(), (df["low"] - c.shift()).abs()], axis=1).max(axis=1)
    x["atr_pct"] = tr.ewm(alpha=1 / 14, adjust=False).mean() / c * 100
    x["vol_ratio"] = df["volume"].rolling(5).mean() / df["volume"].rolling(60).mean()
    for n, k in ((21, "1m"), (63, "3m"), (126, "6m"), (252, "12m")):
        x[f"ret_{k}"] = (c / c.shift(n) - 1) * 100
    x["up_from_1y_low"] = (c / c.rolling(252, min_periods=120).min() - 1) * 100
    x["dd_from_1y_high"] = (c / c.rolling(252, min_periods=120).max() - 1) * 100
    x["ma50"], x["ma200"] = c.rolling(50).mean(), c.rolling(200).mean()
    return x


def crashes(c):
    """지그재그: 고점 대비 -DROP 이면 폭락 시작, 저점 대비 +REBOUND 이면 종료."""
    out, mode = [], "up"
    pk_i, pk = 0, c.iloc[0]
    for i in range(1, len(c)):
        p = c.iloc[i]
        if mode == "up":
            if p > pk:
                pk_i, pk = i, p
            elif p <= pk * (1 - DROP):
                mode, tr_i, tr = "down", i, p
        else:
            if p < tr:
                tr_i, tr = i, p
            elif p >= tr * (1 + REBOUND):
                out.append((pk_i, tr_i, True))
                mode, pk_i, pk = "up", i, p
    if mode == "down":
        out.append((pk_i, tr_i, False))
    return out


def main():
    df = load(sys.argv[1])
    x = indicators(df)
    c = x["close"]
    print(f"데이터: {c.index[0].date()} ~ {c.index[-1].date()}  ({len(c)} 거래일), 최종 종가 {c.iloc[-1]:.2f}\n")

    ev = crashes(c)
    rows, peaks, troughs = [], [], []
    for pk_i, tr_i, done in ev:
        seg = x.iloc[pk_i:tr_i + 1]
        look = x.iloc[max(0, pk_i - 20):pk_i + 1]         # 고점 직전 20거래일
        below50 = seg.index[(seg["close"] < seg["ma50"]).values]
        below200 = seg.index[(seg["close"] < seg["ma200"]).values]
        dc = seg.index[((seg["disp20"] < 100) & (seg["macd_hist"] < 0)).values]
        rows.append({
            "고점일": c.index[pk_i].date(), "저점일": c.index[tr_i].date(),
            "고점": round(c.iloc[pk_i], 2), "저점": round(c.iloc[tr_i], 2),
            "하락률%": round((c.iloc[tr_i] / c.iloc[pk_i] - 1) * 100, 1),
            "기간(거래일)": tr_i - pk_i,
            "50일선이탈(일)": (c.index.get_loc(below50[0]) - pk_i) if len(below50) else None,
            "50일선이탈시 낙폭%": round((c[below50[0]] / c.iloc[pk_i] - 1) * 100, 1) if len(below50) else None,
            "200일선이탈(일)": (c.index.get_loc(below200[0]) - pk_i) if len(below200) else None,
            "확정": done,
        })
        p = x.iloc[pk_i].drop(["close", "ma50", "ma200"]).copy()
        p["max_rsi_d_20d"] = look["rsi_d"].max()
        p["max_rsi_w_20d"] = look["rsi_w"].max()
        p["max_disp20_20d"] = look["disp20"].max()
        # 하락 다이버전스: 고점 종가는 20일 내 최고가인데 일봉 RSI 는 20일 내 최고치보다 5pt 이상 낮음
        p["rsi_div"] = float(look["rsi_d"].max() - x["rsi_d"].iloc[pk_i] >= 5)
        peaks.append(p.rename(c.index[pk_i].date()))
        troughs.append(x.iloc[tr_i].drop(["close", "ma50", "ma200"]).rename(c.index[tr_i].date()))

    print("=== 1. 폭락 구간 (종가 기준 -40% 이상) ===")
    print(pd.DataFrame(rows).to_string(index=False), "\n")

    P, T = pd.DataFrame(peaks), pd.DataFrame(troughs)
    cols_p = ["rsi_d", "max_rsi_d_20d", "rsi_div", "rsi_w", "max_rsi_w_20d", "rsi_m", "disp20", "disp60", "disp120", "disp200",
              "disp_w20", "disp_m10", "bb_pctb", "stoch_k", "atr_pct", "vol_ratio", "ret_3m", "ret_6m", "ret_12m", "up_from_1y_low"]
    print("=== 2. 고점 당일 지표 ===")
    print(P[cols_p].round(1).to_string())
    print("\n중앙값 / 최소 / 최대")
    print(P[cols_p].agg(["median", "min", "max"]).round(1).to_string(), "\n")

    cols_t = ["rsi_d", "rsi_w", "rsi_m", "disp20", "disp60", "disp120", "disp200", "disp_w20", "disp_m10",
              "bb_pctb", "stoch_k", "atr_pct", "vol_ratio", "ret_1m", "ret_3m", "dd_from_1y_high"]
    print("=== 3. 저점 당일 지표 ===")
    print(T[cols_t].round(1).to_string())
    print("\n중앙값 / 최소 / 최대")
    print(T[cols_t].agg(["median", "min", "max"]).round(1).to_string(), "\n")

    print("=== 4. 전체 기간 대비 분포 (백분위) ===")
    q = x[["rsi_d", "rsi_w", "rsi_m", "disp20", "disp60", "disp200", "disp_w20", "disp_m10", "up_from_1y_low"]].quantile(
        [.05, .1, .25, .5, .75, .9, .95]).round(1)
    print(q.to_string(), "\n")

    # 5. 과열 신호의 기저율: 신호가 뜬 뒤 실제로 얼마나 자주 폭락이 왔는가
    fwd = {n: (c.shift(-n) / c - 1) * 100 for n in (21, 63, 126)}
    fmin = {n: (c[::-1].rolling(n).min()[::-1].shift(-1) / c - 1) * 100 for n in (63, 126)}
    pk_dates = [c.index[e[0]] for e in ev]

    def study(name, cond):
        cond = cond.fillna(False)
        # 신호 군집(20거래일 내 연속 발생)은 첫 날만 집계
        first = cond & ~(cond.rolling(20, min_periods=1).max().shift(1).fillna(0).astype(bool))
        d = first[first].index
        d = d[d <= c.index[-127]]
        if not len(d):
            print(f"{name}: 신호 없음")
            return
        hit = sum(any(0 <= (pk - t).days <= 90 for pk in pk_dates) for t in d)
        caught = sum(any(0 <= (pk - t).days <= 90 for t in cond[cond].index) for pk in pk_dates)
        print(f"{name}\n   신호 {len(d)}회 | 90일내 폭락고점 {hit}회({hit / len(d) * 100:.0f}%) | 폭락 {len(pk_dates)}건 중 {caught}건 사전포착"
              f" | 이후 수익률 중앙값 1M {fwd[21][d].median():+.0f}% 3M {fwd[63][d].median():+.0f}% 6M {fwd[126][d].median():+.0f}%"
              f" | 6M내 최대낙폭 중앙값 {fmin[126][d].median():.0f}% | 6M내 -40%↓ 비율 {(fmin[126][d] <= -40).mean() * 100:.0f}%")

    print("=== 5. 과열 신호 기저율 ===")
    base = (fmin[126].dropna() <= -40).mean() * 100
    print(f"(참고) 임의의 날 매수 시 6개월 내 -40% 이상 낙폭을 겪을 확률: {base:.0f}%, 6M 수익률 중앙값 {fwd[126].median():+.0f}%\n")
    study("A. 주봉 RSI >= 70", x["rsi_w"] >= 70)
    study("B. 주봉 RSI >= 75", x["rsi_w"] >= 75)
    study("C. 월봉 RSI >= 70", x["rsi_m"] >= 70)
    study("D. 200일 이격도 >= 150", x["disp200"] >= 150)
    study("E. 200일 이격도 >= 170", x["disp200"] >= 170)
    study("F. 20주 이격도 >= 140", x["disp_w20"] >= 140)
    study("G. 1년 저점 대비 +200% 이상", x["up_from_1y_low"] >= 200)
    study("H. 주봉RSI>=70 & 200일 이격도>=150", (x["rsi_w"] >= 70) & (x["disp200"] >= 150))
    study("I. 일봉RSI>=75 & 20일 이격도>=115", (x["rsi_d"] >= 75) & (x["disp20"] >= 115))
    study("J. 과열 이력(60일내 주봉RSI>=70) 후 50일선 종가 이탈",
          (x["rsi_w"].rolling(60).max() >= 70) & (c < x["ma50"]) & (c.shift(1) >= x["ma50"].shift(1)))
    print()

    print("=== 6. 과매도 신호 기저율 (저점 매수 관점) ===")

    def study_buy(name, cond):
        cond = cond.fillna(False)
        first = cond & ~(cond.rolling(20, min_periods=1).max().shift(1).fillna(0).astype(bool))
        d = first[first].index
        d = d[d <= c.index[-127]]
        if not len(d):
            print(f"{name}: 신호 없음")
            return
        print(f"{name}\n   신호 {len(d)}회 | 이후 수익률 중앙값 1M {fwd[21][d].median():+.0f}% 3M {fwd[63][d].median():+.0f}% 6M {fwd[126][d].median():+.0f}%"
              f" | 6M 승률 {(fwd[126][d] > 0).mean() * 100:.0f}% | 3M내 추가 최대낙폭 중앙값 {fmin[63][d].median():.0f}% (최악 {fmin[63][d].min():.0f}%)")

    study_buy("a. 일봉 RSI <= 30", x["rsi_d"] <= 30)
    study_buy("b. 일봉 RSI <= 25", x["rsi_d"] <= 25)
    study_buy("c. 주봉 RSI <= 35", x["rsi_w"] <= 35)
    study_buy("d. 주봉 RSI <= 30", x["rsi_w"] <= 30)
    study_buy("e. 20일 이격도 <= 80", x["disp20"] <= 80)
    study_buy("f. 200일 이격도 <= 60", x["disp200"] <= 60)
    study_buy("g. 20주 이격도 <= 65", x["disp_w20"] <= 65)
    study_buy("h. 주봉RSI<=35 & 200일 이격도<=65", (x["rsi_w"] <= 35) & (x["disp200"] <= 65))
    print()

    print("=== 7. 현재 상태 ===")
    cur = x.iloc[-1]
    for k in ["rsi_d", "rsi_w", "rsi_m", "disp20", "disp60", "disp120", "disp200", "disp_w20", "disp_m10", "bb_pctb",
              "ret_3m", "ret_6m", "ret_12m", "up_from_1y_low", "dd_from_1y_high"]:
        pct = (x[k].dropna() <= cur[k]).mean() * 100
        print(f"  {k:16s} {cur[k]:8.1f}   (역사적 백분위 {pct:3.0f}%)")
    print(f"  종가 {cur['close']:.2f} / 50일선 {cur['ma50']:.2f} / 200일선 {cur['ma200']:.2f}")

    if len(sys.argv) > 2:
        s = load(sys.argv[2])["close"]
        print("\n=== 8. 기초지수 ETF(SOXX) 같은 구간 낙폭 ===")
        for pk_i, tr_i, _ in ev:
            a, b = c.index[pk_i], c.index[tr_i]
            seg = s[a:b]
            print(f"  {a.date()} ~ {b.date()}  SOXL {(c[b] / c[a] - 1) * 100:6.1f}%   SOXX {(seg.iloc[-1] / seg.iloc[0] - 1) * 100:6.1f}%"
                  f"   고점시 SOXX 200일 이격도 {(s / s.rolling(200).mean() * 100)[a]:.0f}")


if __name__ == "__main__":
    main()
