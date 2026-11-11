#!/usr/bin/env python3
"""주봉 로그차트에 표시된 SOXL 5대 폭락 사이클의 고점/저점 지표 비교.

사용법:  python3 major_cycles.py SOXL.json

고점·저점 날짜는 차트에 표시된 날짜 부근(±10거래일)에서 장중 고가/저가로 다시 찾는다.
"""
import sys

import numpy as np
import pandas as pd

from analyze import htf_asof, indicators, load

# (차트상 고점일, 차트상 저점일)
MAJOR = [
    ("2018-03-13", "2018-12-26"),
    ("2020-01-24", "2020-03-18"),
    ("2022-01-04", "2022-10-13"),
    ("2024-07-11", "2025-04-07"),
    ("2026-06-22", "2026-11-05"),
]

pd.set_option("display.width", 250)
pd.set_option("display.max_columns", 50)


def near(df, date, col, fn, span=10):
    i = df.index.searchsorted(pd.Timestamp(date))
    w = df.iloc[max(0, i - span):i + span + 1][col]
    return getattr(w, fn)()


def main():
    df = load(sys.argv[1])
    x = indicators(df)
    c = x["close"]
    x["disp_w200"] = htf_asof(c, "W-FRI", ma=200)["disp"]
    x["disp_m60"] = htf_asof(c, "M", ma=60)["disp"]
    print(f"데이터: {c.index[0].date()} ~ {c.index[-1].date()}, 최종 종가 {c.iloc[-1]:.2f}\n")

    tops, bots, struct = [], [], []
    prev_pk = None
    for pk_s, tr_s in MAJOR:
        pk = near(df, pk_s, "high", "idxmax")
        tr = near(df, tr_s, "low", "idxmin")
        hi, lo = df.high[pk], df.low[tr]
        pi, ti = c.index.get_loc(pk), c.index.get_loc(tr)

        # --- 고점 ---
        yr = x.iloc[max(0, pi - 252):pi + 1]
        w_max_day = yr["rsi_w"].idxmax()
        m_max_day = yr["rsi_m"].idxmax()
        d200_max_day = yr["disp200"].idxmax()
        p = x.loc[pk]
        tops.append({
            "고점일": pk.date(), "장중고가": round(hi, 2),
            "직전고점후(일)": (pk - prev_pk).days if prev_pk is not None else None,
            "주RSI": p.rsi_w, "1년내 주RSI최고": yr.rsi_w.max(), "그날짜": w_max_day.date(),
            "그때종가대비 고점종가%": (c[pk] / c[w_max_day] - 1) * 100,
            "월RSI": p.rsi_m, "1년내 월RSI최고": yr.rsi_m.max(),
            "200일이격": p.disp200, "1년내 최고": yr.disp200.max(),
            "20주이격": p.disp_w20, "10월이격": p.disp_m10, "200주이격": p.disp_w200,
            "1년저점대비%": p.up_from_1y_low, "12M수익%": p.ret_12m,
        })
        prev_pk = pk

        # --- 저점 ---
        seg = x.iloc[pi:ti + 1]
        t = x.loc[tr]
        w_min_day = seg["rsi_w"].idxmin()
        bots.append({
            "저점일": tr.date(), "장중저가": round(lo, 2), "낙폭%": (lo / hi - 1) * 100, "기간(일)": (tr - pk).days,
            "일RSI": t.rsi_d, "주RSI": t.rsi_w, "구간 주RSI최저": seg.rsi_w.min(), "그날짜": w_min_day.date(),
            "월RSI": t.rsi_m, "20일이격": t.disp20, "60일이격": t.disp60, "200일이격": t.disp200,
            "20주이격": t.disp_w20, "10월이격": t.disp_m10, "200주이격": t.disp_w200, "60월이격": t.disp_m60,
            "ATR%": t.atr_pct, "거래량비": t.vol_ratio,
        })

        # --- 하락 구조 ---
        segc = c.iloc[pi:ti + 1]
        # 구간 내 최대 반등 (저점→이후 고점, 종가)
        run_min = segc.cummin()
        bounce = (segc / run_min - 1).max() * 100
        b50 = seg.index[(seg.close < seg.ma50).values]
        b200 = seg.index[(seg.close < seg.ma200).values]
        dc = seg.index[(seg.ma50 < seg.ma200).values]
        first_os = seg.index[((seg.rsi_w <= 35) & (seg.disp200 <= 60)).values]
        row = {
            "고점일": pk.date(),
            "50일선 이탈(일)": (b50[0] - pk).days if len(b50) else None,
            "그때 낙폭%": (c[b50[0]] / hi - 1) * 100 if len(b50) else None,
            "200일선 이탈(일)": (b200[0] - pk).days if len(b200) else None,
            "그때낙폭%": (c[b200[0]] / hi - 1) * 100 if len(b200) else None,
            "데드크로스(일)": (dc[0] - pk).days if len(dc) else None,
            "그때 낙폭 %": (c[dc[0]] / hi - 1) * 100 if len(dc) else None,
            "구간내 최대반등%": bounce,
            "과매도 첫신호": first_os[0].date() if len(first_os) else None,
            "신호후 저점까지%": (lo / c[first_os[0]] - 1) * 100 if len(first_os) else None,
            "신호→저점(일)": (tr - first_os[0]).days if len(first_os) else None,
        }
        struct.append(row)

    print("=== A. 5대 사이클 고점 ===")
    print(pd.DataFrame(tops).round(1).to_string(index=False), "\n")
    print("=== B. 5대 사이클 저점 ===")
    print(pd.DataFrame(bots).round(1).to_string(index=False), "\n")
    print("=== C. 하락 구조 (과매도 신호 = 주봉 RSI<=35 & 200일 이격도<=60) ===")
    print(pd.DataFrame(struct).round(1).to_string(index=False), "\n")

    # 대형 고점 공통 조건이 전체 기간에서 얼마나 자주 나왔나
    print("=== D. 공통 조건 발생 구간 (전체 기간) ===")

    def spans(name, cond, gap=60):
        d = cond.fillna(False)
        idx = d[d].index
        if not len(idx):
            print(name, ": 없음")
            return
        groups, s, last = [], idx[0], idx[0]
        for t in idx[1:]:
            if (t - last).days > gap:
                groups.append((s, last))
                s = t
            last = t
        groups.append((s, last))
        print(f"{name}  → {len(groups)}개 구간")
        for a, b in groups:
            after = df.low[b:].iloc[:252]
            before_hi = df.high[a:b].max()
            print(f"   {a.date()} ~ {b.date()}  구간고가 {before_hi:8.2f}  이후 1년내 최저 {after.min():8.2f} ({(after.min() / before_hi - 1) * 100:+.0f}%)")

    spans("고점조건: 월RSI>=70 & 200일이격>=150 & 1년저점대비>=+150%",
          (x.rsi_m >= 70) & (x.disp200 >= 150) & (x.up_from_1y_low >= 150))
    print()
    d = ((x.rsi_w <= 35) & (x.disp200 <= 60)).fillna(False)
    idx = d[d].index
    groups, s, last = [], idx[0], idx[0]
    for t in idx[1:]:
        if (t - last).days > 60:
            groups.append((s, last))
            s = t
        last = t
    groups.append((s, last))
    print(f"저점조건: 주RSI<=35 & 200일이격<=60  → {len(groups)}개 구간")
    for a, b in groups:
        lo = df.low[a:].iloc[:126].min()
        f6 = c[a:].iloc[:127]
        f12 = c[a:].iloc[:253]
        print(f"   {a.date()} ~ {b.date()}  첫신호 종가 {c[a]:7.2f}  이후 추가하락 {(lo / c[a] - 1) * 100:+.0f}%"
              f"  6M후 {(f6.iloc[-1] / c[a] - 1) * 100:+.0f}%  12M후 {(f12.iloc[-1] / c[a] - 1) * 100:+.0f}%"
              + ("  (진행중)" if len(f12) < 253 else ""))


if __name__ == "__main__":
    main()
