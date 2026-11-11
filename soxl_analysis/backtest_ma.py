#!/usr/bin/env python3
"""이평선 골드크로스 매수 / (다른 조합의) 데드크로스 매도 전략 백테스트.

사용법:  python3 backtest_ma.py TQQQ.json QQQ.json
         python3 backtest_ma.py SOXL.json SOXX.json

- 매수: (단기 a, 장기 b) 이평선이 골드크로스한 다음 거래일 종가에 전액 매수
- 매도: (단기 c, 장기 d) 이평선이 데드크로스한 다음 거래일 종가에 전액 매도
- 같은 날 두 신호가 겹치면 매도 우선. 현금 수익률 0, 편도 거래비용 0.1%, 세금 제외.
- 배당 포함 수정종가 사용. 200일선 준비 기간 때문에 2011-01-03 부터 평가.
- 두 번째 인자(기초지수 ETF)를 주면 "기초지수 이평선으로 신호를 내고 레버리지 ETF 를 매매"하는 변형도 계산.
"""
import itertools
import json
import sys

import numpy as np
import pandas as pd

SHORT = [5, 10, 20, 30, 50, 60]
LONG = [20, 30, 50, 60, 100, 120, 150, 200]
PAIRS = [(s, l) for s in SHORT for l in LONG if l >= 2 * s]
START, SPLIT = "2011-01-03", "2019-01-01"
COST = 0.001

pd.set_option("display.width", 250)
pd.set_option("display.max_rows", 200)


def load(path):
    r = json.load(open(path))["chart"]["result"][0]
    idx = pd.to_datetime(r["timestamp"], unit="s", utc=True).tz_convert("America/New_York").normalize().tz_localize(None)
    return pd.Series(r["indicators"]["adjclose"][0]["adjclose"], index=idx, dtype=float).dropna()


def crosses(sig):
    """각 이평선 조합의 (골드크로스 발생일, 데드크로스 발생일) 불리언 시리즈."""
    out = {}
    for s, l in PAIRS:
        up = sig.rolling(s).mean() > sig.rolling(l).mean()
        prev = up.shift(1, fill_value=False)
        valid = sig.rolling(l).mean().shift(1).notna()
        out[(s, l)] = ((up & ~prev & valid), (~up & prev & valid))
    return out


def position(buy_ev, sell_ev):
    s = pd.Series(np.nan, index=buy_ev.index)
    s[buy_ev] = 1.0
    s[sell_ev] = 0.0                        # 겹치면 매도 우선
    return s.ffill().fillna(0.0).shift(1).fillna(0.0)   # 다음 거래일 종가 체결


def stats(ret, pos):
    r = ret * pos.shift(1).fillna(0.0) - COST * pos.diff().abs().fillna(pos.iloc[0])
    eq = (1 + r).cumprod()
    yrs = (eq.index[-1] - eq.index[0]).days / 365.25
    entries = (pos.diff() > 0).sum() + (pos.iloc[0] > 0)
    return {
        "배수": eq.iloc[-1], "CAGR%": (eq.iloc[-1] ** (1 / yrs) - 1) * 100,
        "MDD%": (eq / eq.cummax() - 1).min() * 100, "보유비중%": pos.mean() * 100, "매수횟수": int(entries),
    }


def trade_stats(px, pos):
    """개별 매매의 승률과 손익."""
    ch = pos.diff().fillna(pos.iloc[0])
    ins, outs = list(px.index[ch > 0]), list(px.index[ch < 0])
    if len(outs) < len(ins):
        outs.append(px.index[-1])
    pnl = np.array([px[o] / px[i] - 1 for i, o in zip(ins, outs)]) * 100
    if not len(pnl):
        return {}
    return {"매매수": len(pnl), "승률%": (pnl > 0).mean() * 100, "평균이익%": pnl[pnl > 0].mean() if (pnl > 0).any() else 0,
            "평균손실%": pnl[pnl <= 0].mean() if (pnl <= 0).any() else 0, "최대이익%": pnl.max(), "최대손실%": pnl.min()}


def grid(px, sig, a, b):
    """a~b 구간에서 모든 (매수조합, 매도조합) 성과."""
    cr = crosses(sig)
    ret = px.pct_change().fillna(0.0)
    rows = []
    for bp, sp in itertools.product(PAIRS, PAIRS):
        pos = position(cr[bp][0], cr[sp][1])[a:b]
        rows.append({"매수": f"{bp[0]}/{bp[1]}", "매도": f"{sp[0]}/{sp[1]}", **stats(ret[a:b], pos)})
    return pd.DataFrame(rows)


def report(name, px, sig, label):
    ret = px.pct_change().fillna(0.0)
    end = px.index[-1]
    bh = stats(ret[START:], pd.Series(1.0, index=px[START:].index))
    print(f"\n######## {name} — 신호: {label}  ({START} ~ {end.date()}) ########")
    print(f"단순 보유: {bh['배수']:.1f}배, CAGR {bh['CAGR%']:.1f}%, MDD {bh['MDD%']:.0f}%")

    g = grid(px, sig, START, end)
    g["CAGR/|MDD|"] = g["CAGR%"] / g["MDD%"].abs()
    n = len(g)
    print(f"\n[전체 {n}개 조합 분포]")
    print(g[["배수", "CAGR%", "MDD%", "보유비중%", "매수횟수"]].quantile([0, .1, .25, .5, .75, .9, 1]).round(1).to_string())
    print(f"단순 보유보다 CAGR 높은 조합: {(g['CAGR%'] > bh['CAGR%']).mean() * 100:.0f}%"
          f" | MDD 가 더 얕은 조합: {(g['MDD%'] > bh['MDD%']).mean() * 100:.0f}%"
          f" | MDD -50% 이내: {(g['MDD%'] > -50).mean() * 100:.0f}%"
          f" | 둘 다(CAGR↑, MDD↓): {((g['CAGR%'] > bh['CAGR%']) & (g['MDD%'] > bh['MDD%'])).mean() * 100:.0f}%")

    print("\n[CAGR 상위 10]")
    print(g.sort_values("CAGR%", ascending=False).head(10).round(1).to_string(index=False))
    print("\n[같은 조합으로 매수·매도 (대표 조합)]")
    same = g[(g["매수"] == g["매도"]) & g["매수"].isin(["5/20", "10/50", "20/60", "20/120", "50/200", "20/200", "5/200"])]
    print(same.round(1).to_string(index=False))

    # 과최적화 검증: 전반부에서 고른 상위 조합이 후반부에도 통하는가
    g1 = grid(px, sig, START, SPLIT)
    g2 = grid(px, sig, SPLIT, end)
    bh1 = stats(ret[START:SPLIT], pd.Series(1.0, index=px[START:SPLIT].index))
    bh2 = stats(ret[SPLIT:], pd.Series(1.0, index=px[SPLIT:].index))
    top = g1["CAGR%"].rank(ascending=False) <= n * 0.1
    rank2 = g2["CAGR%"].rank(ascending=False, pct=True) * 100
    print(f"\n[과최적화 검증] 전반부({START}~{SPLIT}) 단순보유 CAGR {bh1['CAGR%']:.1f}% / MDD {bh1['MDD%']:.0f}%,"
          f" 후반부 단순보유 CAGR {bh2['CAGR%']:.1f}% / MDD {bh2['MDD%']:.0f}%")
    print(f"  전반부 CAGR 상위 10% 조합의 후반부 성과: CAGR 중앙값 {g2.loc[top, 'CAGR%'].median():.1f}%"
          f" (전체 조합 중앙값 {g2['CAGR%'].median():.1f}%), MDD 중앙값 {g2.loc[top, 'MDD%'].median():.0f}%,"
          f" 후반부 순위 중앙값 상위 {rank2[top].median():.0f}%")
    print(f"  전반부·후반부 CAGR 순위 상관계수: {g1['CAGR%'].rank().corr(g2['CAGR%'].rank()):.2f}")
    print(f"  후반부에 단순 보유를 CAGR 로 이긴 조합: {(g2['CAGR%'] > bh2['CAGR%']).mean() * 100:.0f}%")

    # 평가 종료일 민감도: 직전 고점에서 끝냈다면
    pk = px[START:].idxmax()
    gp = grid(px, sig, START, pk)
    bhp = stats(ret[START:pk], pd.Series(1.0, index=px[START:pk].index))
    print(f"\n[종료일 민감도] 최고가일({pk.date()})에 평가를 끝내면: 단순보유 CAGR {bhp['CAGR%']:.1f}%,"
          f" 조합 CAGR 중앙값 {gp['CAGR%'].median():.1f}%, 단순 보유를 이긴 조합 {(gp['CAGR%'] > bhp['CAGR%']).mean() * 100:.0f}%")

    # 개별 매매 통계 (대표 조합)
    cr = crosses(sig)
    print("\n[개별 매매 통계]")
    best = g.sort_values("CAGR%", ascending=False).iloc[0]
    combos = [((5, 20), (5, 20)), ((20, 60), (20, 60)), ((50, 200), (50, 200)),
              (tuple(map(int, best["매수"].split("/"))), tuple(map(int, best["매도"].split("/"))))]
    for bp, sp in combos:
        pos = position(cr[bp][0], cr[sp][1])[START:]
        t = trade_stats(px[START:], pos)
        print(f"  매수 {bp[0]}/{bp[1]} · 매도 {sp[0]}/{sp[1]}: " + ", ".join(f"{k} {v:.0f}" for k, v in t.items()))
    return g


def main():
    px = load(sys.argv[1])
    name = sys.argv[1].split("/")[-1].replace(".json", "")
    report(name, px, px, "자기 자신의 이평선")
    if len(sys.argv) > 2:
        base = load(sys.argv[2]).reindex(px.index).ffill()
        report(name, px, base, "기초지수 ETF 의 이평선")


if __name__ == "__main__":
    main()
