"""Takas ve serbest oyuncu analizi.

İki farklı bakış birleştirilir:
  - Bizim model: oyuncunun SENİN kategori ihtiyacına katkısı (sezonun kalanında
    haftalık beklenen puan farkı, kalan rakiplerin kadrolarına karşı)
  - Arkadaşın bakışı: insanlar takası çoğunlukla PIR/isim üzerinden tartar.
    Takas, arkadaşın gözünde aldığı PIR >= verdiği PIR x istek katsayısı ise
    "kabul edilebilir" sayılır. Kerem ile hiç takas yok; Gürdal ve Ege fazlasını ister.

Kazançlı takas = aynı "PIR değeri" karşılığında sana daha çok kategori kazandıran oyuncu.
"""
from __future__ import annotations

from itertools import combinations

import numpy as np
import pandas as pd
from scipy.stats import norm

CATS = ["pts", "reb", "ast", "pir"]


class TeamArrays:
    """Bir kadronun beklenen katkı ve varyans dizileri (hızlı hesap için)."""

    def __init__(self, proj: pd.DataFrame, ids: list, league_avg: dict):
        p = proj.loc[ids, "p_play"].to_numpy()
        self.ids = list(ids)
        self.ex = np.stack([p * proj.loc[ids, f"{c}_mu"].to_numpy() for c in CATS], 1)
        sd = np.stack([proj.loc[ids, f"{c}_sd"].to_numpy() for c in CATS], 1)
        mu = np.stack([proj.loc[ids, f"{c}_mu"].to_numpy() for c in CATS], 1)
        self.vx = p[:, None] * (sd ** 2 + mu ** 2) - self.ex ** 2
        self.val = sum(self.ex[:, j] / league_avg[c] for j, c in enumerate(["pts", "reb", "ast"]))


def lineup_weights(val: np.ndarray) -> np.ndarray:
    order = np.argsort(-val)
    w = np.ones(len(val))
    w[order[:5]] = 2
    w[order[0]] = 3
    w[order[-1]] = 0
    return w


def moments(ex: np.ndarray, vx: np.ndarray, val: np.ndarray):
    w = lineup_weights(val)
    return (w[:, None] * ex).sum(0), (w[:, None] ** 2 * vx).sum(0)


def exp_points(m_a, v_a, m_b, v_b) -> float:
    z = (m_a[:3] - m_b[:3]) / np.sqrt(v_a[:3] + v_b[:3] + 1e-9)
    return float(norm.cdf(z).sum())


def remaining_opponents(league: dict, team: str, from_week: int, to_week: int = 31) -> dict:
    cnt = {}
    for w in range(from_week, to_week + 1):
        for a, b in league["fixtures"].get(str(w), []):
            if team in (a, b):
                o = b if a == team else a
                cnt[o] = cnt.get(o, 0) + 1
    return cnt


def season_value(team_arr: tuple, opp_moments: dict, opp_counts: dict) -> float:
    """Kalan haftalarda toplam beklenen kategori puanı."""
    m, v = moments(*team_arr)
    return sum(n * exp_points(m, v, *opp_moments[o]) for o, n in opp_counts.items())


def find_trades(proj: pd.DataFrame, rosters_ids: dict, league: dict, me: str, week: int,
                league_avg: dict, max_size: int = 2, top_n: int = 10) -> pd.DataFrame:
    hf = league.get("human_factors", {})
    blocked = set(hf.get("no_trade_partners", []))
    premium = hf.get("premium_factor", {})
    untradeable = {p for lst in hf.get("untradeable_players", {}).values() for p in lst}

    arrays = {t: TeamArrays(proj, ids, league_avg) for t, ids in rosters_ids.items()}
    opp_mom = {t: moments(a.ex, a.vx, a.val) for t, a in arrays.items()}
    pir = proj["pir_mu"] * proj["p_play"]

    def team_value(team, ex, vx, val):
        cnts = remaining_opponents(league, team, week)
        return season_value((ex, vx, val), opp_mom, cnts)

    me_a = arrays[me]
    base_me = team_value(me, me_a.ex, me_a.vx, me_a.val)
    rows = []
    for partner, pa in arrays.items():
        if partner == me or partner in blocked:
            continue
        need = premium.get(partner, 1.0)
        base_p = team_value(partner, pa.ex, pa.vx, pa.val)
        for k in range(1, max_size + 1):
            my_sets = [s for s in combinations(range(len(me_a.ids)), k)
                       if not any(proj.loc[me_a.ids[i], "name_fantasy"] in untradeable for i in s)]
            their_sets = [s for s in combinations(range(len(pa.ids)), k)
                          if not any(proj.loc[pa.ids[i], "name_fantasy"] in untradeable for i in s)]
            my_pir = {s: pir.loc[[me_a.ids[i] for i in s]].sum() for s in my_sets}
            th_pir = {s: pir.loc[[pa.ids[i] for i in s]].sum() for s in their_sets}
            for gs in my_sets:
                for rs in their_sets:
                    # arkadaşın gözünden kabul: aldığı PIR >= verdiği PIR x istek katsayısı
                    if my_pir[gs] < th_pir[rs] * need:
                        continue
                    # bize de PIR'de çok kaybettirmeyen (bariz soygun olmayan) takaslar
                    if my_pir[gs] > th_pir[rs] * need * 1.25 + 2:
                        continue
                    ex = me_a.ex.copy(); vx = me_a.vx.copy(); val = me_a.val.copy()
                    pex = pa.ex.copy(); pvx = pa.vx.copy(); pval = pa.val.copy()
                    for i, j in zip(gs, rs):
                        ex[i], pex[j] = pa.ex[j], me_a.ex[i]
                        vx[i], pvx[j] = pa.vx[j], me_a.vx[i]
                        val[i], pval[j] = pa.val[j], me_a.val[i]
                    gain = team_value(me, ex, vx, val) - base_me
                    if gain <= 0.05:
                        continue
                    pgain = team_value(partner, pex, pvx, pval) - base_p
                    rows.append({
                        "partner": partner,
                        "give": ", ".join(proj.loc[me_a.ids[i], "name_fantasy"] for i in gs),
                        "get": ", ".join(proj.loc[pa.ids[j], "name_fantasy"] for j in rs),
                        "my_gain_season_pts": round(gain, 2),
                        "partner_model_gain": round(pgain, 2),
                        "give_pir": round(my_pir[gs], 1), "get_pir": round(th_pir[rs], 1),
                    })
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    return df.sort_values("my_gain_season_pts", ascending=False).head(top_n).reset_index(drop=True)


def free_agents(proj: pd.DataFrame, rosters_ids: dict, league: dict, me: str, week: int,
                league_avg: dict, top_n: int = 10, active_ids: dict | None = None) -> pd.DataFrame:
    owned = {p for ids in rosters_ids.values() for p in ids}
    fa = proj[(~proj.index.isin(owned)) & (proj["gp"] >= 2)]
    rosters_ids = active_ids or rosters_ids
    arrays = {t: TeamArrays(proj, ids, league_avg) for t, ids in rosters_ids.items()}
    opp_mom = {t: moments(a.ex, a.vx, a.val) for t, a in arrays.items()}
    me_a = arrays[me]
    cnts = remaining_opponents(league, me, week)
    base = season_value((me_a.ex, me_a.vx, me_a.val), opp_mom, cnts)
    rows = []
    for pid in fa.index:
        fa_a = TeamArrays(proj, [pid], league_avg)
        best = (None, 0.0)
        for i in range(len(me_a.ids)):
            ex = me_a.ex.copy(); vx = me_a.vx.copy(); val = me_a.val.copy()
            ex[i], vx[i], val[i] = fa_a.ex[0], fa_a.vx[0], fa_a.val[0]
            g = season_value((ex, vx, val), opp_mom, cnts) - base
            if g > best[1]:
                best = (proj.loc[me_a.ids[i], "name_fantasy"], g)
        if best[0]:
            rows.append({"player": proj.loc[pid, "name_fantasy"], "team": proj.loc[pid, "team"],
                         "drop": best[0], "my_gain_season_pts": round(best[1], 2),
                         "pts": round(proj.loc[pid, "pts_mu"], 1), "reb": round(proj.loc[pid, "reb_mu"], 1),
                         "ast": round(proj.loc[pid, "ast_mu"], 1), "min": round(proj.loc[pid, "min_proj"], 1)})
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    return df.sort_values("my_gain_season_pts", ascending=False).head(top_n).reset_index(drop=True)
