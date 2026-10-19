"""Tahmin motoru: her oyuncu için belirli bir EuroLeague turunda
PTS / REB / AST / PIR dağılımı üretir.

Yaklaşım (az veriye dayanıklı, açıklanabilir):
  1. Oynama olasılığı  : son takım maçlarında forma giyme oranı (yakın maç ağırlıklı)
  2. Dakika            : oynadığı maçlarda üssel ağırlıklı ortalama + geçen sezonla harmanlama
  3. Dakika başı üretim : bu sezon + geçen sezon (Bayes küçültmesiyle)
  4. Rakip etkisi      : rakibin kategori bazında yedirdiği / lig ortalaması
  5. Saha              : iç sahada küçük bonus
  6. Varyans           : kategori bazında oyuncu geçmişinden ölçülen dağılım
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

CATS = ["pts", "reb", "ast", "pir"]
# Dakika başı oran için geçen sezon ağırlığı (dakika cinsinden "sahte" veri)
PRIOR_MIN = 160.0
LEAGUE_PRIOR_MIN = 60.0
OPP_SHRINK_GAMES = 5.0
HOME_BONUS = {"pts": 1.02, "reb": 1.01, "ast": 1.02, "pir": 1.03}
# varyasyon katsayısı tabanları (oyuncu verisi azken)
BASE_CV = {"pts": 0.38, "reb": 0.45, "ast": 0.55, "pir": 0.60}


@dataclass
class Context:
    box: pd.DataFrame          # bu sezon box score
    prior: pd.DataFrame        # geçen sezon box score
    games: pd.DataFrame        # bu sezon maç listesi


def _team_games(games: pd.DataFrame) -> pd.DataFrame:
    g = games[games["played"]]
    a = g[["round", "game_code", "home", "away"]].rename(columns={"home": "team", "away": "opp"}).assign(is_home=1)
    b = g[["round", "game_code", "away", "home"]].rename(columns={"away": "team", "home": "opp"}).assign(is_home=0)
    return pd.concat([a, b]).sort_values(["team", "round"])


def current_team(box: pd.DataFrame) -> pd.Series:
    last = box.sort_values("round").groupby("player_id").tail(1)
    return last.set_index("player_id")["team"]


def availability_and_minutes(ctx: Context) -> pd.DataFrame:
    """Oyuncunun güncel takımının son maçlarına göre oynama olasılığı ve dakika beklentisi."""
    box, prior = ctx.box, ctx.prior
    teams = current_team(box)
    tg = _team_games(ctx.games)
    rows = []
    prior_min = prior[prior["minutes"] > 0].groupby("player_id")["minutes"].mean() if len(prior) else pd.Series(dtype=float)
    for pid, team in teams.items():
        pg = box[(box["player_id"] == pid) & (box["team"] == team)].set_index("game_code")["minutes"]
        tgames = tg[tg["team"] == team].sort_values("round")
        # takıma katıldığı ilk maçtan itibaren
        first_round = box[(box["player_id"] == pid) & (box["team"] == team)]["round"].min()
        tgames = tgames[tgames["round"] >= first_round]
        mins = np.array([float(pg.get(gc, 0.0)) for gc in tgames["game_code"]])
        if len(mins) == 0:
            continue
        last = mins[-5:]
        w = np.linspace(0.6, 1.4, len(last))
        played = (last > 0).astype(float)
        p_play = float((w * played).sum() / w.sum())
        p_play = (p_play * len(last) + 0.85 * 1.5) / (len(last) + 1.5)  # küçük örnek küçültmesi
        if len(mins) >= 2 and mins[-1] == 0 and mins[-2] == 0:
            p_play = min(p_play, 0.30)  # üst üste iki maç yok: muhtemel sakatlık
        on = mins[mins > 0]
        if len(on):
            ew = 0.35 ** np.arange(len(on))[::-1]  # son maç en ağır
            ewma = float((on * ew).sum() / ew.sum())
            season_avg = float(on.mean())
            m = 0.6 * ewma + 0.4 * season_avg
        else:
            m = float(prior_min.get(pid, 12.0))
        if pid in prior_min.index and len(on) < 4:
            k = len(on) / 4
            m = k * m + (1 - k) * float(prior_min[pid])
        trend = float(on[-3:].mean() - on.mean()) if len(on) >= 4 else 0.0
        rows.append({"player_id": pid, "team": team, "p_play": p_play, "min_proj": m,
                     "min_avg": float(on.mean()) if len(on) else np.nan,
                     "min_trend": trend, "gp": int((mins > 0).sum()), "team_gp": len(mins)})
    return pd.DataFrame(rows).set_index("player_id")


def per_minute_rates(ctx: Context) -> pd.DataFrame:
    box, prior = ctx.box, ctx.prior
    cur = box[box["minutes"] > 0].groupby("player_id")[CATS + ["minutes"]].sum()
    league_rate = cur[CATS].sum() / cur["minutes"].sum()
    if len(prior):
        pr = prior[prior["minutes"] > 0].groupby("player_id")[CATS + ["minutes"]].sum()
    else:
        pr = pd.DataFrame(columns=CATS + ["minutes"])
    out = {}
    for pid, r in cur.iterrows():
        if pid in pr.index and pr.loc[pid, "minutes"] > 50:
            prior_rate = pr.loc[pid, CATS] / pr.loc[pid, "minutes"]
            k = PRIOR_MIN
        else:
            prior_rate = league_rate
            k = LEAGUE_PRIOR_MIN
        out[pid] = (r[CATS] + k * prior_rate) / (r["minutes"] + k)
    rates = pd.DataFrame(out).T.astype(float)
    rates.index.name = "player_id"
    return rates


def dispersion(ctx: Context) -> pd.DataFrame:
    """Kategori bazında varyasyon katsayısı (dakika başı üretimin oynaklığı)."""
    b = ctx.box[ctx.box["minutes"] >= 5].copy()
    res = {}
    for pid, g in b.groupby("player_id"):
        cv = {}
        for c in CATS:
            pm = g[c] / g["minutes"]
            mu = pm.mean()
            n = len(pm)
            if n >= 3 and mu > 0:
                raw = float(pm.std(ddof=1) / mu)
                cv[c] = (raw * n + BASE_CV[c] * 4) / (n + 4)
            else:
                cv[c] = BASE_CV[c]
        res[pid] = cv
    return pd.DataFrame(res).T


def opponent_factors(ctx: Context) -> pd.DataFrame:
    """Rakibin her kategoride yedirdiği üretim / lig ortalaması (40 dakikaya normalize)."""
    b = ctx.box
    per_game = b.groupby(["game_code", "team", "opp"])[CATS + ["minutes"]].sum().reset_index()
    per_game[CATS] = per_game[CATS].div(per_game["minutes"].replace(0, np.nan), axis=0) * 200
    lg = per_game[CATS].mean()
    allowed = per_game.groupby("opp")[CATS].agg(["sum", "count"])
    out = {}
    for team in allowed.index:
        f = {}
        for c in CATS:
            s, n = allowed.loc[team, (c, "sum")], allowed.loc[team, (c, "count")]
            v = (s + OPP_SHRINK_GAMES * lg[c]) / (n + OPP_SHRINK_GAMES)
            f[c] = float(np.clip(v / lg[c], 0.85, 1.15))
        out[team] = f
    return pd.DataFrame(out).T


def names_table(ctx: Context) -> pd.Series:
    return ctx.box.sort_values("round").groupby("player_id")["player_raw"].last()


def project_round(ctx: Context, round_no: int, overrides: dict | None = None) -> pd.DataFrame:
    """round_no turundaki her oyuncu için beklenen değer ve standart sapma."""
    av = availability_and_minutes(ctx)
    rates = per_minute_rates(ctx)
    disp = dispersion(ctx)
    opp_f = opponent_factors(ctx)
    g = ctx.games[ctx.games["round"] == round_no]
    sched = {}
    for _, r in g.iterrows():
        sched[r["home"]] = (r["away"], 1)
        sched[r["away"]] = (r["home"], 0)
    names = names_table(ctx)
    rows = []
    for pid, a in av.iterrows():
        if pid not in rates.index:
            continue
        team = a["team"]
        opp, home = sched.get(team, (None, None))
        p_play = a["p_play"] if opp is not None else 0.0
        minutes = a["min_proj"]
        if overrides and pid in overrides:
            o = overrides[pid]
            p_play = o.get("p_play", p_play)
            minutes = o.get("min_proj", minutes)
        row = {"player_id": pid, "name": names.get(pid, pid), "team": team, "opp": opp, "home": home,
               "p_play": p_play, "min_proj": minutes, "min_trend": a["min_trend"], "gp": a["gp"],
               "team_gp": a["team_gp"]}
        for c in CATS:
            f = opp_f.loc[opp, c] if opp in opp_f.index else 1.0
            h = HOME_BONUS[c] if home == 1 else 1.0
            mu = minutes * rates.loc[pid, c] * f * h
            cv = disp.loc[pid, c] if pid in disp.index else BASE_CV[c]
            row[f"{c}_mu"] = max(mu, 0.0) if c != "pir" else mu
            row[f"{c}_sd"] = max(abs(mu) * cv, 0.8)
        rows.append(row)
    return pd.DataFrame(rows).set_index("player_id")


def project_generic(ctx: Context) -> pd.DataFrame:
    """Rakipten bağımsız, sezonun geri kalanı için ortalama maç tahmini (takas analizi için)."""
    av = availability_and_minutes(ctx)
    rates = per_minute_rates(ctx)
    disp = dispersion(ctx)
    names = names_table(ctx)
    rows = []
    for pid, a in av.iterrows():
        if pid not in rates.index:
            continue
        # sezon geneli: kısa vadeli sakatlık etkisini yumuşat
        p_play = 0.5 * a["p_play"] + 0.5 * 0.9
        row = {"player_id": pid, "name": names.get(pid, pid), "team": a["team"], "p_play": p_play,
               "min_proj": a["min_proj"], "min_trend": a["min_trend"], "gp": a["gp"]}
        for c in CATS:
            mu = a["min_proj"] * rates.loc[pid, c]
            cv = disp.loc[pid, c] if pid in disp.index else BASE_CV[c]
            row[f"{c}_mu"] = mu
            row[f"{c}_sd"] = max(abs(mu) * cv, 0.8)
        rows.append(row)
    return pd.DataFrame(rows).set_index("player_id")
