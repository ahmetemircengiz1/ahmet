"""Kadro kararları: rezerv, ilk 5, kaptan, 6. adam, booster.

Oyunun kuralları:
  - 13 kişiden 1 rezerv (0 puan), 12 aktif
  - ilk 5 x2, kaptan x3, bench x1
  - 6. adam: kaptan oynamazsa kaptanın x3'ünü alır, kaptan oynarsa normal bench (x1)
  - booster: ilk 5'ten birinin çarpanını +1 artırır (fikstür döngüsü başına 1 kez)
  - PTS, REB, AST ayrı ayrı karşılaştırılır; beraberlikte PIR toplamı (çarpanlı) belirler
  - Puan = kazanılan kategori sayısı
"""
from __future__ import annotations

from itertools import combinations

import numpy as np
import pandas as pd
from scipy.stats import norm

CATS = ["pts", "reb", "ast", "pir"]
SCORE_CATS = ["pts", "reb", "ast"]


# ---------------------------------------------------------------- simülasyon
def simulate(proj: pd.DataFrame, n_sims: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Döndürür: X[sim, oyuncu, kategori] ve played[sim, oyuncu]."""
    n = len(proj)
    X = np.zeros((n_sims, n, len(CATS)))
    played = rng.random((n_sims, n)) < proj["p_play"].to_numpy()[None, :]
    for j, c in enumerate(CATS):
        mu = proj[f"{c}_mu"].to_numpy()
        sd = proj[f"{c}_sd"].to_numpy()
        if c == "pir":
            vals = rng.normal(mu, sd, size=(n_sims, n))
        else:
            mu_ = np.maximum(mu, 1e-3)
            shape = (mu_ / sd) ** 2
            scale = sd ** 2 / mu_
            vals = rng.gamma(shape, scale, size=(n_sims, n))
        X[:, :, j] = np.where(played, vals, 0.0)
    return X, played


def generic_value(proj: pd.DataFrame, league_avg: dict) -> pd.Series:
    """İnsanların/rakiplerin kadro kurarken kullandığı kaba değer: kategorileri lig ortalamasına göre eşitle."""
    v = sum(proj[f"{c}_mu"] / league_avg[c] for c in SCORE_CATS)
    return v * proj["p_play"]


def heuristic_lineup(proj: pd.DataFrame, league_avg: dict) -> dict:
    """Rakip için varsayılan dizilim: en değerli 5 ilk 5, en değerlisi kaptan, en zayıfı rezerv."""
    v = generic_value(proj, league_avg).sort_values(ascending=False)
    ids = list(v.index)
    return {"captain": ids[0], "starters": ids[:5], "reserve": ids[-1], "sixth": ids[5] if len(ids) > 5 else None}


def weights_for(ids: list, lineup: dict, booster: str | None = None) -> np.ndarray:
    w = np.ones(len(ids))
    for i, pid in enumerate(ids):
        if pid == lineup["reserve"]:
            w[i] = 0
        elif pid == lineup["captain"]:
            w[i] = 3
        elif pid in lineup["starters"]:
            w[i] = 2
        if booster is not None and pid == booster:
            w[i] += 1
    return w


def team_totals(X: np.ndarray, played: np.ndarray, ids: list, lineup: dict, booster: str | None = None) -> np.ndarray:
    """6. adam kuralı dahil takım toplamları: [sim, kategori]."""
    w = weights_for(ids, lineup, booster)
    tot = np.einsum("sic,i->sc", X, w)
    six = lineup.get("sixth")
    if six is not None and six in ids and six != lineup["reserve"]:
        ci, si = ids.index(lineup["captain"]), ids.index(six)
        cap_out = ~played[:, ci]
        cap_mult = w[ci]  # booster kaptandaysa 4
        # kaptan oynamadıysa 6. adam x1 yerine kaptanın çarpanını alır
        tot += (cap_out[:, None] * (cap_mult - 1)) * X[:, si, :]
    return tot


def match_result(my: np.ndarray, op: np.ndarray) -> dict:
    """Her simülasyonda kategori kazanımları; beraberlikte PIR."""
    pir_win = my[:, 3] > op[:, 3]
    wins = []
    for j in range(3):
        d = my[:, j] - op[:, j]
        wins.append(np.where(np.abs(d) < 1e-9, pir_win, d > 0))
    W = np.stack(wins, axis=1)
    pts = W.sum(1)
    return {"exp_points": float(pts.mean()), "win_prob": float((pts >= 2).mean()),
            "cat_probs": dict(zip(SCORE_CATS, W.mean(0).round(3).tolist())),
            "dist": {k: float((pts == k).mean()) for k in range(4)}}


# ---------------------------------------------------------------- optimizasyon
def optimize_week(my_proj: pd.DataFrame, opp_proj: pd.DataFrame, league_avg: dict,
                  n_sims: int = 3000, seed: int = 7, opp_lineup: dict | None = None) -> dict:
    rng = np.random.default_rng(seed)
    ids = list(my_proj.index)
    oids = list(opp_proj.index)
    X, P = simulate(my_proj, n_sims, rng)
    Y, Q = simulate(opp_proj, n_sims, rng)
    opp_lineup = opp_lineup or heuristic_lineup(opp_proj, league_avg)
    op = team_totals(Y, Q, oids, opp_lineup)

    n = len(ids)
    # Rezerv adayları: oynama ihtimali en düşük / en az katkı verenler
    val = generic_value(my_proj, league_avg)
    reserve_cands = list(val.sort_values().index[:4])

    # Bütün (rezerv, ilk5, kaptan) kombinasyonlarını vektörel değerlendir (6. adamsız)
    lineups, W = [], []
    for res in reserve_cands:
        active = [p for p in ids if p != res]
        for five in combinations(active, 5):
            for cap in five:
                w = np.ones(n)
                w[ids.index(res)] = 0
                for p in five:
                    w[ids.index(p)] = 2
                w[ids.index(cap)] = 3
                lineups.append({"reserve": res, "starters": list(five), "captain": cap})
                W.append(w)
    W = np.array(W)  # [L, n]
    scores = np.empty(len(W))
    pir_op = op[:, 3]
    Xr = X.transpose(1, 0, 2).reshape(n, -1)  # [n, S*C] -> BLAS matmul
    for s in range(0, len(W), 1500):
        Wc = W[s:s + 1500]
        tot = (Wc @ Xr).reshape(len(Wc), n_sims, len(CATS))  # [L, S, C]
        pir_win = tot[:, :, 3] > pir_op[None, :]
        pts = np.zeros(tot.shape[:2])
        for j in range(3):
            d = tot[:, :, j] - op[None, :, j]
            pts += np.where(np.abs(d) < 1e-9, pir_win, d > 0)
        scores[s:s + 1500] = pts.mean(1)
    top = np.argsort(-scores)[:25]

    # En iyi adaylarda 6. adamı seç ve kesin sonucu hesapla
    best = None
    for t in top:
        lu = dict(lineups[t])
        bench = [p for p in ids if p != lu["reserve"] and p not in lu["starters"]]
        for six in bench:
            lu2 = {**lu, "sixth": six}
            r = match_result(team_totals(X, P, ids, lu2), op)
            if best is None or r["exp_points"] > best[1]["exp_points"] + 1e-9:
                best = (lu2, r)
    lineup, res = best

    # Booster bu hafta kullanılırsa kazanç
    boost = []
    for p in lineup["starters"]:
        r = match_result(team_totals(X, P, ids, lineup, booster=p), op)
        boost.append((p, r["exp_points"] - res["exp_points"], r))
    boost.sort(key=lambda x: -x[1])

    my_tot = team_totals(X, P, ids, lineup)
    return {"lineup": lineup, "result": res, "opp_lineup": opp_lineup,
            "booster_options": [(p, round(g, 3)) for p, g, _ in boost],
            "booster_best": {"player": boost[0][0], "gain": boost[0][1], "result": boost[0][2]},
            "my_mean": dict(zip(CATS, my_tot.mean(0).round(1))),
            "opp_mean": dict(zip(CATS, op.mean(0).round(1)))}


# ---------------------------------------------------------------- analitik (takas için hızlı)
def team_moments(proj: pd.DataFrame, lineup: dict) -> tuple[np.ndarray, np.ndarray]:
    ids = list(proj.index)
    w = weights_for(ids, lineup)
    mu, var = np.zeros(4), np.zeros(4)
    p = proj["p_play"].to_numpy()
    for j, c in enumerate(CATS):
        m = proj[f"{c}_mu"].to_numpy()
        s = proj[f"{c}_sd"].to_numpy()
        ex = p * m
        vx = p * (s ** 2 + m ** 2) - ex ** 2
        mu[j] = (w * ex).sum()
        var[j] = (w ** 2 * vx).sum()
    return mu, var


def analytic_points(mu_a, var_a, mu_b, var_b) -> float:
    """Normal yaklaşımla beklenen kategori puanı (PIR beraberliği ihmal edilebilir)."""
    z = (mu_a[:3] - mu_b[:3]) / np.sqrt(var_a[:3] + var_b[:3] + 1e-9)
    return float(norm.cdf(z).sum())


def best_lineup_fast(proj: pd.DataFrame, league_avg: dict) -> dict:
    """Takas analizinde her kadro için hızlı makul dizilim."""
    return heuristic_lineup(proj, league_avg)
