"""Tek bir takas teklifini hafta hafta değerlendirir.

İki fikstürü birlikte kullanır:
  - Gerçek EuroLeague fikstürü: her oyuncunun o turdaki rakibi, iç saha/deplasman
  - Bizim lig fikstürü: o hafta kiminle eşleştiğin ve rakibin kadrosu
Her hafta, takastan önce ve sonra en iyi dizilimle (rezerv/ilk5/kaptan/6. adam)
eşleşmeyi simüle eder; iki taraf için de.

Kullanım:
  python src/evaluate_trade.py --partner Mali --give "Kevin Punter,Umoja Gibson" \
      --get "Wade Baldwin IV,Evan Fournier" --from-week 7 --to-week 16
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import model  # noqa: E402
import optimize  # noqa: E402
import run_weekly as rw  # noqa: E402

DATA = rw.DATA


def evaluate(partner, give, get, from_week, to_week, sims=1500, offline=True, league=None, verbose=True):
    league = league or json.loads((DATA / "league.json").read_text())
    hf = DATA / "human_factors.json"
    if hf.exists():
        league["human_factors"] = json.loads(hf.read_text())
    me = league["my_team"]
    ctx = rw.load(2026, 2025, offline)
    rosters_ids, rep = rw.match_rosters(ctx, league)
    ta = league["team_averages_after_week6"]
    league_avg = {c: float(np.mean([v[i] for v in ta.values()])) / 12 for i, c in enumerate(["pts", "reb", "ast"])}

    name2id = rep.dropna(subset=["player_id"]).set_index("fantasy_name")["player_id"].to_dict()
    give_ids = [name2id[n] for n in give]
    get_ids = [name2id[n] for n in get]
    assert all(g in rosters_ids[me] for g in give_ids), "Verilen oyuncular senin kadronda değil"
    assert all(g in rosters_ids[partner] for g in get_ids), "Alınan oyuncular karşı kadroda değil"

    after = {t: list(ids) for t, ids in rosters_ids.items()}
    after[me] = [p for p in after[me] if p not in give_ids] + get_ids
    after[partner] = [p for p in after[partner] if p not in get_ids] + give_ids

    rows = []
    for w in range(from_week, to_week + 1):
        proj = rw.attach_fantasy_names(model.project_round(ctx, w), rep)
        sub = lambda ids: proj.loc[[i for i in ids if i in proj.index]]  # noqa: E731
        rec = {"hafta": w}
        for side, team in (("ben", me), ("o", partner)):
            opp = rw.opponent_of(league, team, w)
            # rakip partner ise takasın iki tarafı da değişir
            opp_before, opp_after = rosters_ids[opp], after[opp]
            b = optimize.optimize_week(sub(rosters_ids[team]), sub(opp_before), league_avg, n_sims=sims, seed=w)
            a = optimize.optimize_week(sub(after[team]), sub(opp_after), league_avg, n_sims=sims, seed=w)
            rec[f"{side}_rakip"] = opp
            rec[f"{side}_once"] = b["result"]["exp_points"]
            rec[f"{side}_sonra"] = a["result"]["exp_points"]
            if side == "ben":
                rec["kat_once"] = b["result"]["cat_probs"]
                rec["kat_sonra"] = a["result"]["cat_probs"]
                rec["win_once"] = b["result"]["win_prob"]
                rec["win_sonra"] = a["result"]["win_prob"]
                # bu haftaki EuroLeague rakipleri (gerçek fikstür)
                rec["el_maclari"] = {proj.loc[i, "name_fantasy"]: f"{proj.loc[i, 'team']}-{proj.loc[i, 'opp']}{' (ev)' if proj.loc[i, 'home'] == 1 else ''}"
                                     for i in give_ids + get_ids if i in proj.index}
        rows.append(rec)
        if verbose: print(f"H{w}: ben {rec['ben_once']:.2f}->{rec['ben_sonra']:.2f} vs {rec['ben_rakip']} | "
              f"{partner} {rec['o_once']:.2f}->{rec['o_sonra']:.2f}", flush=True)

    gen = rw.attach_fantasy_names(model.project_generic(ctx), rep)
    players = gen.loc[give_ids + get_ids, ["name_fantasy", "team", "p_play", "min_proj", "min_trend",
                                           "pts_mu", "reb_mu", "ast_mu", "pir_mu"]]
    return pd.DataFrame(rows), players


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--partner", required=True)
    ap.add_argument("--give", required=True)
    ap.add_argument("--get", required=True)
    ap.add_argument("--from-week", type=int, default=7)
    ap.add_argument("--to-week", type=int, default=16)
    ap.add_argument("--sims", type=int, default=1500)
    a = ap.parse_args()
    df, players = evaluate(a.partner, [s.strip() for s in a.give.split(",")],
                           [s.strip() for s in a.get.split(",")], a.from_week, a.to_week, a.sims)
    out = DATA / "trade_eval.json"
    out.write_text(json.dumps({"weeks": df.to_dict("records"), "players": players.round(2).to_dict("index")},
                              ensure_ascii=False, indent=1, default=float))
    print(players.round(1).to_string())
    print(f"\nToplam: ben {df['ben_sonra'].sum() - df['ben_once'].sum():+.2f}, "
          f"{a.partner} {df['o_sonra'].sum() - df['o_once'].sum():+.2f}")
