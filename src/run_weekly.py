"""Haftalık ana akış: veri -> tahmin -> dizilim önerisi -> booster zamanlaması -> takas/serbest oyuncu.

Kullanım:
  python src/run_weekly.py                 # sıradaki tur
  python src/run_weekly.py --week 7        # belirli tur
  python src/run_weekly.py --offline       # veri çekmeden, mevcut CSV'lerle
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
import trade  # noqa: E402
from names import build_matcher, pretty  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
DATA, REPORTS = ROOT / "data", ROOT / "reports"
CYCLES = [(1, 11), (12, 22), (23, 31)]


def load(season: int, prior: int, offline: bool):
    if not offline:
        import fetch
        fetch.update_season(prior)
        fetch.update_season(season)
    games = pd.read_csv(DATA / f"games_{season}.csv")
    games["played"] = games["played"].astype(str).str.lower().isin(["true", "1"])
    box = pd.read_csv(DATA / f"boxscores_{season}.csv")
    pp = DATA / f"boxscores_{prior}.csv"
    prior_box = pd.read_csv(pp) if pp.exists() else box.iloc[0:0]
    for b in (box, prior_box):
        b["player_id"] = b["player_id"].astype(str).str.strip()
    return model.Context(box=box, prior=prior_box, games=games)


def match_rosters(ctx, league):
    names = ctx.prior.sort_values("round").groupby("player_id")["player_raw"].last().to_dict() if len(ctx.prior) else {}
    names.update(ctx.box.sort_values("round").groupby("player_id")["player_raw"].last().to_dict())
    this_season = set(ctx.box["player_id"])
    match = build_matcher(names)
    rosters_ids, report = {}, []
    for team, plist in league["rosters"].items():
        ids = []
        for n in plist:
            pid, score = match(n)
            report.append({"team": team, "fantasy_name": n, "player_id": pid,
                           "boxscore_name": names.get(pid), "score": round(score, 2),
                           "this_season": pid in this_season})
            if pid:
                ids.append(pid)
        rosters_ids[team] = ids
    rep = pd.DataFrame(report)
    rep.to_csv(DATA / "name_matching.csv", index=False)
    return rosters_ids, rep


def attach_fantasy_names(proj, rep):
    m = rep.dropna(subset=["player_id"]).set_index("player_id")["fantasy_name"]
    proj = proj.copy()
    proj["name_fantasy"] = [m.get(i, pretty(n)) for i, n in zip(proj.index, proj["name"])]
    return proj


def opponent_of(league, team, week):
    for a, b in league["fixtures"].get(str(week), []):
        if team in (a, b):
            return b if a == team else a
    return None


def next_round(games):
    pending = games[~games["played"]]
    return int(pending["round"].min()) if len(pending) else int(games["round"].max())


def cycle_end(week):
    for a, b in CYCLES:
        if a <= week <= b:
            return b
    return week


def fmt_lineup(lu, proj):
    nm = lambda p: proj.loc[p, "name_fantasy"] if p in proj.index else p  # noqa: E731
    return {"kaptan": nm(lu["captain"]), "ilk5": [nm(p) for p in lu["starters"]],
            "6_adam": nm(lu.get("sixth")) if lu.get("sixth") else None, "rezerv": nm(lu["reserve"])}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--season", type=int, default=2026)
    ap.add_argument("--prior", type=int, default=2025)
    ap.add_argument("--week", type=int)
    ap.add_argument("--offline", action="store_true")
    ap.add_argument("--sims", type=int, default=3000)
    args = ap.parse_args()

    league = json.loads((DATA / "league.json").read_text())
    hf_path = DATA / "human_factors.json"  # repoya girmeyen yerel dosya (varsa)
    if hf_path.exists():
        league["human_factors"] = json.loads(hf_path.read_text())
    me = league["my_team"]
    ctx = load(args.season, args.prior, args.offline)
    week = args.week or next_round(ctx.games)
    rosters_ids, rep = match_rosters(ctx, league)

    ta = league["team_averages_after_week6"]
    league_avg = {c: float(np.mean([v[i] for v in ta.values()])) / 12 for i, c in enumerate(["pts", "reb", "ast"])}

    overrides = {}
    for name, o in league.get("availability_overrides", {}).items():
        pid = rep.loc[rep["fantasy_name"] == name, "player_id"]
        if len(pid) and pd.notna(pid.iloc[0]):
            overrides[pid.iloc[0]] = o

    def proj_for(w):
        p = model.project_round(ctx, w, overrides)
        return attach_fantasy_names(p, rep)

    proj = proj_for(week)
    proj.to_csv(DATA / f"projections_r{week:02d}.csv")

    def team_proj(p, team):
        ids = [i for i in rosters_ids[team] if i in p.index]
        return p.loc[ids]

    opp = opponent_of(league, me, week)
    res = optimize.optimize_week(team_proj(proj, me), team_proj(proj, opp), league_avg, n_sims=args.sims)

    # Booster zamanlaması: döngünün kalan haftalarında booster'ın getirisi
    booster_plan = []
    used = league.get("booster_status", {})
    end = cycle_end(week)
    cycle_key = next(f"H{a}-{b}" for a, b in CYCLES if a <= week <= b)
    if "kullanılmadı" in str(used.get(cycle_key, "kullanılmadı")):
        for w in range(week, end + 1):
            o = opponent_of(league, me, w)
            if o is None:
                continue
            pw = proj if w == week else proj_for(w)
            r = res if w == week else optimize.optimize_week(team_proj(pw, me), team_proj(pw, o), league_avg,
                                                             n_sims=max(1500, args.sims // 2))
            booster_plan.append({"hafta": w, "rakip": o, "booster_oyuncu": pw.loc[r["booster_best"]["player"], "name_fantasy"],
                                 "kazanc_puan": round(r["booster_best"]["gain"], 3),
                                 "booster_siz_beklenen": round(r["result"]["exp_points"], 2)})

    gen = model.project_generic(ctx).pipe(attach_fantasy_names, rep)
    active_ids = {t: [i for i in ids if i in gen.index] for t, ids in rosters_ids.items()}
    trades = trade.find_trades(gen, active_ids, league, me, week, league_avg)
    # sahibi olan ama bu sezon oynamamış oyuncular da serbest sayılmasın
    fas = trade.free_agents(gen, rosters_ids, league, me, week, league_avg, active_ids=active_ids)

    write_report(week, me, opp, res, proj, booster_plan, trades, fas, rep, rosters_ids, cycle_key)


def write_report(week, me, opp, res, proj, booster_plan, trades, fas, rep, rosters_ids, cycle_key):
    REPORTS.mkdir(exist_ok=True)
    lu = fmt_lineup(res["lineup"], proj)
    r = res["result"]
    L = []
    L.append(f"# Hafta {week} raporu — {me} vs {opp}\n")
    L.append(f"**Kazanma olasılığı (2+ kategori): %{r['win_prob']*100:.0f}**  |  Beklenen puan: {r['exp_points']:.2f} / 3\n")
    L.append("| Kategori | Kazanma olasılığı | Senin beklenen | Rakip beklenen |\n|---|---|---|---|")
    labels = {"pts": "Sayı", "reb": "Ribaund", "ast": "Asist"}
    for c in ["pts", "reb", "ast"]:
        L.append(f"| {labels[c]} | %{r['cat_probs'][c]*100:.0f} | {res['my_mean'][c]} | {res['opp_mean'][c]} |")
    L.append(f"| PIR (beraberlik) | – | {res['my_mean']['pir']} | {res['opp_mean']['pir']} |\n")

    L.append("## Önerilen dizilim\n")
    L.append(f"- **Kaptan:** {lu['kaptan']}")
    L.append(f"- **İlk 5:** {', '.join(lu['ilk5'])}")
    L.append(f"- **6. adam:** {lu['6_adam']}")
    L.append(f"- **Rezerv:** {lu['rezerv']}\n")

    L.append("## Senin oyuncuların bu hafta\n")
    L.append("| Oyuncu | Takım | Rakip | Oynama % | Dakika | Sayı | Rib | Ast | PIR | Dakika trendi |\n|---|---|---|---|---|---|---|---|---|---|")
    mine = proj.loc[[i for i in rosters_ids[me] if i in proj.index]].sort_values("pir_mu", ascending=False)
    for pid, x in mine.iterrows():
        tr = f"{x['min_trend']:+.1f}" if abs(x["min_trend"]) >= 0.05 else "0"
        L.append(f"| {x['name_fantasy']} | {x['team']} | {x['opp'] or '-'}{' (ev)' if x['home'] == 1 else ''} | "
                 f"%{x['p_play']*100:.0f} | {x['min_proj']:.1f} | {x['pts_mu']:.1f} | {x['reb_mu']:.1f} | "
                 f"{x['ast_mu']:.1f} | {x['pir_mu']:.1f} | {tr} |")

    if booster_plan:
        L.append(f"\n## Booster ({cycle_key}) zamanlaması\n")
        L.append("Booster'ın her haftaki getirisi (beklenen ekstra kategori puanı):\n")
        L.append("| Hafta | Rakip | En iyi booster oyuncusu | Ekstra puan | Booster'sız beklenen |\n|---|---|---|---|---|")
        for b in booster_plan:
            L.append(f"| {b['hafta']} | {b['rakip']} | {b['booster_oyuncu']} | +{b['kazanc_puan']:.2f} | {b['booster_siz_beklenen']:.2f} |")
        best = max(booster_plan, key=lambda b: b["kazanc_puan"])
        if best["hafta"] == week:
            L.append(f"\n**Öneri: booster'ı BU HAFTA {best['booster_oyuncu']} üzerinde kullan.**")
        else:
            L.append(f"\n**Öneri: bu hafta booster'ı saklama, en verimli hafta {best['hafta']} ({best['rakip']}).** "
                     "Döngünün son haftasında hâlâ kullanmadıysan o hafta mutlaka kullan.")

    L.append("\n## Takas fikirleri (sezonun kalanı)\n")
    if trades is not None and len(trades):
        L.append("Kazanç = sezon sonuna kadar beklenen ekstra kategori puanı. PIR sütunları arkadaşının takasa nasıl bakacağını gösterir.\n")
        L.append("| Arkadaş | Ver | Al | Senin kazancın | Onun modele göre kazancı | Verilen PIR | Alınan PIR |\n|---|---|---|---|---|---|---|")
        for _, t in trades.iterrows():
            L.append(f"| {t['partner']} | {t['give']} | {t['get']} | +{t['my_gain_season_pts']} | {t['partner_model_gain']:+} | {t['give_pir']} | {t['get_pir']} |")
    else:
        L.append("Kabul edilebilir ve kazançlı takas bulunamadı.")

    L.append("\n## Serbest oyuncular (free dönemi)\n")
    if fas is not None and len(fas):
        L.append("| Oyuncu | Takım | Kimi bırakırsın | Sezon kazancı | Sayı | Rib | Ast | Dakika |\n|---|---|---|---|---|---|---|---|")
        for _, f in fas.iterrows():
            L.append(f"| {f['player']} | {f['team']} | {f['drop']} | +{f['my_gain_season_pts']} | {f['pts']} | {f['reb']} | {f['ast']} | {f['min']} |")
    else:
        L.append("Kadronu geliştirecek serbest oyuncu yok.")

    miss = rep[rep["player_id"].isna()]
    low = rep[(rep["score"] < 0.9) & rep["player_id"].notna()]
    if len(miss) or len(low):
        L.append("\n## İsim eşleştirme uyarıları\n")
        for _, m in miss.iterrows():
            L.append(f"- Bulunamadı: {m['fantasy_name']} ({m['team']})")
    idle = rep[rep["player_id"].notna() & ~rep["this_season"].astype(bool)]
    if len(idle):
        L.append("\n## Bu sezon hiç forma giymeyenler (sakat / kadro dışı)\n")
        for _, m in idle.iterrows():
            L.append(f"- {m['fantasy_name']} ({m['team']})")
        for _, m in low.iterrows():
            L.append(f"- Kontrol et: {m['fantasy_name']} → {m['boxscore_name']} ({m['score']})")

    text = "\n".join(L) + "\n"
    (REPORTS / f"week_{week:02d}.md").write_text(text)
    (REPORTS / "latest.md").write_text(text)
    (REPORTS / "latest.json").write_text(json.dumps({
        "week": week, "opponent": opp, "lineup": lu, "result": res["result"],
        "booster_plan": booster_plan,
        "trades": trades.to_dict("records") if trades is not None else [],
        "free_agents": fas.to_dict("records") if fas is not None else []}, ensure_ascii=False, indent=1, default=float))
    print(text)


if __name__ == "__main__":
    main()
