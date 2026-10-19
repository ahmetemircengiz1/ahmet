"""EuroLeague resmi API'sinden fikstür ve oyuncu box score verisini çeker.

Önbellekli çalışır: daha önce çekilmiş maçlar tekrar indirilmez, her hafta
sadece yeni oynanan maçlar eklenir.

Çıktılar (data/ altında):
  games_<sezon>.csv      -> tüm maçlar (round, ev/deplasman, tarih, oynandı mı)
  boxscores_<sezon>.csv  -> maç x oyuncu satırları (dakika, PTS, REB, AST, PIR ...)
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import pandas as pd

from euroleague_api.EuroLeagueData import EuroLeagueData
from euroleague_api.boxscore_data import BoxScoreData

DATA = Path(__file__).resolve().parent.parent / "data"

# API kolon adı -> bizim kolon adımız
COLMAP = {
    "Player_ID": "player_id",
    "Player": "player_raw",
    "Team": "team",
    "IsStarter": "is_starter",
    "IsPlaying": "is_playing",
    "Minutes": "minutes_raw",
    "Points": "pts",
    "TotalRebounds": "reb",
    "OffensiveRebounds": "oreb",
    "DefensiveRebounds": "dreb",
    "Assistances": "ast",
    "Steals": "stl",
    "Turnovers": "tov",
    "BlocksFavour": "blk",
    "Valuation": "pir",
    "FieldGoalsAttempted2": "fga2",
    "FieldGoalsAttempted3": "fga3",
    "FreeThrowsAttempted": "fta",
}


def minutes_to_float(v) -> float:
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return 0.0
    s = str(v).strip()
    if not s or s.upper() in {"DNP", "NAN", "NONE"}:
        return 0.0
    if ":" in s:
        m, sec = s.split(":")[:2]
        try:
            return int(m) + int(sec) / 60
        except ValueError:
            return 0.0
    try:
        return float(s)
    except ValueError:
        return 0.0


def fetch_games(season: int) -> pd.DataFrame:
    df = EuroLeagueData("E").get_gamecodes_season(season)
    keep = {
        "Round": "round",
        "gameCode": "game_code",
        "homecode": "home",
        "awaycode": "away",
        "hometeam": "home_name",
        "awayteam": "away_name",
        "date": "date",
        "played": "played",
        "Phase": "phase",
        "homescore": "home_score",
        "awayscore": "away_score",
    }
    cols = {k: v for k, v in keep.items() if k in df.columns}
    out = df[list(cols)].rename(columns=cols)
    out["season"] = season
    return out


def fetch_boxscore(season: int, game_code: int) -> pd.DataFrame:
    raw = BoxScoreData("E").get_players_boxscore_stats(season, game_code)
    raw = raw[~raw["Player"].isin(["Team", "Total"])].copy()
    cols = {k: v for k, v in COLMAP.items() if k in raw.columns}
    df = raw[list(cols)].rename(columns=cols)
    df["home"] = raw["Home"].values
    df["minutes"] = df["minutes_raw"].map(minutes_to_float)
    df["season"] = season
    df["game_code"] = game_code
    for c in ["pts", "reb", "ast", "pir", "stl", "tov", "blk", "oreb", "dreb", "fga2", "fga3", "fta"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)
    return df.drop(columns=["minutes_raw"])


def update_season(season: int, sleep: float = 0.4) -> tuple[pd.DataFrame, pd.DataFrame]:
    DATA.mkdir(exist_ok=True)
    games = fetch_games(season)
    games.to_csv(DATA / f"games_{season}.csv", index=False)

    box_path = DATA / f"boxscores_{season}.csv"
    box = pd.read_csv(box_path) if box_path.exists() else pd.DataFrame()
    have = set(box["game_code"].unique()) if len(box) else set()

    todo = games[(games["played"]) & (~games["game_code"].isin(have))]
    print(f"[{season}] {len(games)} maç, {len(have)} önbellekte, {len(todo)} yeni çekilecek", flush=True)
    new = []
    for _, g in todo.iterrows():
        try:
            b = fetch_boxscore(season, int(g["game_code"]))
            b["round"] = int(g["round"])
            b["opp"] = b.apply(lambda r: g["away"] if r["home"] == 1 else g["home"], axis=1)
            new.append(b)
        except Exception as e:  # tek maç hatası tüm akışı durdurmasın
            print(f"  ! maç {g['game_code']} alınamadı: {e}", file=sys.stderr)
        time.sleep(sleep)
    if new:
        box = pd.concat([box] + new, ignore_index=True)
        box.to_csv(box_path, index=False)
    return games, box


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--season", type=int, default=2026, help="Sezon başlangıç yılı (2026 = 2026-27)")
    ap.add_argument("--prior-season", type=int, default=2025)
    args = ap.parse_args()
    update_season(args.prior_season)
    update_season(args.season)
