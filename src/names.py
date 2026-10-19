"""Fantasy uygulamasındaki isimleri EuroLeague box score isimleriyle eşleştirir.

Box score isimleri "SOYAD, AD" formatında ve büyük harfle gelir; fantasy
uygulaması "Ad Soyad" yazıyor ve Jr/IV gibi ekleri tutarsız kullanıyor.
"""
from __future__ import annotations

import difflib
import re
import unicodedata

SUFFIXES = {"jr", "sr", "ii", "iii", "iv", "v"}


def norm(name: str) -> str:
    s = str(name)
    if "," in s:  # "BACON, DWAYNE" -> "DWAYNE BACON"
        last, first = s.split(",", 1)
        s = f"{first} {last}"
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode()
    s = s.lower().replace(".", "").replace("'", "").replace("-", " ")
    toks = [t for t in re.split(r"\s+", s) if t and t not in SUFFIXES]
    return " ".join(toks)


def build_matcher(boxscore_names: dict[str, str]):
    """boxscore_names: player_id -> ham isim. Fonksiyon döndürür: fantasy adı -> player_id."""
    index = {}
    for pid, raw in boxscore_names.items():
        index.setdefault(norm(raw), pid)
    keys = list(index)

    def match(fantasy_name: str) -> tuple[str | None, float]:
        n = norm(fantasy_name)
        if n in index:
            return index[n], 1.0
        # "aj lawson" vs "a j lawson" gibi boşluk farkları
        compact = n.replace(" ", "")
        for k in keys:
            if k.replace(" ", "") == compact:
                return index[k], 0.99
        best = difflib.get_close_matches(n, keys, n=1, cutoff=0.82)
        if best:
            return index[best[0]], difflib.SequenceMatcher(None, n, best[0]).ratio()
        # soyad + adın baş harfi
        parts = n.split()
        if len(parts) >= 2:
            cands = [k for k in keys if k.split()[-1] == parts[-1] and k[0] == n[0]]
            if len(cands) == 1:
                return index[cands[0]], 0.8
        return None, 0.0

    return match


def pretty(raw: str) -> str:
    """'JAMES, MIKE' -> 'Mike James'"""
    s = str(raw)
    if "," in s:
        last, first = s.split(",", 1)
        s = f"{first.strip()} {last.strip()}"
    return " ".join(w.capitalize() if w.isupper() else w for w in s.split())
