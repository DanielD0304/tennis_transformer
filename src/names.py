"""
Spielernamen aus verschiedenen Quellen auf einen gemeinsamen Schlüssel bringen.

Formate, die vorkommen:
    tennis-data.co.uk:  'Medvedev D.', 'Carreno Busta P.', 'Struff J.L.'
    Tipico:             'Tabilo, A.', 'Carreno-Busta, P.'
    bwin:               'D. Medvedev RUS', 'J.-L. Struff GER'
    sonstige:           'Daniil Medvedev'

Schlüssel ist (nachname, erster buchstabe des vornamens), alles klein, ohne Akzente,
Bindestriche als Leerzeichen. Bewusst ohne pandas/torch, damit der Scraper es auch nutzen kann.
"""

import re
import unicodedata

_COUNTRY_SUFFIX = re.compile(r"\s+[A-Z]{3}$")


def _clean(text):
    text = unicodedata.normalize("NFKD", str(text))
    text = "".join(c for c in text if not unicodedata.combining(c))
    return text.lower().replace("-", " ").strip()


def _is_initial(token):
    # 'd.', 'j.l.', 'j.', aber auch 'd' ohne Punkt
    return token.endswith(".") or len(token.replace(".", "")) <= 1


def player_key(name):
    """
    Liefert (nachname, initial) oder None, z.B. 'Tabilo, A.' -> ('tabilo', 'a').
    """
    if not name or not str(name).strip():
        return None
    name = _COUNTRY_SUFFIX.sub("", str(name).strip())

    if "," in name:
        surname, first = name.split(",", 1)
        surname = " ".join(_clean(surname).replace(".", "").split())
        first = _clean(first).replace(".", "").strip()
        return (surname, first[:1]) if surname else None

    # Bindestriche erst nach dem Initial-Check auflösen, sonst wird 'J.-L.' zu zwei Tokens
    tokens = [t for t in re.split(r"\s+", unicodedata.normalize("NFKD", name).lower()) if t]
    tokens = ["".join(c for c in t if not unicodedata.combining(c)) for t in tokens]
    if not tokens:
        return None

    if len(tokens) > 1 and _is_initial(tokens[-1]):
        # 'Medvedev D.' (tennis-data)
        initial = tokens[-1].replace(".", "").replace("-", "")[:1]
        surname_tokens = tokens[:-1]
    elif len(tokens) > 1 and _is_initial(tokens[0]):
        # 'D. Medvedev' / 'J.-L. Struff' (bwin)
        i = 0
        while i < len(tokens) - 1 and _is_initial(tokens[i]):
            i += 1
        initial = tokens[0].replace(".", "").replace("-", "")[:1]
        surname_tokens = tokens[i:]
    elif len(tokens) > 1:
        # 'Daniil Medvedev'
        initial = tokens[0][:1]
        surname_tokens = tokens[1:]
    else:
        initial = ""
        surname_tokens = tokens

    surname = " ".join(" ".join(surname_tokens).replace("-", " ").replace(".", "").split())
    return (surname, initial) if surname else None


def display_name(name):
    """Name ohne Länderkürzel (bwin hängt z.B. ' RUS' an)."""
    return _COUNTRY_SUFFIX.sub("", str(name).strip())
