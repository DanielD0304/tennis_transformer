"""
Spielernamen aus verschiedenen Quellen auf einen gemeinsamen Schlüssel bringen.

Formate, die vorkommen:
    tennis-data.co.uk:  'Medvedev D.', 'Carreno Busta P.', 'Struff J.L.', 'Tirante T. A.'
    Tipico:             'Tabilo, A.', 'Carreno-Busta, P.'
    bwin:               'D. Medvedev RUS', 'J.-L. Struff GER'
    Sackmann/ESPN:      'Daniil Medvedev', "Christopher O'Connell", 'Wu Yibing'

Schlüssel ist (nachname, erster buchstabe des vornamens), alles klein, ohne Akzente,
Apostrophe, Bindestriche und Leerzeichen im Nachnamen ('carrenobusta', 'p').
Bewusst ohne pandas/torch, damit Scraper und Daten-Update es auch nutzen können.
"""

import re
import unicodedata

_COUNTRY_SUFFIX = re.compile(r"\s+[A-Z]{3}$")


def _ascii(text):
    text = unicodedata.normalize("NFKD", str(text))
    return "".join(c for c in text if not unicodedata.combining(c)).lower()


def _is_initial(token):
    # 'd.', 'j.l.', 'j.-l.', aber auch 'd' ohne Punkt
    return token.endswith(".") or len(token.replace(".", "").replace("-", "")) <= 1


def _surname_tokens(tokens):
    """Nachnamen-Tokens bereinigen: Bindestriche trennen, Punkte/Apostrophe entfernen."""
    out = []
    for t in tokens:
        for part in t.replace("-", " ").split():
            part = re.sub(r"[.'’`]", "", part)
            if part:
                out.append(part)
    return out


def _parse(name):
    """-> (nachnamen_tokens, initial, art) oder None. art: 'comma', 'abbr', 'full', 'single'."""
    if not name or not str(name).strip():
        return None
    name = _COUNTRY_SUFFIX.sub("", str(name).strip())

    if "," in name:
        surname, first = name.split(",", 1)
        first = re.sub(r"[^a-z]", "", _ascii(first))
        return _surname_tokens(_ascii(surname).split()), first[:1], 'comma'

    tokens = _ascii(name).split()
    if not tokens:
        return None
    if len(tokens) > 1 and _is_initial(tokens[-1]):
        # 'Medvedev D.', 'Tirante T. A.' (tennis-data): erstes der Initialen am Ende
        i = len(tokens)
        while i > 1 and _is_initial(tokens[i - 1]):
            i -= 1
        return _surname_tokens(tokens[:i]), re.sub(r"[^a-z]", "", tokens[i])[:1], 'abbr'
    if len(tokens) > 1 and _is_initial(tokens[0]):
        # 'D. Medvedev', 'J.-L. Struff', 'J.J. Wolf'
        i = 0
        while i < len(tokens) - 1 and _is_initial(tokens[i]):
            i += 1
        return _surname_tokens(tokens[i:]), re.sub(r"[^a-z]", "", tokens[0])[:1], 'abbr'
    if len(tokens) > 1:
        # 'Daniil Medvedev'
        return _surname_tokens(tokens[1:]), tokens[0][:1], 'full'
    return _surname_tokens(tokens), "", 'single'


def player_key(name):
    """
    Liefert (nachname, initial) oder None, z.B. 'Tabilo, A.' -> ('tabilo', 'a').
    """
    parsed = _parse(name)
    if not parsed or not parsed[0]:
        return None
    surname, initial, _ = parsed
    return "".join(surname), initial


def candidate_keys(name, reverse=True):
    """
    Alle plausiblen Schlüssel eines Namens, der erste ist player_key(name).

    Bei vollen Namen ist unklar, wo der Nachname beginnt und endet:
        'Juan Manuel Cerundolo'       -> auch ('cerundolo', 'j')      (tennis-data: 'Cerundolo J.M.')
        'Giovanni Mpetshi Perricard'  -> auch ('mpetshi', 'g')        (tennis-data: 'Mpetshi G.')
        'Wu Yibing'                   -> auch ('wu', 'y')             (chinesische Reihenfolge)

    reverse=False lässt die umgekehrte Reihenfolge weg. Die ist unsicher ('Novak Djokovic' ergäbe
    ('novak', 'd') wie 'Dennis Novak') und taugt nur, wenn noch andere Hinweise dazukommen.
    """
    parsed = _parse(name)
    if not parsed or not parsed[0]:
        return []
    surname, initial, kind = parsed
    keys = [("".join(surname), initial)]
    if kind == 'full':
        for i in range(1, len(surname)):
            keys.append(("".join(surname[i:]), initial))
        if len(surname) > 1:
            keys.append((surname[0], initial))
        first = _ascii(_COUNTRY_SUFFIX.sub("", str(name).strip())).split()
        if reverse and len(first) == 2:
            keys.append(("".join(_surname_tokens(first[:1])), first[1][:1]))
    out = []
    for k in keys:
        if k not in out:
            out.append(k)
    return out


def display_name(name):
    """Name ohne Länderkürzel (bwin hängt z.B. ' RUS' an)."""
    return _COUNTRY_SUFFIX.sub("", str(name).strip())
