"""
Schema der Quoten-Snapshots und Ablage als Parquet (ein File pro Tag, UTC).

Ein Snapshot ist der Preis eines Matches bei einer Venue zu einem Zeitpunkt. price_a/price_b sind der
native Preis der Venue: bei Buchmachern die Dezimalquote (price_type 'decimal'), bei Prediction
Markets der Preis in 0–1 (price_type 'probability'). odds_a/odds_b sind immer Dezimalquoten
(bei PM 1/Preis, ohne Gebühren; die rechnet src/evaluation/venues.py).

Venue-Namen passen zu src/evaluation/venues.py ('pinnacle', 'betfair', 'kalshi', 'polymarket').
Buchmacher ohne eigenes Profil heißen wie bei der Quelle ('bet365', 'tipico', 'bwin', ...);
die DE-Steuer kommt erst im Venue-Profil dazu (bet365 -> bet365_de).
"""
import math
import os
from dataclasses import asdict, dataclass, fields
from datetime import date, datetime, timezone

from ..sources.common import DATA_DIR

ODDS_DIR = os.path.join(DATA_DIR, "odds")
SNAPSHOT_DIR = os.path.join(ODDS_DIR, "snapshots")
HISTORY_DIR = os.path.join(ODDS_DIR, "history")

PRICE_TYPES = ('decimal', 'probability')
TOURS = ('atp', 'wta', None)


@dataclass
class Snapshot:
    ts_utc: datetime
    venue: str
    event_id: str  # venue-intern (Polymarket: Markt-ID, Kalshi: Event-Ticker, Odds API: Event-ID)
    player_a: str
    player_b: str
    price_a: float
    price_b: float
    price_type: str = 'decimal'
    tour: str = None  # 'atp' | 'wta' | None (unbekannt)
    level: str = None  # 'main' | 'challenger' | 'itf' | None
    tourney: str = None
    start_time_utc: datetime = None
    odds_a: float = None  # wird in __post_init__ aus price_* abgeleitet
    odds_b: float = None
    best_bid_a: float = None  # nur Prediction Markets / Exchanges
    best_ask_a: float = None
    best_bid_b: float = None
    best_ask_b: float = None
    volume: float = None
    liquidity: float = None
    source_url: str = None

    def __post_init__(self):
        if self.odds_a is None:
            self.odds_a = to_decimal(self.price_a, self.price_type)
        if self.odds_b is None:
            self.odds_b = to_decimal(self.price_b, self.price_type)


COLUMNS = [f.name for f in fields(Snapshot)]
DATETIME_COLUMNS = ['ts_utc', 'start_time_utc']
FLOAT_COLUMNS = ['price_a', 'price_b', 'odds_a', 'odds_b', 'best_bid_a', 'best_ask_a', 'best_bid_b',
                 'best_ask_b', 'volume', 'liquidity']


# ------------------------------------------------------------ Preis <-> Quote

def _num(x):
    try:
        x = float(x)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(x) else x


def price_to_odds(p):
    """PM-Preis (0–1) -> Dezimalquote. 0 oder ungültig -> None."""
    p = _num(p)
    return 1 / p if p is not None and 0 < p <= 1 else None


def odds_to_price(odds):
    """Dezimalquote -> implizite Wahrscheinlichkeit (mit Marge). Quote <= 1 -> None."""
    odds = _num(odds)
    return 1 / odds if odds is not None and odds > 1 else None


def to_decimal(price, price_type):
    if price_type == 'probability':
        return price_to_odds(price)
    price = _num(price)
    return price if price is not None and price > 1 else None


def mid(bid, ask):
    """Mitte aus Bid/Ask, nur wenn beide eine echte Seite haben (0 < bid <= ask < 1)."""
    bid, ask = _num(bid), _num(ask)
    if bid is None or ask is None or not (0 < bid <= ask < 1):
        return None
    return (bid + ask) / 2


def utc(value):
    """ISO-String ('2026-10-25T13:00:00Z', '2026-10-25 13:00:00+00'), Unix-Sekunden oder datetime -> datetime (UTC)."""
    if value is None or value == "":
        return None
    if isinstance(value, datetime):
        return value.replace(tzinfo=timezone.utc) if value.tzinfo is None else value.astimezone(timezone.utc)
    if isinstance(value, (int, float)):
        return datetime.fromtimestamp(value, timezone.utc)
    s = str(value).strip().replace("Z", "+00:00")
    if s.endswith("+00"):
        s += ":00"
    dt = datetime.fromisoformat(s)
    return dt.replace(tzinfo=timezone.utc) if dt.tzinfo is None else dt.astimezone(timezone.utc)


def now_utc():
    return datetime.now(timezone.utc).replace(microsecond=0)


# ---------------------------------------------------------------- Validierung

def validate(snapshot):
    """Wirft ValueError bei einem ungültigen Snapshot, sonst wird er unverändert zurückgegeben."""
    s = snapshot
    if not isinstance(s.ts_utc, datetime) or s.ts_utc.tzinfo is None:
        raise ValueError("ts_utc muss ein datetime mit Zeitzone (UTC) sein")
    if s.start_time_utc is not None and (not isinstance(s.start_time_utc, datetime) or s.start_time_utc.tzinfo is None):
        raise ValueError("start_time_utc muss None oder ein datetime mit Zeitzone sein")
    for name in ('venue', 'event_id', 'player_a', 'player_b'):
        if not getattr(s, name) or not str(getattr(s, name)).strip():
            raise ValueError(f"{name} fehlt")
    if s.player_a == s.player_b:
        raise ValueError("player_a und player_b sind gleich")
    if s.price_type not in PRICE_TYPES:
        raise ValueError(f"price_type muss in {PRICE_TYPES} sein, nicht {s.price_type!r}")
    if s.tour not in TOURS:
        raise ValueError(f"tour muss 'atp', 'wta' oder None sein, nicht {s.tour!r}")
    for side in ('a', 'b'):
        price = _num(getattr(s, f'price_{side}'))
        if price is None:
            continue
        if s.price_type == 'probability' and not 0 <= price <= 1:
            raise ValueError(f"price_{side}={price} liegt nicht in [0, 1]")
        if s.price_type == 'decimal' and price <= 1:
            raise ValueError(f"price_{side}={price} ist keine Dezimalquote (> 1)")
    if _num(s.price_a) is None and _num(s.price_b) is None:
        raise ValueError("weder price_a noch price_b vorhanden")
    return s


def clean(snapshots, log=print):
    """Ungültige Snapshots verwerfen (mit Hinweis), statt den ganzen Lauf abzubrechen."""
    out = []
    for s in snapshots:
        try:
            out.append(validate(s))
        except ValueError as e:
            log(f"  Snapshot verworfen ({s.venue} {s.event_id}): {e}")
    return out


# ------------------------------------------------------------------ Ablage

def to_frame(snapshots):
    import pandas as pd
    df = pd.DataFrame([asdict(s) for s in snapshots], columns=COLUMNS)
    return normalize_frame(df)


def normalize_frame(df):
    import pandas as pd
    for c in DATETIME_COLUMNS:
        df[c] = pd.to_datetime(df[c], utc=True)
    for c in FLOAT_COLUMNS:
        df[c] = pd.to_numeric(df[c], errors='coerce').astype(float)
    for c in COLUMNS:
        if c not in DATETIME_COLUMNS + FLOAT_COLUMNS:
            df[c] = df[c].astype(object).where(df[c].notna(), None)
    return df[COLUMNS]


def day_path(day, root=SNAPSHOT_DIR):
    return os.path.join(root, f"{day:%Y-%m-%d}.parquet")


def append_snapshots(snapshots, root=SNAPSHOT_DIR):
    """Hängt Snapshots an die Tagesdatei (nach ts_utc) an. Gibt die Anzahl geschriebener Zeilen zurück."""
    import pandas as pd
    snapshots = [validate(s) for s in snapshots]
    if not snapshots:
        return 0
    df = to_frame(snapshots)
    os.makedirs(root, exist_ok=True)
    for day, part in df.groupby(df['ts_utc'].dt.date):
        path = day_path(day, root)
        if os.path.exists(path):
            part = pd.concat([pd.read_parquet(path), part], ignore_index=True)
        tmp = path + ".tmp"
        normalize_frame(part).to_parquet(tmp, index=False)
        os.replace(tmp, path)
    return len(df)


def _as_date(value):
    if value is None:
        return None
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    return date.fromisoformat(str(value)[:10])


def load_snapshots(start=None, end=None, root=SNAPSHOT_DIR, venues=None):
    """Alle Snapshots mit start <= Tag(ts_utc) <= end (Datum oder 'YYYY-MM-DD', None = offen)."""
    import pandas as pd
    start, end = _as_date(start), _as_date(end)
    frames = []
    if os.path.isdir(root):
        for name in sorted(os.listdir(root)):
            if not name.endswith(".parquet"):
                continue
            try:
                day = date.fromisoformat(name[:10])
            except ValueError:
                continue
            if (start and day < start) or (end and day > end):
                continue
            frames.append(pd.read_parquet(os.path.join(root, name)))
    if not frames:
        return normalize_frame(pd.DataFrame(columns=COLUMNS))
    df = normalize_frame(pd.concat(frames, ignore_index=True))
    if venues:
        df = df[df['venue'].isin(venues)]
    return df.sort_values('ts_utc').reset_index(drop=True)

