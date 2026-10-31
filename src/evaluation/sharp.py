"""
Strategie A „Sharp vs. Rest“ auf Quoten-Snapshots (data/odds/snapshots, schema.load_snapshots).

Die faire Wahrscheinlichkeit kommt aus Pinnacle bzw. Betfair Exchange ohne Marge. Gewettet wird bei der Venue
mit dem besten Netto-Preis (Line Shopping über Polymarket, Kalshi, Tipico/bwin und die Buchmacher der Odds
API, jeweils nach Gebühr bzw. 5,3 % Wettsteuer), wenn p_fair * Netto-Quote - 1 > Schwelle. Eine Quote weit über den
anderen desselben Zeitpunkts zählt dabei nicht (_drop_outliers).

    from src.odds.schema import load_snapshots
    from src.evaluation import sharp
    signals = sharp.signals(load_snapshots('2026-10-01'))
    bets = sharp.first_signals(signals)

Verknüpfung: Dasselbe Match bei verschiedenen Venues wird über build.MatchIndex (Namen + Datum) erkannt
(link_events), die Snapshots über die Zeit mit merge_asof: ein Preis zählt nur, wenn er höchstens WINDOW
(5 Minuten) vom Sharp-Snapshot entfernt ist (align).
Staking: Viertel-Kelly mit Deckel (2 % der Bankroll), p vorher Richtung des Venue-Preises geschrumpft
(shrink_to_market); Flat läuft zum Vergleich mit.
"""
import numpy as np
import pandas as pd

from ..sources.build import ODDS_RANGE, MatchIndex
from ..sources.common import yyyymmdd
from .betting import Kelly
from .predictors import remove_margin
from .timing import PLACEHOLDER
from .venues import DE_BET_TAX, VENUES, Venue

SHARP = ('pinnacle', 'betfair')  # Reihenfolge = Priorität
# Plausibles Buch (Summe der Kehrwerte) einer Sharp-Quote, wie bei den historischen Exchange-Quoten: Pinnacle liegt
# bei ~1,03, Betfair nahe 1,00. Außerhalb ist der Markt dünn oder veraltet (z.B. 1,06/1,10) und kein fairer Preis.
SHARP_BOOK = ODDS_RANGE['BFEW']
WINDOW = pd.Timedelta(minutes=5)
DEFAULT_THRESHOLD = 0.02
SHRINK = 0.5  # Anteil, um den p für Kelly Richtung Venue-Preis geschrumpft wird
PM_SPREAD = 0.02  # Kalshi ohne Ask im Snapshot: Kauf zum Preis + halber Spread (Polymarket: kein Kauf, siehe buy_odds)
# Ausreißer beim Line Shopping (_drop_outliers), Abstand zum Median aller Quoten desselben Zeitpunkts. Aus der Verteilung
# von 5.934 Quoten in 182 Märkten (27. bis 30.10.2026), nicht aus dem Gewinn: In Punkten Wahrscheinlichkeit ist er über
# alle Quotenbereiche gleich (p99 3,1 bis 3,6) und halbiert sich je halben Punkt, das Maximum liegt bei 5,2, über 6 Punkten
# wäre weniger als eine Zeile normal. Als Quotient wächst er mit der Quote (p99 1,04 bis 1,34, Maximum 1,36), zählt also
# nur für lange Außenseiter, wo Punkte nicht mehr greifen: Bei Median 12 läge die Grenze sonst erst bei Quote 43.
MAX_GAP = 0.06
MAX_RATIO = 1.5
MIN_BOOKS = 6  # darunter trägt der Median nicht, Bücher teilen sich Feeds (williamhill/sport888, unibet_nl/casumo)
# Buchmacher ohne eigenes Profil: deutsche Wettsteuer auf den Einsatz, RET void (Annahme, je Buch prüfen)
DE_BOOK = Venue('de_book', price='none', stake_tax=DE_BET_TAX, ret_rule='void')
PROFILES = {
    'polymarket': VENUES['polymarket'],
    'kalshi': VENUES['kalshi'],
    'tipico': DE_BOOK.with_(name='tipico'),
    'bwin': DE_BOOK.with_(name='bwin'),
    'bet365': VENUES['bet365_de'],
    'betfair': VENUES['betfair'],
    'pinnacle': VENUES['pinnacle'],
}
PREDICTION_MARKETS = ('polymarket', 'kalshi')


def profile(venue):
    return PROFILES.get(venue) or DE_BOOK.with_(name=venue)


def shrink_to_market(p, p_market, weight=SHRINK):
    """p Richtung Marktpreis ziehen: (1 - weight) * p + weight * p_market."""
    return (1 - weight) * np.asarray(p, float) + weight * np.asarray(p_market, float)


# ----------------------------------------------------------------- Verknüpfung

def link_events(snaps):
    """
    Ein Match-Schlüssel je (venue, event_id): gleiche Matches verschiedener Venues bekommen denselben match_key.
    flip = True, wenn player_a des Events player_b des Schlüssel-Events ist. Zuerst die Sharp-Venues, damit deren
    Reihenfolge (A/B) die Referenz ist.
    """
    when = snaps['start_time_utc'].where(snaps['start_time_utc'].notna(), snaps['ts_utc'])
    events = snaps.assign(_when=when).drop_duplicates(['venue', 'event_id'])
    events = events.assign(_sharp=~events['venue'].isin(SHARP)).sort_values(['_sharp', 'ts_utc'], kind='stable')
    index = MatchIndex()
    keys, flips = {}, {}
    for r in events.to_dict('records'):
        day = yyyymmdd(r['_when'])
        q = {'source': 'snap', 'winner_name': r['player_a'], 'loser_name': r['player_b'], 'match_date': day,
             'tourney_date': day}
        other, same = index.find(q)
        if other is not None and other['_venue'] != r['venue']:
            keys[(r['venue'], r['event_id'])] = other['_key']
            flips[(r['venue'], r['event_id'])] = not same
            continue
        key = f"{r['venue']}:{r['event_id']}"
        keys[(r['venue'], r['event_id'])] = key
        flips[(r['venue'], r['event_id'])] = False
        index.add({**q, '_key': key, '_venue': r['venue']})
    pairs = list(zip(snaps['venue'], snaps['event_id']))
    return snaps.assign(match_key=[keys[p] for p in pairs], flip=[flips[p] for p in pairs])


def _oriented(df):
    """Preise/Quoten aus Sicht des Schlüssel-Events (A/B tauschen, wo flip)."""
    df = df.copy()
    f = df['flip'].to_numpy(bool)
    for x, y in (('odds_a', 'odds_b'), ('price_a', 'price_b'), ('best_ask_a', 'best_ask_b'), ('best_bid_a', 'best_bid_b')):
        a, b = df[x].to_numpy(float), df[y].to_numpy(float)
        df[x], df[y] = np.where(f, b, a), np.where(f, a, b)
    return df


def buy_odds(df, pm_spread=PM_SPREAD):
    """
    Bruttoquoten zum Kaufen (A, B), NaN = kein Kauf möglich. Buchmacher: die Quote. Prediction Markets: 1 / Ask, wobei
    ein Ask unter dem Preis (Mitte) nicht zählt, das kommt in keinem echten Orderbuch vor.
    Polymarket hat ohne brauchbares Ask keinen Kaufpreis: Preis + halber Spread wäre in leeren Orderbüchern (Bid 0,
    Ask 1) ein Preis, den es nie gab, und beim Line Shopping (signals) gewänne er gerade wegen seiner hohen Quote. Ein
    Preis von genau 0,5 ist wie im Timing-Backtest der Platzhalter eines leeren Orderbuchs. Ein Ask von genau 0,5 bei
    einer anderen Mitte ist dagegen ein echtes Buch (0,49/0,50). Kalshi kauft ohne Ask zum Preis + halben Spread.
    """
    pm = df['venue'].isin(PREDICTION_MARKETS).to_numpy()
    poly = (df['venue'] == 'polymarket').to_numpy()
    out = []
    for side in ('a', 'b'):
        ask = df[f'best_ask_{side}'].to_numpy(float)
        price = df[f'price_{side}'].to_numpy(float)
        with np.errstate(invalid='ignore'):
            usable = np.isfinite(ask) & (ask > 0) & (ask < 1) & ~(ask < price)
        buy = np.where(usable, ask, np.where(poly, np.nan, price + pm_spread / 2))
        buy = np.where(poly & (price == PLACEHOLDER), np.nan, buy)
        with np.errstate(divide='ignore', invalid='ignore'):
            out.append(np.where(pm, np.where((buy > 0) & (buy < 1), 1 / buy, np.nan), df[f'odds_{side}'].to_numpy(float)))
    return out


def sharp_fair(df):
    """Faire Wahrscheinlichkeit für A aus den Sharp-Snapshots (ohne Marge), NaN bei unplausiblem Buch (SHARP_BOOK)."""
    odds_a, odds_b = df['odds_a'].to_numpy(float), df['odds_b'].to_numpy(float)
    with np.errstate(divide='ignore', invalid='ignore'):
        book = 1 / odds_a + 1 / odds_b
    low, high = SHARP_BOOK
    return np.where((book >= low) & (book <= high), remove_margin(odds_a, odds_b), np.nan)


def align(snaps, window=WINDOW, pm_spread=PM_SPREAD):
    """
    Pro Sharp-Snapshot (Pinnacle vor Betfair, je Match und Zeitpunkt) und Venue der nächste Snapshot derselben
    Venue innerhalb von ±window. Ergebnis: eine Zeile pro (Sharp-Snapshot, Venue) mit ts_utc, match_key,
    sharp_venue, p_fair (A), venue, gross_a/gross_b (Kaufquoten), start_time_utc, Spielern, tour, level.
    A/B und Preise sind aus Sicht des Sharp-Events; venue_flip/venue_player_a/venue_player_b beschreiben das
    Event der Venue (venue_flip: dort sind A und B vertauscht).
    """
    snaps = _oriented(link_events(snaps))
    sharp = snaps[snaps['venue'].isin(SHARP)].copy()
    if sharp.empty:
        return pd.DataFrame()
    sharp['p_fair'] = sharp_fair(sharp)
    sharp['_rank'] = sharp['venue'].map({v: i for i, v in enumerate(SHARP)})
    sharp = sharp[np.isfinite(sharp['p_fair'])].sort_values(['match_key', 'ts_utc', '_rank'])
    sharp = sharp.drop_duplicates(['match_key', 'ts_utc'])
    left = sharp[['ts_utc', 'match_key', 'venue', 'p_fair', 'player_a', 'player_b', 'start_time_utc', 'tour']] \
        .rename(columns={'venue': 'sharp_venue'}).sort_values('ts_utc')
    rest = snaps[~snaps['venue'].isin(SHARP)].copy()
    rest['gross_a'], rest['gross_b'] = buy_odds(rest, pm_spread)
    rest['implied_a'] = rest['price_a'].where(rest['price_type'] == 'probability', 1 / rest['odds_a'])
    out = []
    for venue, g in rest.groupby('venue'):
        right = g[['ts_utc', 'match_key', 'event_id', 'gross_a', 'gross_b', 'level', 'start_time_utc', 'flip',
                   'player_a', 'player_b', 'tourney']] \
            .rename(columns={'ts_utc': 'venue_ts', 'start_time_utc': 'venue_start', 'flip': 'venue_flip',
                             'player_a': 'venue_player_a', 'player_b': 'venue_player_b'}).sort_values('venue_ts')
        m = pd.merge_asof(left, right, left_on='ts_utc', right_on='venue_ts', by='match_key', tolerance=window,
                          direction='nearest')
        m = m[m['venue_ts'].notna()]
        out.append(m.assign(venue=venue))
    if not out:
        return pd.DataFrame()
    df = pd.concat(out, ignore_index=True)
    df['start_time_utc'] = df['start_time_utc'].fillna(df['venue_start'])
    return df.drop(columns=['venue_start'])


# --------------------------------------------------------------------- Signale

def _drop_outliers(cand):
    """
    Kandidaten ohne Ausreißer. Beim Line Shopping gewinnt die höchste Quote, also auch eine veraltete oder falsche (27.10.:
    10,0 bei William Hill und 888 gegen Median 5,75, acht Stunden später 6,5). Raus fällt je Sharp-Snapshot und Seite, was
    mehr als MAX_GAP Wahrscheinlichkeit oder mehr als das MAX_RATIO-Fache über dem Median der Bruttoquoten liegt, sobald es
    MIN_BOOKS Quoten gibt. Die nächste Quote kann weiter ein Signal sein.
    """
    quotes = cand.groupby(['match_key', 'ts_utc', 'side'])['odds']
    median, books = quotes.transform('median'), quotes.transform('size')
    out = (1 / median - 1 / cand['odds'] > MAX_GAP) | (cand['odds'] / median > MAX_RATIO)
    return cand[~(out & (books >= MIN_BOOKS))]


def signals(snaps, threshold=DEFAULT_THRESHOLD, window=WINDOW, pm_spread=PM_SPREAD, before_start=True):
    """
    Line Shopping: pro Sharp-Snapshot und Seite die Venue mit der besten Netto-Quote ohne Ausreißer (_drop_outliers),
    Signal wenn p_fair * netto - 1 > threshold. before_start: nur Preise vor der Startzeit (wo bekannt).
    Spalten: ts_utc, match_key, side ('a'/'b'), player, venue, event_id, odds (brutto), net_odds, p_fair, edge, ...
    """
    df = align(snaps, window, pm_spread)
    if df.empty:
        return pd.DataFrame(columns=['ts_utc', 'match_key', 'side', 'player', 'venue', 'event_id', 'odds', 'net_odds',
                                     'p_fair', 'edge'])
    if before_start:
        df = df[df['start_time_utc'].isna() | (df['ts_utc'] < df['start_time_utc'])]
    rows = []
    for side in ('a', 'b'):
        gross = df[f'gross_{side}'].to_numpy(float)
        net = np.full(len(df), np.nan)
        for venue in df['venue'].unique():
            m = (df['venue'] == venue).to_numpy()
            net[m] = profile(venue).net_odds_from_gross(gross[m])
        p = df['p_fair'].to_numpy(float) if side == 'a' else 1 - df['p_fair'].to_numpy(float)
        part = df.assign(side=side, player=df[f'player_{side}'], odds=gross, net_odds=net, p_fair=p,
                         edge=p * net - 1)
        rows.append(part)
    cand = pd.concat(rows, ignore_index=True)
    cand = _drop_outliers(cand[np.isfinite(cand['edge'])])
    # Line Shopping: beste Netto-Quote je Sharp-Snapshot und Seite
    best = cand.sort_values('net_odds', ascending=False).drop_duplicates(['match_key', 'ts_utc', 'side'])
    return best[best['edge'] > threshold].sort_values('ts_utc').reset_index(drop=True)


def first_signals(sig):
    """Eine Wette je Match und Seite: das erste Signal."""
    return sig.sort_values('ts_utc', kind='stable').drop_duplicates(['match_key', 'side']).reset_index(drop=True)


def stake_sizes(bets, bankroll=1000.0, kelly=Kelly(0.25, 0.02), shrink=SHRINK, flat=10.0):
    """Einsatz Viertel-Kelly (Deckel 2 %) mit geschrumpftem p und flat zum Vergleich (feste Bankroll)."""
    implied = 1 / bets['odds'].to_numpy(float)
    p = shrink_to_market(bets['p_fair'].to_numpy(float), implied, shrink)
    kelly_stake = [kelly.amount(bankroll, pi, o) for pi, o in zip(p, bets['net_odds'].to_numpy(float))]
    return bets.assign(p_shrunk=p, stake_kelly=kelly_stake, stake_flat=flat)


def closing_fair(snaps):
    """Letzte faire Sharp-Wahrscheinlichkeit (A) je match_key vor der Startzeit."""
    snaps = _oriented(link_events(snaps))
    sharp = snaps[snaps['venue'].isin(SHARP)]
    sharp = sharp[sharp['start_time_utc'].isna() | (sharp['ts_utc'] < sharp['start_time_utc'])]
    sharp = sharp.assign(p_close=sharp_fair(sharp)).dropna(subset=['p_close']).sort_values('ts_utc')
    return sharp.drop_duplicates('match_key', keep='last').set_index('match_key')['p_close']


def evaluate(bets, outcomes, close=None):
    """
    bets (first_signals + stake_sizes), outcomes: Series match_key -> 1 (A gewinnt), 0, NaN (offen) und optional
    status ('W/O' void, 'RET' nach Regel der Venue) als DataFrame mit Spalten a_won, status.
    Gibt die Wetten mit profit_flat, profit_kelly, clv (gegen closing_fair) zurück.
    """
    out = bets.copy()
    if isinstance(outcomes, pd.Series):
        outcomes = outcomes.to_frame('a_won')
    res = outcomes.reindex(out['match_key'])
    a_won = res['a_won'].to_numpy(float)
    status = res['status'].fillna('completed').to_numpy(object) if 'status' in res else np.full(len(out), 'completed')
    won = np.where(out['side'] == 'a', a_won == 1, a_won == 0)
    void = np.array([profile(v).is_void([s])[0] for v, s in zip(out['venue'], status)], dtype=bool)
    settled = np.isfinite(a_won)
    for kind in ('flat', 'kelly'):
        stake = out[f'stake_{kind}'].to_numpy(float)
        profit = np.where(won, stake * (out['net_odds'] - 1), -stake)
        out[f'profit_{kind}'] = np.where(settled & ~void, profit, 0.0)
    out['void'], out['settled'], out['won'] = void, settled, won & settled & ~void
    if close is not None:
        pc = close.reindex(out['match_key']).to_numpy(float)
        p_side = np.where(out['side'] == 'a', pc, 1 - pc)
        out['clv'] = p_side * out['odds'].to_numpy(float) - 1
    return out
