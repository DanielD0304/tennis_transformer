"""
Erste Auswertung ohne Pinnacle: Wie gut sind die Closing-Preise von Polymarket und Kalshi?

    python -m src.odds.report
    python -m src.odds.report --min-volume 1000

Für jede Venue: Anzahl Märkte, Zeitraum, Volumen, Spread, Log Loss, Brier, ECE und Kalibrierungstabelle
der Closing- und Eröffnungspreise gegen das Ergebnis aus der Markt-Auflösung. Dazu der direkte
Vergleich auf gemeinsamen Matches beider Venues, die Abgleich-Quote gegen data/matches.csv und, wo
matches.csv Pinnacle-Schlussquoten hat (tennis-data, nur ATP), der Vergleich mit Pinnacle ohne Marge.
Es werden nur Märkte mit Sieger ausgewertet (50-50-Auflösungen bei Absage/W/O fallen raus).
"""
import argparse

import numpy as np
import pandas as pd

from ..evaluation import metrics
from ..evaluation.predictors import remove_margin
from ..evaluation.venues import get_venue
from .history import HISTORY_DIR, read_history
from .matching import link_matches, load_matches, pair_venues
from .schema import SNAPSHOT_DIR, load_snapshots

VENUES = ('polymarket', 'kalshi')
EXTREME = 0.03  # Closing-Preis unter 3 % oder über 97 %: Verdacht auf Live-Handel vor der geplanten Startzeit
# Liegt der letzte Preis mehr als 2 h vor der geplanten Startzeit, wurde der Markt meist schon vorher
# aufgelöst (Startzeit nachträglich verschoben): Preis ist dann das Ergebnis, nicht die Prognose
MAX_STALE = pd.Timedelta(hours=2)


def stale(df):
    lag = pd.to_datetime(df['start_time_utc'], utc=True) - pd.to_datetime(df['close_ts'], utc=True)
    return lag > MAX_STALE  # ohne close_ts (NaT): nicht veraltet


def evaluable(df, min_volume=0.0):
    df = df[df['winner'].isin(['a', 'b']) & df['close_price_a'].notna() & ~stale(df)]
    if min_volume:
        df = df[df['volume'].fillna(0) >= min_volume]
    return df.assign(y=(df['winner'] == 'a').astype(float))


def overview(df, venue):
    ev = evaluable(df)
    spread = (df['close_ask_a'] - df['close_bid_a']).dropna()
    return {
        'venue': venue,
        'märkte': len(df),
        'mit_sieger': int(df['winner'].isin(['a', 'b']).sum()),
        'ausgewertet': len(ev),
        'closing_veraltet': int(stale(df).sum()),
        'von': df['start_time_utc'].min(),
        'bis': df['start_time_utc'].max(),
        'volumen_median': float(df['volume'].median()) if df['volume'].notna().any() else np.nan,
        'volumen_summe': float(df['volume'].sum()) if df['volume'].notna().any() else np.nan,
        'spread_median': float(spread.median()) if len(spread) else np.nan,
        'closing_extrem': float(((ev['close_price_a'] < EXTREME) | (ev['close_price_a'] > 1 - EXTREME)).mean()) if len(ev) else np.nan,
    }


def score_rows(ev, label):
    rows = []
    for name, col in (('closing', 'close_price_a'), ('closing-1h', 'close_1h_price_a'), ('eröffnung', 'open_price_a')):
        sample = ev[ev[col].notna()]
        if len(sample):
            rows.append({'gruppe': label, 'preis': name, **metrics.summarize(sample['y'], sample[col])})
    return rows


def scores(df, venue, min_volume=0.0):
    ev = evaluable(df, min_volume)
    rows = score_rows(ev, f"{venue} gesamt")
    for (tour, level), g in ev.groupby([ev['tour'].fillna('?'), ev['level'].fillna('?')]):
        if len(g) >= 30:
            rows += score_rows(g, f"{venue} {tour}/{level}")
    return pd.DataFrame(rows)


def head_to_head(pm, ks, min_volume=0.0):
    """Gemeinsame Matches: Closing beider Venues aus Sicht desselben Spielers, gleiche Stichprobe."""
    pm, ks = evaluable(pm, min_volume), evaluable(ks, min_volume)
    pairs = pair_venues(pm, ks)
    if pairs.empty:
        return pairs, pd.DataFrame()
    m = pairs.merge(pm[['event_id', 'close_price_a', 'y']], left_on='left_event_id', right_on='event_id') \
             .merge(ks[['event_id', 'close_price_a', 'y']], left_on='right_event_id', right_on='event_id',
                    suffixes=('_pm', '_ks'))
    m['p_ks'] = np.where(m['same_order'], m['close_price_a_ks'], 1 - m['close_price_a_ks'])
    m['y_ks'] = np.where(m['same_order'], m['y_ks'], 1 - m['y_ks'])
    agree = float((m['y_pm'] == m['y_ks']).mean())
    same = m[m['y_pm'] == m['y_ks']]
    rows = [{'venue': 'polymarket', **metrics.summarize(same['y_pm'], same['close_price_a_pm'])},
            {'venue': 'kalshi', **metrics.summarize(same['y_pm'], same['p_ks'])},
            {'venue': 'mittel', **metrics.summarize(same['y_pm'], (same['close_price_a_pm'] + same['p_ks']) / 2)}]
    out = pd.DataFrame(rows)
    out.attrs['ergebnis_gleich'] = agree
    out.attrs['mittlere_abweichung'] = float((same['close_price_a_pm'] - same['p_ks']).abs().mean())
    return m, out


def vs_pinnacle(df, venue, matches, min_volume=0.0, price_col='close_price_a', bench='ps'):
    """
    Venue-Preis gegen Pinnacle-Closing (bench='ps') bzw. Ø-Schlussquote (bench='avg'), beide ohne Marge,
    auf denselben Matches: Log Loss, Brier, gepaarter
    Bootstrap der Log-Loss-Differenz, mittlere Preisabweichung und Anteil mit Edge nach Gebühren
    (faire Pinnacle-Wahrscheinlichkeit × Netto-Quote der Venue - 1 > 0, gekauft zum Ask, sonst zur Mitte).
    Nur Matches, deren Sieger laut Markt-Auflösung und matches.csv übereinstimmt.
    """
    ev = evaluable(df, min_volume)
    ev = ev[ev[price_col].notna()]
    if ev.empty or not matches:
        return None
    m = link_matches(ev, matches)
    m = m[m['match_found'] & m[f'{bench}_a'].notna() & m[f'{bench}_b'].notna()]
    m = m[(m['y'] == 1) == m['a_won'].astype(bool)]
    if m.empty:
        return None
    y, p = m['y'].to_numpy(float), m[price_col].to_numpy(float)
    p_pin = remove_margin(m[f'{bench}_a'].to_numpy(float), m[f'{bench}_b'].to_numpy(float))
    loss_v, loss_p = metrics.pointwise_log_loss(y, p), metrics.pointwise_log_loss(y, p_pin)
    boot = metrics.paired_bootstrap(loss_v, loss_p)

    profile = get_venue(venue)
    # Kauf zum Ask, wo es eins gibt (Kalshi-Closing), sonst Mitte plus halber Spread des Venue-Profils
    half = profile.spread / 2 if profile.spread else 0.01
    has_ba = (m['close_ask_a'].notna() & m['close_bid_a'].notna()).to_numpy() if price_col == 'close_price_a' else False
    cost_a = np.where(has_ba, m['close_ask_a'], p + half)
    cost_b = np.where(has_ba, 1 - m['close_bid_a'], 1 - p + half)
    ok = (cost_a > 0.01) & (cost_a < 0.99) & (cost_b > 0.01) & (cost_b < 0.99)
    net_a = profile.net_odds_from_gross(1 / np.clip(cost_a, 0.01, 0.99))
    net_b = profile.net_odds_from_gross(1 / np.clip(cost_b, 0.01, 0.99))
    ev_a, ev_b = p_pin * net_a - 1, (1 - p_pin) * net_b - 1
    side_a = ev_a >= ev_b
    edge = np.where(side_a, ev_a, ev_b)
    bet = ok & (edge > 0)
    won = np.where(side_a, y == 1, y == 0)
    profit = np.where(won, np.where(side_a, net_a, net_b) - 1, -1.0)
    return {'venue': venue, 'preis': price_col.replace('_price_a', ''), 'gegen': bench, 'n': len(m),
            'll_venue': float(loss_v.mean()), 'll_bench': float(loss_p.mean()),
            'brier_venue': metrics.brier(y, p), 'brier_bench': metrics.brier(y, p_pin),
            'll_diff': boot['mean_diff'], 'ci_low': boot['ci_low'], 'ci_high': boot['ci_high'],
            'p_boot': boot['p_value'], 'mittel_abs_dp': float(np.abs(p - p_pin).mean()),
            'anteil_edge': float(bet.mean()), 'wetten': int(bet.sum()),
            'edge_mittel': float(edge[bet].mean()) if bet.any() else np.nan,
            'yield_flat': float(profit[bet].mean()) if bet.any() else np.nan}


def matching_table(frames, matches):
    rows = []
    for venue, df in frames.items():
        linked = link_matches(df, matches)
        for (tour, level), g in linked.groupby([linked['tour'].fillna('?'), linked['level'].fillna('?')]):
            hit = g[g['match_found']]
            with_winner = hit[hit['winner'].isin(['a', 'b'])]
            agree = ((with_winner['winner'] == 'a') == with_winner['a_won'].astype(bool)).mean() if len(with_winner) else np.nan
            rows.append({'venue': venue, 'tour': tour, 'level': level, 'n': len(g), 'gefunden': len(hit),
                         'quote': len(hit) / len(g) if len(g) else np.nan, 'sieger_stimmt': agree})
    return pd.DataFrame(rows)


def _fmt(df):
    return df.to_string(index=False, float_format=lambda x: f"{x:.4f}") if len(df) else "  (keine Daten)"


def report(min_volume=0.0, history_root=HISTORY_DIR, snapshot_root=SNAPSHOT_DIR, matches=None, log=print):
    frames = {v: read_history(v, history_root) for v in VENUES}
    frames = {v: df for v, df in frames.items() if len(df)}
    if not frames:
        log("Keine Historie gefunden. Erst laden: python -m src.odds.collect --history")
        return {}

    log("=== Historische Match-Märkte ===")
    log(_fmt(pd.DataFrame([overview(df, v) for v, df in frames.items()])))
    log(f"(closing_extrem = Anteil Closing-Preise < {EXTREME:.0%} oder > {1 - EXTREME:.0%}: "
        "Hinweis auf Live-Preise, weil die Startzeit nur geplant ist)")

    out = {}
    for venue, df in frames.items():
        log(f"\n=== {venue}: Log Loss / Brier / ECE (Volumen >= {min_volume:g}) ===")
        table = scores(df, venue, min_volume)
        log(_fmt(table))
        ev = evaluable(df, min_volume)
        log(f"\nKalibrierung Closing ({venue}, n={len(ev)}):")
        log(_fmt(metrics.calibration_table(ev['y'], ev['close_price_a'])) if len(ev) else "  (keine Daten)")
        out[venue] = table

    if len(frames) == 2:
        pairs, h2h = head_to_head(frames['polymarket'], frames['kalshi'], min_volume)
        log(f"\n=== Gemeinsame Matches Polymarket/Kalshi: {len(pairs)} ===")
        if len(h2h):
            log(f"Ergebnis stimmt überein: {h2h.attrs['ergebnis_gleich']:.1%}, "
                f"mittlere |Δ Closing|: {h2h.attrs['mittlere_abweichung']:.3f}")
            log(_fmt(h2h))
        out['head_to_head'] = h2h

    matches = load_matches() if matches is None else matches
    if matches:
        log("\n=== Abgleich mit data/matches.csv (enthält nur ATP-Haupttour) ===")
        out['matching'] = matching_table(frames, matches)
        log(_fmt(out['matching']))
        rows = [r for bench in ('ps', 'avg') for v, df in frames.items() for col in ('close_price_a', 'close_1h_price_a')
                if (r := vs_pinnacle(df, v, matches, min_volume, col, bench))]
        out['pinnacle'] = pd.DataFrame(rows)
        log("\n=== Gegen Schlussquoten ohne Marge: ps = Pinnacle (bis Jan. 2026), avg = Ø der Buchmacher "
            "(gleiche Matches, ATP aus tennis-data) ===")
        log(_fmt(out['pinnacle']))
        log("ll_diff < 0: Venue besser als die Benchmark. anteil_edge/yield_flat: Kauf zum Ask (Kalshi) bzw. zur "
            "Mitte + 1 Cent (Polymarket, historisch kein Ask) nach Gebühren, wenn fair × Netto-Quote > 1. "
            "Pinnacle-Closing ist zeitlich später als der PM-Preis vor der geplanten Startzeit.")
    else:
        log("\n(data/matches.csv fehlt, kein Abgleich. Erst python -m src.update_data)")

    snaps = load_snapshots(root=snapshot_root)
    if len(snaps):
        log("\n=== Gesammelte Snapshots ===")
        summary = snaps.groupby('venue').agg(snapshots=('event_id', 'size'), maerkte=('event_id', 'nunique'),
                                             von=('ts_utc', 'min'), bis=('ts_utc', 'max')).reset_index()
        log(_fmt(summary))
        if matches:
            linked = link_matches(snaps.drop_duplicates(['venue', 'event_id']), matches)
            rate = linked.groupby(['venue', linked['tour'].fillna('?')])['match_found'].agg(['size', 'sum', 'mean'])
            log("Abgleich-Quote Snapshots (laufende Matches sind oft noch nicht in matches.csv):")
            log(rate.rename(columns={'size': 'n', 'sum': 'gefunden', 'mean': 'quote'}).to_string(float_format=lambda x: f"{x:.3f}"))
    return out


def main(argv=None):
    p = argparse.ArgumentParser(description="Kalibrierung und Log Loss der Prediction-Market-Closing-Preise")
    p.add_argument('--min-volume', type=float, default=0.0, help="nur Märkte mit mindestens so viel Volumen")
    args = p.parse_args(argv)
    report(args.min_volume)


if __name__ == '__main__':
    main()
