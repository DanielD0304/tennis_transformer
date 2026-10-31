"""
Venue-Profile: woher der Preis kommt, welche Steuer/Gebühr anfällt, wie RET abgerechnet wird.

Alles wird auf eine Netto-Dezimalquote pro Seite zurückgeführt: Gewinn bei Treffer =
Einsatz * (netto - 1), bei Niederlage = -Einsatz, bei void = 0.

Betfair Exchange hat echte historische Kurse (tennis-data BFEW/BFEL, ab August 2025), 'betfair' rechnet
damit. Für Kalshi und Polymarket (und Betfair vor August 2025: 'betfair_proxy') dient die faire
Pinnacle-Wahrscheinlichkeit (ohne Marge) als Preis, plus optionalem Spread. Diese Venues sind im
Output als „Proxy“ markiert.

Polymarket-Gebühr (docs.polymarket.com/trading/fees, Gamma: feeType sports_fees_v3, Stand Okt. 2026): je Anteil
Rate * p * (1 - p) in USDC, nur für den Taker. Die Doku sagt nicht, ob sie beim Kauf in Anteilen oder USDC eingezogen
wird. Gerechnet wird mit Anteilen: Rate * (1 - p) Anteile je Anteil, also die Rate auf den möglichen Gewinn. Das ist die
vorsichtigere Lesart (Unterschied unter 0,2 % der Quote) und gilt exakt auch für den Durchschnittspreis mehrerer Level.

Die RET-Regeln sind Annahmen und unterscheiden sich je Buchmacher/Markt (vor echtem Einsatz in den
AGB prüfen). W/O ist immer void.
"""
from dataclasses import dataclass, replace

import numpy as np

DE_BET_TAX = 0.053
POLYMARKET_FEE = 0.05  # Rate für Sport, fetch_current meldet, wenn Gamma etwas anderes nennt


def ceil_to_cent(x):
    """Auf den nächsten Cent aufrunden (mit Toleranz gegen Float-Fehler wie 0.07*0.5*0.5)."""
    return np.ceil(np.round(np.asarray(x, float) * 100, 9)) / 100


def kalshi_fee(price, contracts=1):
    """Kalshi-Taker-Fee in Dollar: ceil_to_cent(0,07 * Kontrakte * p * (1 - p))."""
    price = np.asarray(price, float)
    return ceil_to_cent(0.07 * contracts * price * (1 - price))


@dataclass(frozen=True)
class Venue:
    name: str
    price: str  # Quotenkürzel aus predictions ('ps', 'b365', 'avg', 'max', 'bfe') oder 'fair_ps' (Proxy)
    stake_tax: float = 0.0  # Anteil des Einsatzes, der als Steuer einbehalten wird (DE: 5,3 %)
    commission: float = 0.0  # Anteil des Nettogewinns (Exchange)
    fee_model: str = 'none'  # 'none' | 'kalshi' | 'polymarket'
    taker_fee: float = 0.0  # Polymarket: Rate der Gebühr (siehe oben), Anteil des möglichen Gewinns
    spread: float = 0.0  # Proxy-Preise: Aufschlag auf die faire Wahrscheinlichkeit (halber Spread je Seite)
    ret_rule: str = 'settled'  # 'void' (Einsatz zurück) oder 'settled' (Sieger wird ausgezahlt)
    max_stake: float = None  # Limit pro Wette in Währung, None = unbegrenzt
    proxy: bool = False

    def __post_init__(self):
        if self.ret_rule not in ('void', 'settled'):
            raise ValueError(f"ret_rule muss 'void' oder 'settled' sein, nicht {self.ret_rule!r}")
        if self.fee_model not in ('none', 'kalshi', 'polymarket'):
            raise ValueError(f"Unbekanntes fee_model {self.fee_model!r}")

    @property
    def label(self):
        return f"{self.name} (Proxy)" if self.proxy else self.name

    def gross_odds(self, predictions):
        """Bruttoquoten (A, B) vor Steuer/Gebühr. NaN = kein Preis."""
        if self.price == 'fair_ps':
            p = predictions['fair_ps'].to_numpy(float)
            half = self.spread / 2
            qa, qb = np.clip(p + half, 1e-6, 1), np.clip(1 - p + half, 1e-6, 1)
            return 1 / qa, 1 / qb
        if f'{self.price}_a' not in predictions.columns:  # z.B. ältere predictions.parquet ohne BFE
            nan = np.full(len(predictions), np.nan)
            return nan, nan.copy()
        return predictions[f'{self.price}_a'].to_numpy(float), predictions[f'{self.price}_b'].to_numpy(float)

    def net_odds_from_gross(self, odds):
        odds = np.asarray(odds, float)
        if self.fee_model == 'kalshi':
            # Preis pro Kontrakt (zahlt 1 $) plus Fee; pro Kontrakt gerundet (konservativ)
            price = 1 / odds
            net = 1 / (price + kalshi_fee(price))
        elif self.fee_model == 'polymarket':
            net = 1 + (odds - 1) * (1 - self.taker_fee)
        else:
            net = odds * (1 - self.stake_tax)
        return 1 + (net - 1) * (1 - self.commission)

    def net_odds(self, predictions):
        a, b = self.gross_odds(predictions)
        return self.net_odds_from_gross(a), self.net_odds_from_gross(b)

    def is_void(self, status):
        status = np.asarray(status).astype(str)
        void = status == 'W/O'
        if self.ret_rule == 'void':
            void |= np.isin(status, ['RET', 'DEF'])
        return void

    def with_(self, **changes):
        return replace(self, **changes)


VENUES = {
    'pinnacle': Venue('pinnacle', price='ps'),
    'bet365_de': Venue('bet365_de', price='b365', stake_tax=DE_BET_TAX),
    'avg_de': Venue('avg_de', price='avg', stake_tax=DE_BET_TAX),
    'betfair': Venue('betfair', price='bfe', commission=0.05),  # echte Exchange-Kurse, 5 % auf Gewinne
    'betfair_proxy': Venue('betfair_proxy', price='fair_ps', commission=0.05, proxy=True),
    'kalshi': Venue('kalshi', price='fair_ps', fee_model='kalshi', proxy=True),
    'polymarket': Venue('polymarket', price='fair_ps', fee_model='polymarket', taker_fee=POLYMARKET_FEE, spread=0.02, proxy=True),
}


def get_venue(name, **overrides):
    if name not in VENUES:
        raise ValueError(f"Unbekannte Venue {name!r} (bekannt: {', '.join(VENUES)})")
    return VENUES[name].with_(**overrides) if overrides else VENUES[name]
