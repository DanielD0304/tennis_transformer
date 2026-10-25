"""
Trainiert den Transformer und wertet ihn danach mit src.evaluation aus:
Baseline (Rang), Metriken gegen den Markt (Pinnacle ohne Marge), Wett-Strategien pro Venue
und Kalibrierung.

    python main.py                 # trainieren + auswerten
    python main.py --eval-only     # nur auswerten (vorhandenes models/best_model.pt)
"""
import argparse

from src.config import default_config
from src.evaluation import backtest, metrics
from src.evaluation.venues import VENUES

# Bins wie im alten Kalibrierungs-Check
CALIBRATION_BINS = [0.0, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]


def evaluate(config=default_config, with_transformer=True, refresh=True):
    """
    Walk-forward ab dem ersten Jahr nach dem Training (val_year), damit der Blend auf früheren
    Transformer-Vorhersagen lernen kann. Ausgewertet wird das Testjahr.
    """
    from src.data_loader import load_matches

    models = ['market', 'rank', 'elo'] + (['transformer'] if with_transformer else []) + ['blend']
    df = load_matches(first_year=config.data_years_start, last_year=config.test_year, refresh=refresh)
    predictions, names = backtest.run(df, models, config.train_years_end + 1, config.test_year, config=config)
    print(f"Vorhersagen gespeichert: {backtest.save(predictions)}")
    test = predictions[predictions['year'] == config.test_year]

    print("\n" + "=" * 60)
    print(f"BASELINE-ACCURACY {config.test_year} (alle gespielten Matches mit Vorhersage)")
    print("=" * 60)
    played = test[test['status'] != 'W/O']
    for m in names:
        sub = played[played[f'p_{m}'].notna()]
        if len(sub):
            print(f"{m:<12} {metrics.accuracy(sub['label'], sub[f'p_{m}']):.2%}  (n={len(sub)})")

    backtest.report(test, names, venues=[VENUES['bet365_de'], VENUES['pinnacle']])

    # Transformer, sonst Blend; ohne Quoten (kein Blend möglich) Elo
    candidates = (['transformer'] if with_transformer else []) + ['blend', 'elo']
    model = next(m for m in candidates if played[f'p_{m}'].notna().any() or m == 'elo')
    sub = played[played[f'p_{model}'].notna()]
    print("\n" + "=" * 60)
    print(f"DIAGNOSE: KALIBRIERUNG ({model}, {config.test_year})")
    print("=" * 60)
    if len(sub):
        print(metrics.calibration_table(sub['label'], sub[f'p_{model}'], CALIBRATION_BINS).to_string(
            index=False, float_format=lambda v: f"{v:.3f}"))
        print(f"ECE: {metrics.ece(sub['label'], sub[f'p_{model}']):.4f}")
        print("Hinweis: Negative 'diff' bedeutet, das Modell ist zu optimistisch (overconfident).")
    return predictions


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--eval-only', action='store_true', help="Nicht trainieren, nur auswerten")
    ap.add_argument('--no-transformer', action='store_true', help="Nur market, rank, elo, blend")
    args = ap.parse_args()

    if not args.eval_only and not args.no_transformer:
        from src.train import train
        print("Training Transformer Model...\n")
        train()
    evaluate(with_transformer=not args.no_transformer)


if __name__ == "__main__":
    main()
