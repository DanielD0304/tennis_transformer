class tennis_match:
    def __init__(self, player1_name, player1_odd, player2_name, player2_odd, source):
        self.player1 = {'name': player1_name, 'odd': player1_odd, 'source': source}
        self.player2 = {'name': player2_name, 'odd': player2_odd, 'source': source}
        # Wird vom Modell gefüllt (Prediction aus src/predict.py)
        self.prediction = None

    def update_odds(self, player1_odd, player2_odd, source):
        if player1_odd > self.player1['odd']:
            self.player1['odd'] = player1_odd
            self.player1['source'] = source
        if player2_odd > self.player2['odd']:
            self.player2['odd'] = player2_odd
            self.player2['source'] = source

    def bet_share(self):
        """Summe der impliziten Wahrscheinlichkeiten der besten Quoten (< 1 = Surebet)."""
        return 1 / self.player1['odd'] + 1 / self.player2['odd']
