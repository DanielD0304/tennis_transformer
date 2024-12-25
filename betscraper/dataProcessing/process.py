class process_data:
    def __init__(self, game_data):
        self.game_data = game_data

    def process_data(self):
        result_dict = {}
        for game in self.game_data:
            draw_name = game.draw['name']
            team1_odd = float(game.team1['odd'].replace(',', '.'))
            draw_odd = float(game.draw['odd'].replace(',', '.'))
            team2_odd = float(game.team2['odd'].replace(',', '.'))
            bet_share = 1 / team1_odd + 1 / draw_odd + 1 / team2_odd
            result_dict[draw_name] = bet_share

        return result_dict

    # Weitere Verarbeitungsschritte