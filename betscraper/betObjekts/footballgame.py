class football_game:
    def __init__(self, team1_name, team1_odd, draw_odd, team2_name, team2_odd, source):
        self.team1 = {'name': team1_name, 'odd': team1_odd, 'source': source}
        self.draw = {'name':f"{team1_name} - {team2_name}", 'odd': draw_odd, 'source': source}
        self.team2 = {'name': team2_name, 'odd': team2_odd, 'source': source}

    def update_odds(self, team1_odd, draw_odd, team2_odd, source):
        if team1_odd > self.team1['odd']:
            self.team1['odd'] = team1_odd
            self.team1['source'] = source
        if draw_odd > self.draw['odd']:
            self.draw['odd'] = draw_odd
            self.draw['source'] = source
        if team2_odd > self.team2['odd']:
            self.team2['odd'] = team2_odd
            self.team2['source'] = source