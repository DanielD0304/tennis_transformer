class Football_game:
    def __init__(self, team1_name, team1_odd, draw_odd, team2_name, team2_odd):
        self.team1 = {'name': team1_name, 'odd': team1_odd}
        self.draw = {'name':f"{team1_name} - {team2_name}", 'odd': draw_odd}
        self.team2 = {'name': team2_name, 'odd': team2_odd}