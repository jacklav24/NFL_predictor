import pandas as pd


def get_df():
    df = pd.read_csv("/Users/jcklvrgn/Documents/GitHub/NFL_predictor/Football/nfl_team_stats_2002-2023.csv")

    df['winner'] = df.apply(
        lambda row: row['home'] if row['score_home'] > row['score_away'] else row['away'],
        axis=1
        )



    return df
    

def get_team_stats(year_start, year_end, df):

  
    teams = pd.concat([df['home'], df['away']]).unique()
    abv = [team[:4] for team in teams]
    df_using = df[(df['season'] >= year_start) & (df['season'] <= year_end)]
    all_team_stats = pd.DataFrame({'team': teams,
                            'abbrev': abv})
    all_team_stats['total_wins'] = all_team_stats['team'].apply(lambda team: (df_using['winner'] == team).sum())
    all_team_stats['win_with_ToP'] = all_team_stats['team'].apply((lambda team: (
        (
            (df_using['winner'] == team) &  # Check if the team won
            (
                ((df_using['home'] == team) & (df_using['possession_home'] >= df_using['possession_away'])) |  # Home team has top possession
                ((df_using['away'] == team) & (df_using['possession_away'] >= df_using['possession_home']))  # Away team has top possession
            )
        ).sum()
    ))) #/ team_stats['total_wins']
    all_team_stats['percent_W_ToP'] = all_team_stats['win_with_ToP'] / all_team_stats['total_wins']
    all_team_stats.sort_values(by='total_wins', inplace=True, ascending = False)
    return all_team_stats, teams, abv
def get_team_stats_all():
    return get_team_stats(2002,2023)

get_team_stats(2002,2023, get_df())

def get_home_stats(df):
    return df[[]]