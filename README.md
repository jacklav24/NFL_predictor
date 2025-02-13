# NFL_predictor
A project to take specific stats and predict specific results of NFL games.

Uses a kaggle dataset: https://www.kaggle.com/datasets/cviaxmiwnptr/nfl-team-stats-20022019-espn?resource=download

As a base for training model. Using this to learn more machine learning.

results: answers.csv stores various model results based on different parameters. Currently doesn't state which features were used on each run but will in the future.
chiefgle.csv shows predicted results of the chiefs/eagles superbowl based on the model each time. Correctly predicts a high probability of a chiefs loss. (unfortunately)

Testing_football.ipynb houses the ml models for matchup prediction.

basics.py holds some basic (haha) functions for getting df.

graphing.ipynb holds some light UI for comparision of stats and game results. you can compare two stats on the two axes and see all game results. It also has some graphs at the bottom which show wins / time of possesion.
