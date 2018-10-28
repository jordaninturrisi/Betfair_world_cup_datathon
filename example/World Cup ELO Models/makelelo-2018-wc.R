# +++++++
# Model: MakelELO
# Date: 23 / 5 / 2018
# +++++++

library(readr)
library(dplyr)
library(elo)
library(lubridate)

# Read in the world cup CSV data
training = read_csv("wc_datathon_dataset.csv")
training$date = dmy(training$date)

# Lowercase team names
training$team_1 = tolower(training$team_1)
training$team_2 = tolower(training$team_2)

# Read in submission file
wc_2018 = read_csv("john_smith_numbersman1.csv")

# Fix the ELO k factor - here you can try different values to see if improves the model performance
k_fac = 20

# Run ELO
elo_run = elo.run(
  score(team_1_goals, team_2_goals) ~ team_1 + team_2,
  data = training,
  k = k_fac
)

# Draw Rates
draw_rates = data.frame(win_prob = elo_run$elos[,3],win_loss_draw = elo_run$elos[,4]) %>%
  mutate(prob_bucket = abs(round((win_prob-(1-win_prob))*20)) / 20) %>%
  group_by(prob_bucket) %>%
  summarise(draw_prob = sum(ifelse(win_loss_draw==0.5, 1, 0)) / n())

# Run predictions on 2018 world cup: the predict function, in this case, just needs the home and away team names for the tournament
wc_2018_home_probabilities = predict(elo_run, newdata = wc_2018 %>% select(team_1, team_2))

# To our WC 2018 dataset let's add in our predicted win probabilities and fold in the expected draw rates from our table above
wc_2018 = wc_2018 %>%
  select(-prob_team_1_draw) %>%
  mutate(
    prob_team_1_win = wc_2018_home_probabilities,
    prob_team_1_lose = 1 - prob_team_1_win,
    prob_bucket = round(20 * abs((prob_team_1_win - prob_team_1_lose))) / 20
  ) %>%
  left_join(draw_rates) %>%
  mutate(
    prob_team_1_win = prob_team_1_win - 0.5 * draw_prob,
    prob_team_1_lose = prob_team_1_lose - 0.5 * draw_prob
  ) %>%
  select(date, match_id, team_1, team_2, prob_team_1_win, "prob_team_1_draw" = draw_prob, prob_team_1_lose, -prob_bucket)

# Write submission to file
write_csv(wc_2018, "betfair_datascientists_makelelo.csv")
