import pandas as pd

def load_elo():

    elo1999, elo2000, elo2001, elo2002, elo2003, elo2004, elo2005, elo2006, \
    elo2007, elo2008, elo2009, elo2010, elo2011, elo2012, elo2013, elo2014, \
    elo2015, elo2016, elo2017, elo2018 = [], [], [], [], [], [], [], [], [], \
    [], [], [], [], [], [], [], [], [], [], []

    elo1997 = pd.read_csv('data/historic_elo/' + '1997.tsv', delimiter='\t', header=None, keep_default_na=False)
    elo1998 = pd.read_csv('data/historic_elo/' + '1998.tsv', delimiter='\t', header=None, keep_default_na=False)
    elo1999 = pd.read_csv('data/historic_elo/' + '1999.tsv', delimiter='\t', header=None, keep_default_na=False)
    elo2000 = pd.read_csv('data/historic_elo/' + '2000.tsv', delimiter='\t', header=None, keep_default_na=False)
    elo2001 = pd.read_csv('data/historic_elo/' + '2001.tsv', delimiter='\t', header=None, keep_default_na=False)
    elo2002 = pd.read_csv('data/historic_elo/' + '2002.tsv', delimiter='\t', header=None, keep_default_na=False)
    elo2003 = pd.read_csv('data/historic_elo/' + '2003.tsv', delimiter='\t', header=None, keep_default_na=False)
    elo2004 = pd.read_csv('data/historic_elo/' + '2004.tsv', delimiter='\t', header=None, keep_default_na=False)
    elo2005 = pd.read_csv('data/historic_elo/' + '2005.tsv', delimiter='\t', header=None, keep_default_na=False)
    elo2006 = pd.read_csv('data/historic_elo/' + '2006.tsv', delimiter='\t', header=None, keep_default_na=False)
    elo2007 = pd.read_csv('data/historic_elo/' + '2007.tsv', delimiter='\t', header=None, keep_default_na=False)
    elo2008 = pd.read_csv('data/historic_elo/' + '2008.tsv', delimiter='\t', header=None, keep_default_na=False)
    elo2009 = pd.read_csv('data/historic_elo/' + '2009.tsv', delimiter='\t', header=None, keep_default_na=False)
    elo2010 = pd.read_csv('data/historic_elo/' + '2010.tsv', delimiter='\t', header=None, keep_default_na=False)
    elo2011 = pd.read_csv('data/historic_elo/' + '2011.tsv', delimiter='\t', header=None, keep_default_na=False)
    elo2012 = pd.read_csv('data/historic_elo/' + '2012.tsv', delimiter='\t', header=None, keep_default_na=False)
    elo2013 = pd.read_csv('data/historic_elo/' + '2013.tsv', delimiter='\t', header=None, keep_default_na=False)
    elo2014 = pd.read_csv('data/historic_elo/' + '2014.tsv', delimiter='\t', header=None, keep_default_na=False)
    elo2015 = pd.read_csv('data/historic_elo/' + '2015.tsv', delimiter='\t', header=None, keep_default_na=False)
    elo2016 = pd.read_csv('data/historic_elo/' + '2016.tsv', delimiter='\t', header=None, keep_default_na=False)
    elo2017 = pd.read_csv('data/historic_elo/' + '2017.tsv', delimiter='\t', header=None, keep_default_na=False)
    elo2018 = pd.read_csv('data/historic_elo/' + '2018.tsv', delimiter='\t', header=None, keep_default_na=False)

    return elo1997, elo1998, elo1999, elo2000, elo2001, elo2002, elo2003, \
    elo2004, elo2005, elo2006, elo2007, elo2008, elo2009, elo2010, elo2011, \
    elo2012, elo2013, elo2014, elo2015, elo2016, elo2017, elo2018


def load_stats(eloStats, homeTeam, awayTeam, game, train):
    home, away = 0, 0

    homeStats, awayStats = [], []
    homeRank, awayRank = 0, 0
    homeOther, awayOther = 0, 0
    homeElo, awayElo = 0, 0

    # Search all team's ratings
    for t in range(eloStats.shape[0]):

        # If home team's code match
        if eloStats[1][t] == homeTeam:
            homeStats = eloStats.loc[t]
            homeRank = homeStats[0]
            homeElo = homeStats[2]
            homeOther = homeStats[3:]

            train[game, 0] = homeRank
            train[game, 1] = homeElo
            train[game, 2:31] = homeOther

            home = 1

        elif eloStats[1][t] == awayTeam:
            awayStats = eloStats.loc[t]
            awayRank = awayStats[0]
            awayElo = awayStats[2]
            awayOther = awayStats[3:]

            train[game, 31] = awayRank
            train[game, 32] = awayElo
            train[game, 33:62] = awayOther

            away = 1

        if home and away:
            train[game, 62] = homeRank - awayRank
            train[game, 63] = homeElo - awayElo
            train[game, 64:] = homeOther - awayOther

            return train


def create_elo_vectors(data, train):
    elo1997, elo1998, elo1999, elo2000, elo2001, elo2002, elo2003, elo2004, \
    elo2005, elo2006, elo2007, elo2008, elo2009, elo2010, elo2011, elo2012, \
    elo2013, elo2014, elo2015, elo2016, elo2017, elo2018 = load_elo()

    for game in range(data.shape[0]):
        year = data['year'][game]
        eloYear = year
        homeTeam = data['team_1_code'][game]
        awayTeam = data['team_2_code'][game]

        # For Year 2000, use ELO ratings from 1999
        if eloYear == 1997:
            train = load_stats(elo1999, homeTeam, awayTeam, game, train)

        elif eloYear == 1998:
            train = load_stats(elo2000, homeTeam, awayTeam, game, train)

        elif eloYear == 1999:
            train = load_stats(elo2000, homeTeam, awayTeam, game, train)

        elif eloYear == 2000:
            train = load_stats(elo2000, homeTeam, awayTeam, game, train)

        elif eloYear == 2001:
            train = load_stats(elo2001, homeTeam, awayTeam, game, train)

        elif eloYear == 2002:
            train = load_stats(elo2002, homeTeam, awayTeam, game, train)

        elif eloYear == 2003:
            train = load_stats(elo2003, homeTeam, awayTeam, game, train)

        elif eloYear == 2004:
            train = load_stats(elo2004, homeTeam, awayTeam, game, train)

        elif eloYear == 2005:
            train = load_stats(elo2005, homeTeam, awayTeam, game, train)

        elif eloYear == 2006:
            train = load_stats(elo2006, homeTeam, awayTeam, game, train)

        elif eloYear == 2007:
            train = load_stats(elo2007, homeTeam, awayTeam, game, train)

        elif eloYear == 2008:
            train = load_stats(elo2008, homeTeam, awayTeam, game, train)

        elif eloYear == 2009:
            train = load_stats(elo2009, homeTeam, awayTeam, game, train)

        elif eloYear == 2010:
            train = load_stats(elo2010, homeTeam, awayTeam, game, train)

        elif eloYear == 2011:
            train = load_stats(elo2011, homeTeam, awayTeam, game, train)

        elif eloYear == 2012:
            train = load_stats(elo2012, homeTeam, awayTeam, game, train)

        elif eloYear == 2013:
            train = load_stats(elo2013, homeTeam, awayTeam, game, train)

        elif eloYear == 2014:
            train = load_stats(elo2014, homeTeam, awayTeam, game, train)

        elif eloYear == 2015:
            train = load_stats(elo2015, homeTeam, awayTeam, game, train)

        elif eloYear == 2016:
            train = load_stats(elo2016, homeTeam, awayTeam, game, train)

        elif eloYear == 2017:
            train = load_stats(elo2017, homeTeam, awayTeam, game, train)

        elif eloYear == 2018:
            train = load_stats(elo2018, homeTeam, awayTeam, game, train)



    return train
