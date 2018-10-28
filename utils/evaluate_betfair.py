import pandas as pd
from sklearn.metrics import confusion_matrix, log_loss
from utils.plots import*
from sklearn.metrics import f1_score, precision_score, recall_score, precision_recall_fscore_support


def evaluate_betfair():
    classes = ['team 1 win', 'draw', 'team 2 win']

    betfair = pd.read_csv('Betfair_odds_history_no_friendly.csv')

    result = betfair[['team_1_win', 'draw', 'team_2_win']]
    odds = betfair[['team_1_betfair_prob', 'draw_betfair_prob', 'team_2_betfair_prob']]

    result = np.asarray(result)
    odds = np.asarray(odds)

    y_true = np.argmax(result, axis=1)
    y_pred = np.argmax(odds, axis=1)
    acc = np.mean(np.equal(y_true, y_pred))
    loss = log_loss(result, odds)

    # Calculate Precision, Recall, F1-Score
    score = precision_recall_fscore_support(y_true, y_pred, average='macro')
    print('     Loss: %0.5f' % loss)
    print(' Accuracy: %0.2f%%' % (acc*100))
    print('Precision: %0.2f%%' % (score[0]*100))
    print('   Recall: %0.2f%%' % (score[1]*100))
    print(' F1-Score: %0.2f%%' % (score[2]*100))

    cm = confusion_matrix(y_true, y_pred)
    plot_confusion_matrix(cm, classes, normalize=True)

    return
