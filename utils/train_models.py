import keras

from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.models import load_model

from sklearn.metrics import confusion_matrix, precision_recall_fscore_support, log_loss

from utils.model import *
from utils.plots import*


def train_models(model_type, x_train, y_train, x_val, y_val, x_test, y_test):
    VAL_FAIL = 10
    EPOCHS = 500
    BATCH_SIZE = 32

    num_inputs = x_train.shape[1]
    name = model_type.__name__

    # Create Model
    model = model_type(num_inputs=num_inputs)

    # Compile Model
    model.compile(loss='categorical_crossentropy', optimizer=keras.optimizers.SGD(lr=1e-2, momentum=0.9), metrics=['accuracy'])

    # Callbacks
    early_stopping = EarlyStopping(monitor='val_loss', patience=VAL_FAIL)
    checkpointer = ModelCheckpoint('models/' + name, monitor='val_loss', verbose=0, save_best_only=True, save_weights_only=False, mode='auto', period=1)

    # Train Model
    class_weight = {0: 1, 1: 1.995750708, 2: 1.764337591}
    history = model.fit(x_train, y_train,
                        batch_size=BATCH_SIZE, epochs=EPOCHS,
                        validation_data=[x_val, y_val],
                        verbose=0,
                        shuffle=False,
                        class_weight=class_weight,
                        callbacks=[early_stopping, checkpointer])

    plot_loss(history)

    evaluate_model(model_type, x_train, y_train, x_val, y_val, x_test, y_test)

    return model


def evaluate_model(model_type, x_train, y_train, x_val, y_val, x_test, y_test):

    name = model_type.__name__
    model = load_model('models/' + name)
    classes = ['team 1 win', 'draw', 'team 2 win']

    # Evaluate Training, Validation, and Test Set
    train_score = model.evaluate(x_train, y_train, verbose=0)
    val_score = model.evaluate(x_val, y_val, verbose=0)
    test_score = model.evaluate(x_test, y_test, verbose=0)

    # Calculate probabilities and convert to dense vector
    prob_train = model.predict(x_train)
    y_true_train = np.argmax(y_train, axis=1)
    y_pred_train = np.argmax(prob_train, axis=1)

    prob_val = model.predict(x_val)
    y_true_val = np.argmax(y_val, axis=1)
    y_pred_val = np.argmax(prob_val, axis=1)

    prob_test = model.predict(x_test)
    y_true_test = np.argmax(y_test, axis=1)
    y_pred_test = np.argmax(prob_test, axis=1)

    # Calculate Precision, Recall and F1 Score
    score_train = precision_recall_fscore_support(y_true_train, y_pred_train, average='macro')
    score_val = precision_recall_fscore_support(y_true_val, y_pred_val, average='macro')
    score_test = precision_recall_fscore_support(y_true_test, y_pred_test, average='macro')

    # Print Performance to Display
    print(name)
    print('Train Loss: %0.5f     Train Accuracy: %0.2f%%     Precision: %0.2f%%     Recall: %0.2f%%     F1-Score: %0.2f%%' % (train_score[0], (train_score[1]*100), (score_train[0]*100), (score_train[1]*100), (score_train[2]*100)))
    print('  Val Loss: %0.5f       Val Accuracy: %0.2f%%     Precision: %0.2f%%     Recall: %0.2f%%     F1-Score: %0.2f%%' % (val_score[0], (val_score[1]*100), (score_val[0]*100), (score_val[1]*100), (score_val[2]*100)))
    print(' Test Loss: %0.5f      Test Accuracy: %0.2f%%     Precision: %0.2f%%     Recall: %0.2f%%     F1-Score: %0.2f%%' % (test_score[0], (test_score[1]*100), (score_test[0]*100), (score_test[1]*100), (score_test[2]*100)))

    # Confusion Matrix
    cm_train = confusion_matrix(y_true_train, y_pred_train)
    cm_val = confusion_matrix(y_true_val, y_pred_val)
    cm_test = confusion_matrix(y_true_test, y_pred_test)

    plot_confusion_matrix(cm_train, classes, normalize=True)
    plot_confusion_matrix(cm_val, classes, normalize=True)
    plot_confusion_matrix(cm_test, classes, normalize=True)

    return



def ensemble_v1(X, Y):
    linear_model = load_model('models/linear_model')
    tanh_model = load_model('models/tanh_model')
    sigmoid_model = load_model('models/sigmoid_model')
    softmax_model = load_model('models/softmax_model')
    relu_model = load_model('models/relu_model')
    CNN_model = load_model('models/CNN_model')

    linear_prob = linear_model.predict(X)
    tanh_prob = tanh_model.predict(X)
    sigmoid_prob = sigmoid_model.predict(X)
    softmax_prob = softmax_model.predict(X)
    relu_prob = relu_model.predict(X)
    CNN_prob = CNN_model.predict(X)

    overall_prob = np.mean((linear_prob, tanh_prob, sigmoid_prob, softmax_prob, relu_prob, CNN_prob), axis=0)

    ## FUNCTION
    y_true = np.argmax(Y, axis=1)
    y_pred = np.argmax(overall_prob, axis=1)
    acc = np.mean(np.equal(y_true, y_pred))
    loss = log_loss(Y, overall_prob)

    score = precision_recall_fscore_support(y_true, y_pred, average='macro')

    print('     Loss: %0.5f' % loss)
    print(' Accuracy: %0.2f%%' % (acc*100))
    print('Precision: %0.2f%%' % (score[0]*100))
    print('   Recall: %0.2f%%' % (score[1]*100))
    print(' F1-Score: %0.2f%%' % (score[2]*100))

    cm = confusion_matrix(y_true, y_pred)
    classes = ['team 1 win', 'draw', 'team 2 win']
    plot_confusion_matrix(cm, classes, normalize=True)

    return


def ensemble_v2(X, Y):
    linear_model = load_model('models/linear_model')
    softmax_model = load_model('models/softmax_model')
    relu_model = load_model('models/relu_model')
    CNN_model = load_model('models/CNN_model')

    linear_prob = linear_model.predict(X)
    softmax_prob = softmax_model.predict(X)
    relu_prob = relu_model.predict(X)
    CNN_prob = CNN_model.predict(X)

    overall_prob = np.mean((linear_prob, softmax_prob, relu_prob, CNN_prob), axis=0)

    ## FUNCTION
    y_true = np.argmax(Y, axis=1)
    y_pred = np.argmax(overall_prob, axis=1)
    acc = np.mean(np.equal(y_true, y_pred))
    loss = log_loss(Y, overall_prob)

    score = precision_recall_fscore_support(y_true, y_pred, average='macro')

    print('     Loss: %0.5f' % loss)
    print(' Accuracy: %0.2f%%' % (acc*100))
    print('Precision: %0.2f%%' % (score[0]*100))
    print('   Recall: %0.2f%%' % (score[1]*100))
    print(' F1-Score: %0.2f%%' % (score[2]*100))

    cm = confusion_matrix(y_true, y_pred)
    classes = ['team 1 win', 'draw', 'team 2 win']
    plot_confusion_matrix(cm, classes, normalize=True)

    return


def ensemblev1(X, Y=None):
    linear_model = load_model('models/linear_model')
    tanh_model = load_model('models/tanh_model')
    sigmoid_model = load_model('models/sigmoid_model')
    softmax_model = load_model('models/softmax_model')
    relu_model = load_model('models/relu_model')
    CNN_model = load_model('models/CNN_model')

    linear_prob = linear_model.predict(X)
    tanh_prob = tanh_model.predict(X)
    sigmoid_prob = sigmoid_model.predict(X)
    softmax_prob = softmax_model.predict(X)
    relu_prob = relu_model.predict(X)
    CNN_prob = CNN_model.predict(X)

    overall_prob = np.mean((linear_prob, tanh_prob, sigmoid_prob, softmax_prob, relu_prob, CNN_prob), axis=0)

    return overall_prob


def ensemblev2(X, Y=None):
    linear_model = load_model('models/linear_model')
    softmax_model = load_model('models/softmax_model')
    relu_model = load_model('models/relu_model')
    CNN_model = load_model('models/CNN_model')

    linear_prob = linear_model.predict(X)
    softmax_prob = softmax_model.predict(X)
    relu_prob = relu_model.predict(X)
    CNN_prob = CNN_model.predict(X)

    overall_prob = np.mean((linear_prob, softmax_prob, relu_prob, CNN_prob), axis=0)

    return overall_prob
