import numpy as np
import matplotlib.pyplot as plt
import itertools


# Plot Confusion Matrix
def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues, figsize=None):
	"""
	This function prints and plots the confusion matrix.
	Normalization can be applied by setting `normalize=True`.
	(This function is copied from the scikit docs.)
	"""
	plt.figure(figsize=(figsize))
	plt.title(title, fontsize=22)
	tick_marks = np.arange(len(classes))
	plt.xticks(tick_marks, classes, fontsize=14, rotation=45)
	plt.yticks(tick_marks, classes, fontsize=14)

	# Print unformatted confusion matrix
	# print(cm)

	if normalize:
	    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
	    print('\nNormalized confusion matrix')
	else:
	    print('\nConfusion matrix, without normalization')

	fmt = '.2f' if normalize else 'd'
	thresh = cm.max() / 2

	for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
	    plt.text(j, i, format(cm[i, j], fmt), fontsize=14, horizontalalignment="center", color="white" if cm[i, j] > thresh else "black")

	plt.imshow(cm, interpolation='nearest', cmap=cmap)
	plt.colorbar()
	plt.tight_layout()
	plt.ylabel('True label', fontsize=18)
	plt.xlabel('Predicted label', fontsize=18)

	plt.show()



# Plot the train/val loss/accuracy as a function of epoch
# Include test loss & accuracy
def plot_loss(history):

	# Set size of figure
	plt.figure(figsize=(12,4))

	# Subplot 1: Loss
	# Logarithmic y axis, indicating test loss and lowest validation loss
	plt.subplot(1, 2, 1)
	plt.semilogy(history.history['loss'], color='r')
	plt.semilogy(history.history['val_loss'], color='g')
	plt.axvline(np.argmin(history.history['val_loss']), color='k', linestyle=':')
	plt.title('Model Loss', fontsize=18)
	plt.ylabel('Loss', fontsize=14)
	plt.xlabel('Epoch', fontsize=14)
	plt.legend(['Train', 'Val', 'Test'], loc='upper right')

	# Subplot 2: Accuracy
	# Indicating test accuracy
	plt.subplot(1, 2, 2)
	plt.plot([i*100 for i in history.history['acc']], color='r')
	plt.plot([i*100 for i in history.history['val_acc']], color='g')
	plt.axvline(np.argmin(history.history['val_loss']), color='k', linestyle=':')
	plt.title('Model Accuracy', fontsize=18)
	plt.ylabel('Accuracy', fontsize=14)
	plt.xlabel('Epoch', fontsize=14)
	plt.legend(['Train', 'Val', 'Test'], loc='lower right')

	# Accommodate for tight layout & show image
	plt.tight_layout()
	plt.show()

	return



# Plot Class Frequency
# Show how many of each class are in each set
# Input sparse matrices & convert to dense target vectors
def plot_class_frequency(y_train, y_val, y_test):
	# Data
	count_train = np.unique(np.argmax(y_train, axis=1), return_counts=True)
	count_val = np.unique(np.argmax(y_val, axis=1), return_counts=True)
	count_test = np.unique(np.argmax(y_test, axis=1), return_counts=True)

	total_count = count_train[1] + count_val[1] + count_test[1]

	rel_train = count_train[1] / total_count
	rel_val = count_val[1] / total_count
	rel_test = count_test[1] / total_count

	width = 0.35	# Width of bars

	plt.figure(figsize=(16,12))

	# Subplot 1: Training Set
	plt.subplot(2, 2, 1)
	plt.bar(count_train[0], count_train[1], width, color='b', label='Train')
	plt.xticks(count_train[0] + width, count_train[0])
	plt.ylabel('Occurrences')
	plt.legend(loc='upper right', fontsize=12)

	# Subplot 2: Validation & Test Set
	plt.subplot(2, 2, 2)
	plt.bar(count_val[0] + width, count_val[1], width, color='r', label='Val')
	plt.bar(count_test[0] + 2*width, count_test[1], width, color='y', label='Test')
	plt.xticks(count_train[0] + width*1.5, count_train[0])
	plt.ylabel('Occurrences')
	plt.legend(loc='upper right', fontsize=12)

	# Subplot 3: Training Set
	plt.subplot(2, 2, 3)
	plt.bar(count_train[0], (rel_train*100), width, color='b', label='Train')
	plt.xticks(count_train[0] + width, count_train[0])
	plt.ylabel('Percentage (%)')
	plt.legend(loc='lower right', fontsize=12)

	# Subplot 4: Validation & Test Set
	plt.subplot(2, 2, 4)
	plt.bar(count_val[0] + width, (rel_val*100), width, color='r', label='Val')
	plt.bar(count_test[0] + 2*width, (rel_test*100), width, color='y', label='Test')
	plt.xticks(count_train[0] + width*1.5, count_train[0])
	plt.ylabel('Percentage (%)')
	plt.legend(loc='lower right', fontsize=12)

	plt.suptitle('Frequency of Outcomes', fontsize=20)

	plt.show()

	return
