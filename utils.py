import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, multilabel_confusion_matrix, precision_recall_fscore_support
from sklearn.metrics.pairwise import cosine_similarity
import pickle
import numpy as np
import pandas as pd

class plot_results:
	# Instantiation of class
	def __init__(self, train_losses_plot, val_losses_plot, val_accuracies_plot,
				 figname=["Training.png", "Validation.png"], smooth=False):
		self.train_losses_plot = train_losses_plot
		self.val_losses_plot = val_losses_plot
		self.val_accuracies_plot = val_accuracies_plot
		self.smooth = smooth
		self.figname = figname

		flatten = lambda l: [item for sublist in l for item in sublist]
		if smooth:
			# If smooth is True, taking the average of every epoch
			self.train_losses_plot = np.array([np.mean(loss_iter) for loss_iter in list(self.train_losses_plot)])
			self.val_losses_plot = np.array([np.mean(loss_iter) for loss_iter in list(self.val_losses_plot)])
			self.val_accuracies_plot = np.array([np.mean(loss_iter) for loss_iter in list(self.val_accuracies_plot)])
		else:
			# If smooth is False, flattening out the list to display avg values for every 10 iterations
			self.train_losses_plot = flatten(self.train_losses_plot)
			self.val_losses_plot = flatten(self.val_losses_plot)
			self.val_accuracies_plot = flatten(self.val_accuracies_plot)

	# After instantiating the class, this is the function that needs to be called:
	def run(self, figure_sep=True):
		#        %matplotlib inline
		plt.style.use('classic')
		plt.figure(figsize=(10, 8))
		print("Starting to plot figures.... \n\n")
		if figure_sep:
			fig, ax = plt.subplot(2, 1, 1)
			ax.plot(self.train_losses_plot)
			if self.smooth:
				title_str = 'Average training loss for every Epoch'
			else:
				title_str = "Training losses for every 10 iterations"
			ax.set(ylabel='Training set values', title=title_str)
			ax.grid()
			if self.figname is not None:
				fig.savefig(self.figname[0])
			plt.show()

			fig, ax = plt.subplot(2, 1, 2)
			ax.plot(self.val_losses_plot, color='blue')
			ax.plot(self.val_accuracies_plot, color='green')
			ax.legend(['Val losses', 'Val Accuracies'], loc='best')
			if self.smooth:
				title_str = 'Average Validation losses and accuracy for every Epoch'
			else:
				title_str = "Validation losses and accuracy for every 10 iterations"
			ax.set(ylabel='Validation set values', title=title_str)
			ax.grid()
			if self.figname is not None:
				fig.savefig(self.figname[1])
			plt.show()

		else:
			fig, ax = plt.subplot(2, 1, 2)
			ax.plot(self.train_losses_plot, color='red')
			ax.plot(self.val_losses_plot, color='blue')
			ax.plot(self.val_accuracies_plot, color='green')
			ax.legend(['Training_loss', 'Val loss', 'Val Accuracies'], loc='best')
			if self.smooth:
				title_str = 'Average Training loss Validation losses and accuracy for every Epoch'
			else:
				title_str = "Training loss Validation loss and accuracy for every 10 iterations"
			ax.set(title=title_str)
			ax.grid()
			if self.figname is not None:
				fig.savefig(self.figname[0])
			plt.show()

		print("All graphs plotted! \n\n")


def print_classification_report(pred_list, title, target_names=['Direct', 'Duplicate', 'Indirect', 'Isolated'],
								save_result_path="Expt_results/results.csv"):

	flatten = lambda l: np.array([item for sublist in l for item in sublist])
#	pred_list = flatten(pred_list)
	print(pred_list)
	y_pred = np.array([x[0].tolist() for x in pred_list])
	y_true = np.array([x[1].tolist() for x in pred_list])
	y_pred = flatten(y_pred)
	y_true = flatten(y_true)
	print(y_true)
	str_title = "Printing Classification Report : " + title + " \n\n"
	print(str_title)
	report = classification_report(y_true, y_pred, target_names=target_names, output_dict=True)
	df = pd.DataFrame(report)
	df.to_csv(save_result_path)
	print(report)

	str_title = "\n\n Printing Multilabel Confusion Matrix : " + title + " \n\n"
	print(str_title)
	print(multilabel_confusion_matrix(y_true, y_pred))

	str_title = "Printing Micro averaged scores : " + title + " \n\n"
	print(str_title)
	print(precision_recall_fscore_support(y_true, y_pred, average='micro'))
	print("\n")

	print("\n All Results Printed !! \n")


def nearest_word(OOV_word, word_embedding, word2matrix):
	try:
		OOV_word_embedding = word_embedding[OOV_word]
	except KeyError:
		OOV_word_embedding = np.random.normal(scale=0.6, size=(1, 300))
	sim_max = 0
	for i in range(len(word2matrix)):
		embedding = word2matrix[i]
		sim = cosine_similarity(embedding, OOV_word_embedding)
		if sim > sim_max:
			sim_max = sim
			index_of_closest_word = i

	return index_of_closest_word


def save_object(obj, filename):
	with open(filename, 'wb') as output:  # Overwrites any existing file.
		pickle.dump(obj, output, pickle.HIGHEST_PROTOCOL)
