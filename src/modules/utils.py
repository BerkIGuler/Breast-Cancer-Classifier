import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


class Plotter:

	@staticmethod
	def plot_simple_loss(loss_history, path):
		plt.figure(figsize=(12, 12))
		plt.plot((np.arange(len(loss_history)) + 1) * 100, loss_history)
		plt.xlabel("Iteration", fontsize=12)
		plt.ylabel("Cross-Entropy Loss", fontsize=12)
		plt.savefig(path)

	@staticmethod
	def plot_simple_acc(acc_history, path):
		plt.figure(figsize=(12, 12))
		plt.plot((np.arange(len(acc_history)) + 1) * 100, acc_history)
		plt.xlabel("Iteration", fontsize=12)
		plt.ylabel("Accuracy", fontsize=12)
		plt.savefig(path)

	@staticmethod
	def plot_double_acc(acc_history_train, acc_history_val, path):
		plt.figure(figsize=(12, 12))
		plt.plot(
			(np.arange(len(acc_history_train)) + 1) * 100,
			acc_history_train, color='b', label="Train"
		)
		plt.plot(
			(np.arange(len(acc_history_val)) + 1) * 100,
			acc_history_val, color='r', linestyle="--", label="Test"
		)
		plt.xlabel("Iteration", fontsize=12)
		plt.ylabel("Accuracy", fontsize=12)
		plt.legend(loc="lower right", fontsize=10)
		plt.savefig(path)

	@staticmethod
	def plot_double_loss(loss_history_train, loss_history_val,  path):
		plt.figure(figsize=(12, 12))
		plt.plot(
			(np.arange(len(loss_history_train)) + 1) * 100,
			loss_history_train, color='b', label="Train"
		)
		plt.plot(
			(np.arange(len(loss_history_val)) + 1) * 100,
			loss_history_val, color='r', linestyle="--", label="Test"
		)
		plt.xlabel("Iteration", fontsize=12)
		plt.ylabel("Cross Entropy Loss", fontsize=12)
		# plt.ylim([0, 1])
		plt.legend(loc="lower left", fontsize=10)
		plt.savefig(path)

	@staticmethod
	def plot_cm(y_test, y_pred, class_list, save_path):

		cm = confusion_matrix(y_test, y_pred)
		disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_list)
		fig, ax = plt.subplots(figsize=(12, 12))
		disp.plot(ax=ax)
		plt.savefig(save_path)
