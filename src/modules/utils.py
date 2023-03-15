import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


class Statistics:
	def __init__(self, columns: list, init_vals: list = None):
		stats = {}
		for i in range(len(columns)):
			if init_vals:
				assert len(columns) == len(init_vals), "columns len not equal to init_vals length"
				if isinstance(init_vals[i], list):
					stats[columns[i]] = init_vals[i]
				else:
					raise ValueError(f"Column data type expected list, given {type(init_vals[i])}")
			else:
				self.stats = stats[columns[i]] = []
		self.__stats = stats

	def add_single_data_entry(self, col_names: list, col_data: list):
		assert len(col_names) == len(col_data), "columns len not equal to init_vals length"
		for i in range(len(col_names)):
			self.__stats[col_names[i]].append(col_data[i])

	def get_statistics(self) -> dict:
		return self.__stats


class Plotter:

	@staticmethod
	def plot_simple_loss(loss_history, path):
		plt.figure(figsize=(12, 12))
		plt.plot((np.arange(len(loss_history)) + 1), loss_history)
		plt.xlabel("Step", fontsize=12)
		plt.ylabel("Cross-Entropy Loss", fontsize=12)
		plt.savefig(path)

	@staticmethod
	def plot_simple_acc(acc_history, path):
		plt.figure(figsize=(12, 12))
		plt.plot((np.arange(len(acc_history)) + 1), acc_history)
		plt.xlabel("Step", fontsize=12)
		plt.ylabel("Accuracy", fontsize=12)
		plt.savefig(path)

	@staticmethod
	def plot_double_acc(acc_history_train, acc_history_val, path):
		plt.figure(figsize=(12, 12))
		plt.plot(
			(np.arange(len(acc_history_train)) + 1),
			acc_history_train, color='b', label="Train"
		)
		plt.plot(
			(np.arange(len(acc_history_val)) + 1),
			acc_history_val, color='r', linestyle="--", label="Test"
		)
		plt.xlabel("Step", fontsize=12)
		plt.ylabel("Accuracy", fontsize=12)
		plt.legend(loc="lower right", fontsize=10)
		plt.savefig(path)

	@staticmethod
	def plot_double_loss(loss_history_train, loss_history_val,  path):
		plt.figure(figsize=(12, 12))
		plt.plot(
			(np.arange(len(loss_history_train)) + 1),
			loss_history_train, color='b', label="Train"
		)
		plt.plot(
			(np.arange(len(loss_history_val)) + 1),
			loss_history_val, color='r', linestyle="--", label="Test"
		)
		plt.xlabel("Step", fontsize=12)
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
