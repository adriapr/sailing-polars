import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import math

import scipy.interpolate as interpolate


def main():

	sns.set_style("ticks")
	#matplotlib.rcParams['font.sans-serif'] = "Comic Sans MS" # for the radial ticks thatn I didnt manage to change otherwise

	print('Reading race data...')
	data = pd.read_csv('https://docs.google.com/spreadsheets/d/16S4WFeQv5_mVwf5IkeHNBLFwDj3kpTra4C8fNkLJJLQ/export?gid=0&format=csv', dtype='float')
	print('   DONE\n')	

	data = data.sort_values(by=['TWA'])
	# print(data)

	TWA = data['TWA'].values
	TWA_resampled = np.arange(0, 180.1, .1)

	# calculate polynomial
	if False:
		polar_fit = np.polyfit(data['TWA'], data['normalised_SPD_8nm'], 3)
		f = np.poly1d(polar_fit)
		SPD_fitted = f(TWA_resampled)

	else:
		f = interpolate.splrep(data['TWA'], data['normalised_SPD_8nm'], k=3, s=1)
		SPD_fitted = interpolate.splev(TWA_resampled, f)


	plt.figure()
	fig_grid = plt.GridSpec(2, 2)

	# --- FITTED MODEL ---
	with plt.xkcd():
		ax = plt.subplot(fig_grid[0, 1])

	ax.plot( TWA, data['normalised_SPD_8nm'].values, marker='o', color='gray', alpha=0.2, linestyle = 'None' )
	ax.plot( TWA_resampled, SPD_fitted, marker='None', linestyle = '-', color='darkorange', label='Sailing speed' )

	with plt.xkcd():
		ax.xaxis.grid(linewidth=1.0)
		ax.yaxis.grid(linewidth=1.0)
		ax.set_xticks(np.arange(0,200,20))
		plt.legend(loc='lower right')
		plt.ylabel('Sailing speed (nm)')
		sns.despine(bottom=True)


	# --- UPWIND SPEED ---
	upwind_SPD = SPD_fitted * np.cos(np.radians(TWA_resampled))
	max_id = np.argmax(upwind_SPD)
	min_id = np.argmin(upwind_SPD)

	with plt.xkcd():
		ax = plt.subplot(fig_grid[1, 1])

	ax.plot( TWA_resampled, upwind_SPD, marker='None', linestyle = '-', color='darkred', label='Upwind speed' )

	with plt.xkcd():

		ax.plot( TWA_resampled[max_id], upwind_SPD[max_id], marker='o', color='black', alpha=1, linestyle = 'None', label='Optimal sailing points')
		ax.plot( TWA_resampled[min_id], upwind_SPD[min_id], marker='o', color='black', alpha=1, linestyle = 'None')

		plt.annotate(
			'Max Upwind Speed ({:.2f}kn) at {:.1f} TWD'.format( upwind_SPD[max_id], TWA_resampled[max_id] ),
			xy=(TWA_resampled[max_id], upwind_SPD[max_id]), arrowprops=dict(arrowstyle='->'), xytext=(0, 2.2))			

		plt.annotate(
			'Max Downwind Speed ({:.2f}kn) at {:.1f} TWD'.format( -upwind_SPD[min_id], TWA_resampled[min_id] ),
			xy=(TWA_resampled[min_id], upwind_SPD[min_id]), arrowprops=dict(arrowstyle='->'), xytext=(90, -2.8))			

		ax.xaxis.grid(linewidth=1.0)
		ax.yaxis.grid(linewidth=1.0)
		ax.set_xticks(np.arange(0,200,20))
		plt.ylabel('Upwind speed (nm)')
		plt.xlabel('Wind direction (TWA)')
		plt.legend(loc='lower left')
		sns.despine()


	# FITTED MODEL POLAR
	with plt.xkcd():
		ax = plt.subplot(fig_grid[:, 0], projection='polar')

	ax.plot( np.radians(TWA), data['normalised_SPD_8nm'].values, marker='o', color='gray', alpha=0.2, linestyle = 'None', label='Measurements' )
	ax.plot( np.radians(TWA_resampled), SPD_fitted, marker='None', linestyle = '-', color='darkorange', label='Sailing speed' )

	with plt.xkcd():

		ax.plot( np.radians(TWA_resampled[max_id]), SPD_fitted[max_id], marker='o', color='black', alpha=1, linestyle = 'None', label='Optimal sailing points')
		ax.plot( np.radians(TWA_resampled[min_id]), SPD_fitted[min_id], marker='o', color='black', alpha=1, linestyle = 'None')

		plt.xlim([0, math.pi])
		ax.xaxis.grid(linewidth=1.0)

		ax.set_yticklabels([])
		# ax.set_rticks([0.5, 1, 1.5, 2])  # less radial ticks
		ax.set_theta_zero_location("N")
		# ax.set_xticks(np.arange(0,210,30))
		# plt.title('Sailing speed')
		ax.set_xticks(np.radians(np.linspace(0,180,10)))

		plt.legend(loc='lower left')
		# sns.despine()


	with plt.xkcd():
		plt.suptitle("Caribbean Rose sailing points @ 8nm winds", fontsize=20)

	plt.show()


if __name__ == "__main__":
	main()