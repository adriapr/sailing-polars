import urllib.request
import json
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.ticker as plticker
import numpy as np
from math import radians, degrees, cos, sin, asin, sqrt, atan2
import time
import math

from geopy.distance import geodesic

def main():

	sns.set(style='whitegrid', context='notebook')
	# sns.set_palette('BuGn_d')

	# loc = plticker.MultipleLocator(base=1.0) # this locator puts ticks at regular intervals

	print('Reading race data...')
	with urllib.request.urlopen("https://gcloud.ingenium.net.au/racelog?racenr=15376") as url:
	    raw_data = json.loads(url.read().decode())

	# with open('racelog.json') as f:
	# 	raw_data = json.load(f)	

	# mark_pos_old = [30.6015, -17.1946]
	mark_pos = [21.4190, -75.3245]
	# print(bearing(mark_pos_old, mark_pos) )

	# mark_pos = [30.6015, -17.1946]

	print('   DONE\n')


	data = []

	for racer in raw_data['result']:

		# raw user data
		raw_ud = raw_data['result'][racer]

		# (processed) user data
		ud = {}
		ud['name'] = racer
		ud['last_speed'] = float(raw_ud['lastreport_speed'])
		ud['last_heading'] = float(raw_ud['lastreport_heading'])
		ud['last_pos'] = raw_ud['track'][0]

		ud['dists_to_mark'], ud['displacements_to_mark'], ud['displacements'], ud['positions'] = boat_distances(raw_ud, mark_pos)

		ud['dist_to_mark'] = ud['dists_to_mark'][-1]
		ud['bearing_to_mark'] = bearing(ud['last_pos'], mark_pos)

		ud['dist_travelled'] = sum(ud['displacements'])

		last_update = time.strftime('%d/%m %H:%M', time.localtime(raw_ud['timestamp']/1000))

		if ud['dist_travelled'] > 50:
			data.append(ud)

	data = sorted(data, key=lambda k: k['dist_to_mark'])
	num_tacks = len(data[0]['positions'])

	# fix issue with later starters having less items in tack list
	for racer in data:
		mising_tacks = num_tacks - len(racer['positions'])
		if mising_tacks > 0:
			racer['dists_to_mark'] 		= [racer['dists_to_mark'][0]] * mising_tacks + racer['dists_to_mark']
			racer['displacements_to_mark'] = [0] * mising_tacks + racer['displacements_to_mark']
			racer['displacements'] 		= [0] * mising_tacks + racer['displacements']
			racer['positions'] 			= [racer['positions'][0]] * mising_tacks + racer['positions']


	print('Updated: {} // Calculations towards: {}'.format(last_update, mark_pos) )

	for racer in data:
		# position data updates are mostly every 30', so use median and x2 to make speeds in knots
		time_left = racer['dist_to_mark'] / (np.median(racer['displacements_to_mark']) * 2)
		dist_to_first = racer['dist_to_mark'] - data[0]['dist_to_mark']
		# efficency = sum(racer['displacements_to_mark'][-2:]) / sum(racer['displacements'][-2:]) * 100
		last_tack_heading = bearing(racer['positions'][-2],racer['positions'][-1])

		print( 	' {:>30} >> Distance left: {:6.1f}nm (+{:6.2f}nm) @ {:.1f}° // '.format( racer['name'], racer['dist_to_mark'], dist_to_first, racer['bearing_to_mark']) + \
				'Last tack: {:5.2f}kn @ {:5.1f}°, (+{:5.2f}nm towards mark) // '.format(racer['displacements'][-1], last_tack_heading, racer['displacements_to_mark'][-1]) + \
				'Last report: {:4.1f}kn @ {}° // '.format(racer['last_speed'], racer['last_heading']) + \
				'Dist. travelled: {:5.1f}nm - time left: {:.1f}h'.format(racer['dist_travelled'], time_left) )



	plt.figure()
	ax = plt.subplot(121)

	for racer in data:
		y = racer['displacements_to_mark']
		x = range(len(y))
		plt.plot(x, y,label=racer['name'], marker='o')

	plt.ylabel('Nautical miles')
	plt.xlabel('Sector')
	# ax.xaxis.set_major_locator(loc)

	plt.legend(loc='best')
	plt.title('Distance travelled toward mark in {} (updated: {})'.format(mark_pos, last_update) )
	sns.despine()



	# ax = plt.gca()
	ax = plt.subplot(122)

	for racer in data:
		try:
			y = np.array(data[0]['dists_to_mark']) - np.array(racer['dists_to_mark'])
			x = range(len(y))
			plt.plot(x, y,label=racer['name'], marker='o')
		except:
			pass

	plt.ylabel('Nautical miles')
	plt.xlabel('Sector')
	# ax.xaxis.set_major_locator(loc)

	plt.legend(loc='best')
	plt.title('Distance to mark in {}, compared to leader (updated: {})'.format(mark_pos, last_update) )
	sns.despine()

	plt.show()



def boat_distances( ud, mark_pos ):

	dists_to_mark = []
	for p in ud['track']:
		dists_to_mark.append( geodesic(p,mark_pos).nm )

	positions = list(reversed(ud['track']))

	# Actually, track is stored from most recent point to start, so needs reversing
	dists_to_mark.reverse()

	displacements_to_mark = (-np.diff(dists_to_mark)).tolist()

	displacements = []
	for p1,p2 in zip(ud['track'][0:-1], ud['track'][1:]):
		displacements.append( geodesic(p1,p2).nm )

	displacements.reverse()

	return dists_to_mark, displacements_to_mark, displacements, positions


# def bearing(p1, p2):

#     """
#    Calculation of direction between two geographical points
#    """
#     lon1, lat1, lon2, lat2 = map(radians, [p1[0], p1[1], p2[0], p2[1]])
#     dlon = lon2-lon1
#     b = atan2(sin(dlon)*cos(lat2), cos(lat1)*sin(lat2) - sin(lat1)*cos(lat2)*cos(dlon))
#     return (degrees(b) + 360) % 360


def bearing(p1, p2):

	lat1 = math.radians(p1[0])
	lat2 = math.radians(p2[0])

	diffLong = math.radians(p2[1] - p1[1])

	x = math.sin(diffLong) * math.cos(lat2)
	y = math.cos(lat1) * math.sin(lat2) - (math.sin(lat1)
			* math.cos(lat2) * math.cos(diffLong))

	initial_bearing = math.atan2(x, y)

	# Now we have the initial bearing but math.atan2 return values
	# from -180° to + 180° which is not what we want for a compass bearing
	# The solution is to normalize the initial bearing as shown below
	initial_bearing = math.degrees(initial_bearing)
	compass_bearing = (initial_bearing + 360) % 360

	return compass_bearing

# def distance(p1, p2):
# 	# https://stackoverflow.com/a/15737218
#     """
#     Calculate the great circle distance between two points 
#     on the earth (specified in decimal degrees)
#     """
#     # convert decimal degrees to radians 
#     lon1, lat1, lon2, lat2 = map(radians, [p1[0], p1[1], p2[0], p2[1]])
#     # haversine formula 
#     dlon = lon2 - lon1 
#     dlat = lat2 - lat1 
#     a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
#     c = 2 * asin(sqrt(a)) 
#     # Radius of earth in kilometers is 6371, (* 0.539957, for nutical miles)
#     distance = (6371 * 0.539957) * c
#     return distance


if __name__ == "__main__":
	main()