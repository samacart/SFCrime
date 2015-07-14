import pandas as pd
import numpy as np

def write(res, fname):
	'''
	write takes a numpy array of prediction in label encoder
	format (i.e. int 0 - 38), and a desired filename and
	convert into the required output format
	'''
	rows = len(res)
	ret = np.zeros((rows,39))
	for i in range(rows):
		j = res[i]
		ret[i][j] = 1
	df = pd.DataFrame(ret)
	colnames = ['ARSON', 'ASSAULT', 'BAD CHECKS', 'BRIBERY', 'BURGLARY', 'DISORDERLY CONDUCT',
				'DRIVING UNDER THE INFLUENCE', 'DRUG/NARCOTIC', 'DRUNKENNESS', 'EMBEZZLEMENT',
				'EXTORTION', 'FAMILY OFFENSES', 'FORGERY/COUNTERFEITING', 'FRAUD', 'GAMBLING',
				'KIDNAPPING', 'LARCENY/THEFT', 'LIQUOR LAWS', 'LOITERING', 'MISSING PERSON',
				'NON-CRIMINAL', 'OTHER OFFENSES', 'PORNOGRAPHY/OBSCENE MAT', 'PROSTITUTION',
				'RECOVERED VEHICLE', 'ROBBERY', 'RUNAWAY', 'SECONDARY CODES',
				'SEX OFFENSES FORCIBLE', 'SEX OFFENSES NON FORCIBLE', 'STOLEN PROPERTY',
				'SUICIDE', 'SUSPICIOUS OCC', 'TREA', 'TRESPASS', 'VANDALISM', 'VEHICLE THEFT',
				'WARRANTS', 'WEAPON LAWS']
	
	df.index.name = 'Id'
	df.columns = colnames
	df.reset_index(inplace=True)
	print "Writing to {}...".format(fname)
	df.to_csv(fname, index=False)
	#print df.head()
	print "Write complete."
	r, c = df.shape
	print "Total number of columns: {}".format(c)
	print "Total number of rows: {}".format(r)
	d = df.describe()
	print d.ix['mean',:]
	return df

if __name__=="__main__":
	arr = np.array([np.random.randint(0,39) for i in range(10)])
	write(arr,'test.csv')
