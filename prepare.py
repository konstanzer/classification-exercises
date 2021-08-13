from acquire import get_iris_data
import pandas as pd


def prep_iris():

	iris = get_iris_data()
	iris = iris.drop(['species_id', 'measurement_id'], axis=1)
	iris = iris.rename(columns={'species_name':'species'})
	#this creates a df without the string label
	dum = pd.get_dummies(iris, columns=['species'])
	#but I want to keep it
	iris = pd.concat([iris.species, dum], axis=1)

	return iris


if __name__ == '__main__':

	print(prep_iris().head())