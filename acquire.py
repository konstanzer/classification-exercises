import pandas as pd
from pydataset import data
import env


def get_connection(db, user=env.username, host=env.host, password=env.password):
    return f'mysql+pymysql://{user}:{password}@{host}/{db}'

def get_titanic_data():
	return pd.read_sql('SELECT * FROM passengers', get_connection('titanic_db'))

def get_iris_data():
	query = """SELECT * FROM measurements JOIN species USING(species_id)"""
	return pd.read_sql(query, get_connection('iris_db'))


if __name__ == '__main__':

	#iris df
	iris = data('iris')
	print(iris.head(3))
	print(iris.info()) #150x5
	print(iris.describe()) #scaling is required

	#excel sheet
	sheet_url = 'https://docs.google.com/spreadsheets/d/1ikh-N0tMyhQ9ulnUcdBwPXZJBgHzcdnWoIK17jEy58Y/edit#gid=1379271646'    
	csv_export_url = sheet_url.replace('/edit#gid=', '/export?format=csv&gid=')
	df_sheet = pd.read_csv(csv_export_url)
	df_sheet_sample = df_sheet.head(100)
	print(df_sheet.info()) #7049x12, six obj.
	print(df_sheet.describe()) #six numeric

	titanic = get_titanic_data()
	print(titanic.head(3))
	print(titanic.info()) #891x13
	print(titanic.describe())

	#nominal/categorical & ordinal classes 
	print(f'survived: {titanic.survived.unique()}')
	print(f'sex: {titanic.sex.unique()}')
	print(f'embarked: {titanic.embarked.unique()}')

	#caching overwrites existing file
	titanic.to_csv("titanic.csv")
	get_iris_data().to_csv("iris.csv")


