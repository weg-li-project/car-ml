import os

dirname = os.path.dirname(os.path.realpath(__file__))
root_path = os.path.join(dirname, '..')

data_path = os.path.join(root_path, 'data/')
city_ids_path = os.path.join(data_path, 'city_ids.csv')

charges_schroeder_path = os.path.join(data_path, 'charges_schroeder/')

testdata_path = os.path.join(data_path, 'testdata/')
vision_api_results_path = os.path.join(testdata_path, 'vision_api_results.csv')
