import os

dirname = os.path.dirname(os.path.realpath(__file__))
root_path = os.path.join(dirname, '..')

data_path = os.path.join(root_path, 'data/')
city_ids_path = os.path.join(data_path, 'city_ids.csv')

charges_schroeder_path = os.path.join(data_path, 'charges_schroeder/')
charges_csv_filepath = os.path.join(charges_schroeder_path, 'charges.csv')

checkpoints_path = os.path.join(data_path, 'checkpoints/')
resnet_weights_filepath = os.path.join(checkpoints_path, 'resnet152_weights_tf.h5')
yolo_lp_model_path = os.path.join(checkpoints_path, 'yolo_lp/')
yolo_car_model_path = os.path.join(checkpoints_path, 'yolo_car/')
cnn_alpr_model_path = os.path.join(checkpoints_path, 'cnn_alpr/training')
cnn_color_rec_model_path = os.path.join(checkpoints_path, 'cnn_color_rec/training')
cnn_car_rec_model_path = os.path.join(checkpoints_path, 'cnn_car_rec/training')

testdata_path = os.path.join(data_path, 'testdata/')
vision_api_results_path = os.path.join(testdata_path, 'vision_api_results.csv')

yolo_cnn_path = os.path.join(root_path, 'yolo_cnn/')
car_brands_filepath = os.path.join(yolo_cnn_path, 'car_brands.txt')
car_colors_filepath = os.path.join(yolo_cnn_path, 'car_colors.txt')
