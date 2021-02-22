# weg-li CarML Service

![CI](https://github.com/weg-li-project/car-ml/workflows/CI/badge.svg?branch=main&event=push)

Turns images of cars into suggestions for license plate number, make and color.

## Setup

Guide on how to setup the different environments for running and deploying the CarML service.

### Prerequisites

1. Install or use Python 3.8
1. Clone this repository
1. Install (test-)requirements

        pip install -r requirements.txt
        pip install -r test-requirements.txt

1. Download the ML models from https://tubcloud.tu-berlin.de/s/Q8brqAmdNdiXpcg
   and extract checkpoints folder into the data directory
1. Install Google Cloud SDK

### Google Cloud

1. Create a Google Cloud Project with Billing activated
1. Enable the following Google APIs
    * Cloud Build API
    * Cloud Run API
    * Cloud Vision API
1. Create a service account key in IAM with the following roles
    * Editor

### Google Cloud Storage

1. Create the following buckets or make sure they exist in Cloud Storage
    * Bucket for images
    * Bucket for ML models
1. Copy ML model data into bucket for ML models

        gsutil -m cp -r ./data/checkpoints gs://${{ SERVICE_BUCKET_NAME }}

### CI/CD

1. In your Github repository settings add the following secrets
    * WEGLI_IMAGES_BUCKET_NAME - Name of the bucket where your user images are stored
    * SERVICE_BUCKET_NAME - Name of the bucket where the ML model data is stored
    * GCP_PROJECT_ID - Project ID of the Google Cloud project
    * GCP_SERVICE_ACCOUNT_KEY - Content of a json service account key with all required permissions
1. Pushes to main will now automatically deploy the API

### Deployment

Make sure you noted every needed environment variable.

Authenticate with Google Cloud and make sure your account has the necessary rights to Cloud Storage (read), Container
Registry (write)
and Cloud Run (write)

    gcloud auth login

Use gcloud in the terminal and invoke the following:

    gcloud builds submit --tag gcr.io/${{ GCP_PROJECT_ID }}/image-analysis --gcs-source-staging-dir=gs://${{ SERVICE_BUCKET_NAME }}/source
    gcloud run deploy image-analysis --image gcr.io/${{ GCP_PROJECT_ID }}/image-analysis --no-allow-unauthenticated --region=europe-west3 --cpu=4 --memory=8G --platform=managed

### Misc

For running the service locally, just set the environment variable `GOOGLE_APPLICATION_CREDENTIALS` with the path to a
service account file that covers all needed roles and run the following command in the root directory

    python main.py

## Testing

For getting test and coverage reports execute the following command from the root directory of the project

    pytest

## API

### POST /

Request Body

```json5
{
  "google_cloud_urls": [
    "gs://$BUCKET_NAME/$IMAGE_TOKEN/0.jpg",
    // ... 
    "gs://$BUCKET_NAME/$IMAGE_TOKEN/4.jpg"
  ]
}
```

Response Body

```json5
{
  "suggestions": {
    "license_plate_number": [
      "...",
      "etc."
    ],
    "color": [
      "...",
      "etc."
    ],
    "make": [
      "...",
      "etc."
    ]
  }
}
```  

## Retraining the networks

For retraining any of the models you have to arrange your data using the following structure further explained [here](https://www.tensorflow.org/api_docs/python/tf/keras/preprocessing/image_dataset_from_directory):

      main_directory/
      ...class_a/
      ......a_image_1.jpg
      ......a_image_2.jpg
      ...class_b/
      ......b_image_1.jpg
      ......b_image_2.jpg

All dependencies are already installed when all requirements from the `requirements.txt` are fulfilled.

If you are using a graphic card for training, please install all the necessary drivers as
described [here](https://www.tensorflow.org/install/gpu)

Then execute the following command and specify your data directory, the checkpoint directory where the model should be
saved to, the number of epochs to train, the patience, the number of categories that the dataset contains, and the path
to the resnet152 weights needed for training.

      python cnn_resnet152.py --data_dir <directory> --checkpoint_dir <directory> --epochs <number> --patience <number> --num_classes <number> --resnet_weights <directory>

For more information on the required arguments call

      python cnn_resnet152.py -h

## Contributing

This software is developed for free. You are welcome to contribute and support, here are a few ways:

- Report bugs, make suggestions and new ideas
- Fork the project and do pull requests

## License

TBD
