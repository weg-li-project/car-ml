# CarML Service
![CI](https://github.com/weg-li-project/car-ml/workflows/CI/badge.svg?branch=main&event=push)

Turns images of cars into suggestions for license plate number, make and color.

## Setup
Describes how to prepare the local environment to be able to test and deploy this service.

### Prerequisites
Describe how to prepare your environments.

Locally do the following
1. Install or use Python 3.8
1. Clone this repository
1. Install (test-)requirements 
    
        pip install -r requirements.txt
        pip install -r test-requirements.txt
    
1. Google Cloud SDK installed with access to gcloud

In your GCP Console do the following
1. Google Cloud Project with Billing activated
1. Activate the following Google APIs
    * Cloud Build
    * Cloud Run
1. Create a Service Account with the following Roles in Google IAM
    * Editor
1. Create the following Buckets or make sure they exist in Google Cloud Storage
    * Bucket for images
    * Bucket for models
1. Copy ML model data into bucket for models
    
    gsutil -m cp -r ./data/checkpoints gs://${{ SERVICE_BUCKET_NAME }}

In your Github repository add the following secrets to your repo for CI
* WEGLI_IMAGES_BUCKET_NAME - Name of the bucket where your user images are stored
* SERVICE_BUCKET_NAME - Name of the bucket where the ML model data is stored
* GCP_PROJECT_ID - Project ID of the Google Cloud project
* GCP_SERVICE_ACCOUNT_KEY - Content of a json service account key with all required permissions (See Deployment)

### Deployment
Describes how to manually and locally deploy this service.

If using the GCloud Console and having the necessary rights to Cloud Storage (read), Container Registry (write) and Cloud Run (write),
invoke the following:

First get the model data by running in the root directory

    gsutil -m cp -r gs://${{ SERVICE_BUCKET_NAME }}/checkpoints ./data
    
Then run the following two commands

    gcloud builds submit --tag gcr.io/${{ GCP_PROJECT_ID }}/image-analysis  missing flags
    gcloud run deploy image-analysis --image gcr.io/${{ GCP_PROJECT_ID }}/image-analysis --no-allow-unauthenticated --region=europe-west3 --memory=4G --platform=managed

### Misc
To run the service locally, just set the environment variable `GOOGLE_APPLICATION_CREDENTIALS` with the path to a service account file that covers all needed roles and any further needed ones and run the following command in the root directory

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
        "license_plate_number": ["...", "etc."],
        "color": ["...", "etc."],
        "make": ["...", "etc."],
        "model": ["...", "etc."]
    }
}
```