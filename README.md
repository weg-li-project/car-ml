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

### Deployment
Make sure you noted every needed environment variable.

Authenticate with Google Cloud
and make sure your account has the necessary rights to 
Cloud Storage (read), Container Registry (write) 
and Cloud Run (write)

    gcloud auth login

Use gcloud in the terminal and invoke the following:

    gcloud builds submit --tag gcr.io/${{ GCP_PROJECT_ID }}/image-analysis --gcs-source-staging-dir=gs://${{ SERVICE_BUCKET_NAME }}/source
    gcloud run deploy image-analysis --image gcr.io/${{ GCP_PROJECT_ID }}/image-analysis --no-allow-unauthenticated --region=europe-west3 --memory=4G --platform=managed

### Misc
For running the service locally, just set the environment variable `GOOGLE_APPLICATION_CREDENTIALS` with the path to a service account file that covers all needed roles and run the following command in the root directory

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
        "make": ["...", "etc."]
    }
}
```  
  
## Contributing
This software is developed for free. You are welcome to contribute and support, here are a few ways:
- Report bugs, make suggestions and new ideas
- Fork the project and do pull requests

## License
TBD
