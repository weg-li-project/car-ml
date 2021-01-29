# Internal Endpoint for Image Analysis
This endpoint will be used to start the ML related tasks. Based on the image analysis it will return suggestions for license plate number and vehicle features.

## Deployment

### Via Cloud Run
If using the GCloud Console and having the necessary rights to Cloud Storage (read), Container Registry (write) and Cloud Run (write),
invoke the following command:

    gsutil -m cp -r gs://${{ secrets.SERVICE_BUCKET_NAME }}/checkpoints ./alpr_yolo_cnn
    gcloud builds submit --tag gcr.io/${{ secrets.GCP_PROJECT_ID }}/image-analysis
    gcloud run deploy image-analysis --image gcr.io/${{ secrets.GCP_PROJECT_ID }}/image-analysis --no-allow-unauthenticated --region=europe-west3 --memory=4G --platform managed

### Via Cloud Functions
If using the GCloud Console and having the necessary rights to deploy Cloud Functions,
invoke the following command:

    gcloud functions deploy get_image_analysis_suggestions --runtime python38 --memory=128MB --region=europe-west3 --trigger-http
    
### Misc
For being able to invoke the function publicly add the flag `--allow-unauthenticated` to the commands with `--no-allow-unauthenticated`.

## Testing
To test the main entrypoint execute the following command from the tests directory of the project

    pytest --disable-pytest-warnings .

To get a coverage report execute the following

    pytest --disable-pytest-warnings --cov .