# Internal Endpoint for Image Analysis

This endpoint will be used to start the ML related tasks. Based on the image analysis it will return suggestions for license plate number and vehicle features.

## Deployment

If using the GCloud Console and having the necessary rights to deploy Cloud Functions,
invoke the following command:

    `gcloud functions deploy get_image_analysis_suggestions --runtime python38 --memory=128MB --region=europe-west3 --trigger-http`
    
For being able to invoke the function publicly add the flag `--allow-unauthenticated` to the command above.

## Testing

To test the main entrypoint execute the following command from the tests directory of the project

    pytest --disable-pytest-warnings

To get a coverage report execute the following

    pytest --disable-pytest-warnings --cov=wegliML
