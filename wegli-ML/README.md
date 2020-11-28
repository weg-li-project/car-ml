# Internal Endpoint for Image Analysis

This endpoint will be used to start the ML related tasks. Based on the image analysis it will return suggestions for licens plate number and vehicle features.

## Deploy

If using the GCloud Console and having the necessary rights to deploy Cloud Functions,
invoke the following command:

    `gcloud functions deploy alpr_analysis --runtime python38 --memory=128MB --region=europe-west3 --trigger-http`
    
For being able to invoke the function publicly add the flag `--allow-unauthenticated` to the command above.