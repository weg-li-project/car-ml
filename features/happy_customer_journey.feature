Feature: Happy Customer Journey
  As an API consumer
  I want to experience the best case scenario
  so that it keeps me happy

  Background: API url and Cloud Storage urls
    Given the API url "/"
      And at most five Cloud Storage image urls
        """
        gs://{WEGLI_IMAGES_BUCKET_NAME}/test/0.jpg
        gs://{WEGLI_IMAGES_BUCKET_NAME}/test/1.jpg
        gs://{WEGLI_IMAGES_BUCKET_NAME}/test/2.jpg
        """

  Scenario: Detect car color
     When I post the Cloud Storage image urls to the API url
     Then return the status "200"
      And return the detected car colors

  Scenario: Detect car make
     When I post the Cloud Storage image urls to the API url
     Then return the status "200"
      And return the detected car makes

  Scenario: Detect license plate number
     When I post the Cloud Storage image urls to the API url
     Then return the status "200"
      And return the detected license plate numbers
