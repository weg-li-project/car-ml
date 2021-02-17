import re
from typing import Final

import requests
from behave import *

from requests import Response

FIVE_MINUTES_IN_SECONDS: Final = 60 * 5


@given('the API url "{api_url}"')
def step_impl(context, api_url):
    context.api_url = context.base_url + api_url


@step("at most five Cloud Storage image urls")
def step_impl(context):
    context.cloud_storage_urls = list(map(lambda x: re.sub('\r', '', x), re.sub('{WEGLI_IMAGES_BUCKET_NAME}', context.wegli_images_bucket_name, context.text).split('\n')))


@when("I post the Cloud Storage image urls to the API url")
def step_impl(context):
    json = {'google_cloud_urls': context.cloud_storage_urls}
    context.response = requests.post(context.api_url, json=json, timeout=FIVE_MINUTES_IN_SECONDS)


@then('return the status "{status_code}"')
def step_impl(context, status_code):
    response: Response = context.response
    assert response.status_code == int(status_code)


@step("return the detected car colors")
def step_impl(context):
    response: Response = context.response
    assert len(response.json()['suggestions']['color']) > 0


@step("return the detected car makes")
def step_impl(context):
    response: Response = context.response
    assert len(response.json()['suggestions']['make']) > 0


@step("return the detected license plate numbers")
def step_impl(context):
    response: Response = context.response
    assert len(response.json()['suggestions']['license_plate_number']) > 0
