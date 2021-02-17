import os
from behave import fixture, use_fixture


@fixture
def base_url(context, *args, **kwargs):
    context.base_url = os.environ["BASE_URL"]
    yield context.base_url


@fixture
def wegli_images_bucket_name(context, *args, **kwargs):
    context.wegli_images_bucket_name = os.environ['WEGLI_IMAGES_BUCKET_NAME']
    yield context.wegli_images_bucket_name


def before_all(context):
    use_fixture(base_url, context)
    use_fixture(wegli_images_bucket_name, context)
