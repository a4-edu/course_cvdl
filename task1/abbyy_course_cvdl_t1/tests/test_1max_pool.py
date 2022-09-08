import json
import os
from pathlib import Path

import pytest

from abbyy_course_cvdl_t1.max_pool import MaxPoolLayer

TESTS_PATH_FILE = os.environ.get("ABBYY_TESTS_PATH", Path(__file__).parent / 'data' / 'maxpool.json')


with open(TESTS_PATH_FILE) as f:
    TESTS = json.loads(f.read())


@pytest.fixture(params=TESTS, scope='module')
def test_data(request):
    return request.param


@pytest.fixture(scope='module')
def test_layer_cls():
    return MaxPoolLayer

from abbyy_course_cvdl_t1.tests.base import TestLayer
