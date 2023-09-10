import json
import os
from pathlib import Path

import pytest

from a4_course_cvdl_t1.cross_entropy import CrossEntropyLoss

TESTS_PATH_FILE = os.environ.get("A4_TESTS_PATH", Path(__file__).parent / 'data' / 'crossentropy.json')


with open(TESTS_PATH_FILE) as f:
    TESTS = json.loads(f.read())


@pytest.fixture(params=TESTS, scope='module')
def test_data(request):
    return request.param


@pytest.fixture(scope='module')
def test_layer_cls():
    class CrossEntropyLossFacade(CrossEntropyLoss):
        def forward(self, input):
            pred = input[0]
            gt = input[1]
            return super().forward(pred, gt)
        def backward(self, output_grad):
            return super().backward()
    return CrossEntropyLossFacade

from a4_course_cvdl_t1.tests.base import TestLayer
