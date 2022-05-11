from unittest import TestCase

from fedl.simulator import random_federation


class TestFederation(TestCase):

    def test_random_federation(self):
        try:
            random_federation(8, 3, 11, 42)
        except Exception:
            self.fail('no exception expected')
