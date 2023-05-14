import unittest
from unittest.mock import MagicMock
import math

from tf_pydf import Model, Doc, _compute_idf, _compute_tf


class TestModel(unittest.TestCase):
    def setUp(self):
        self.model = Model()

    def test_add_doc(self):
        self.model.add_doc(1, ["apple", "banana", "apple"])
        self.assertEqual(self.model.docs[1].tf["apple"], 2)
        self.assertEqual(self.model.df["apple"], 1)

    def test_remove_doc(self):
        self.model.add_doc(1, ["apple", "banana", "apple"])
        self.model.remove_doc(1)
        self.assertNotIn(1, self.model.docs)
        self.assertNotIn("apple", self.model.df)

    def test_contains(self):
        self.model.add_doc(1, ["apple", "banana", "apple"])
        self.assertTrue(self.model.contains(1))
        self.assertFalse(self.model.contains(2))

    def test_from_dict(self):
        doc_dict = {1: ["apple", "banana"], 2: ["orange"]}
        model = Model.from_dict(doc_dict)
        self.assertEqual(len(model.docs), 2)
        self.assertEqual(len(model.df), 3)

    def test_search_query(self):
        doc_dict = {1: ["apple", "banana", "apple"], 2: ["orange", "banana"]}
        self.model = Model.from_dict(doc_dict)
        result = self.model.search_query(["apple", "banana"])
        self.assertEqual(len(result), 2)
        self.assertEqual(result[0][0], 1)
        self.assertEqual(result[1][0], 2)

    def test_compute_tf(self):
        doc = MagicMock(count=3, tf={"apple": 2, "banana": 1})
        tf = _compute_tf("apple", doc)
        self.assertEqual(tf, 2 / 3)

    def test_compute_idf(self):
        df = {"apple": 2, "banana": 1}
        idf = _compute_idf("apple", 3, df)
        self.assertEqual(idf, math.log10(3 / 2))
