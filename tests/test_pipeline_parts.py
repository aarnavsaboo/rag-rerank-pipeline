import unittest

from rag_rerank_pipeline.records import Document, Hit
from rag_rerank_pipeline.retrieval import BM25
from rag_rerank_pipeline.fusion import rrf
from rag_rerank_pipeline.packing import pack


class Tests(unittest.TestCase):
    def test_bm25(self):
        docs = [Document("a","hybrid retrieval combines rankings"), Document("b","local model inference")]
        self.assertEqual(BM25(docs).search("retrieval")[0].id, "a")

    def test_rrf(self):
        a = Hit("a","a",1)
        b = Hit("b","b",1)
        rows = rrf([[a,b],[b,a]])
        self.assertEqual({x.id for x in rows}, {"a","b"})

    def test_packing(self):
        rows = [Hit("a","x"*100,1), Hit("b","y"*100,1)]
        packed = pack(rows, 160)
        self.assertGreater(len(packed.ids), 0)
        self.assertLessEqual(packed.chars, 160)


if __name__ == "__main__":
    unittest.main()
