import unittest
import unionfind


class UnionFindTest(unittest.TestCase):

    def setUp(self):
        self.forest = unionfind.UnionFind(10)

    def test_basic(self):
        for i in range(10):
            self.assertEqual(self.forest.find(i), i)

        self.forest.union(1,2)
        self.assertEqual(self.forest.find(1), self.forest.find(2))
        self.assertEqual(self.forest.n_sets, 9)

        self.forest.union(2,3)
        self.forest.union(3,4)
        for i in range(5, 10):
            self.assertNotEqual(self.forest.find(1), self.forest.find(i))

        self.forest.union(5,6)
        self.forest.union(6,7)
        self.forest.union(7,8)
        self.forest.union(8,9)
        self.assertEqual(self.forest.n_sets, 3)

        self.forest.union(0,9)
        self.assertEqual(self.forest.n_sets, 2)
        self.assertEqual(self.forest.find(9), self.forest.find(0))
        self.assertNotEqual(self.forest.find(9), self.forest.find(1))


if __name__ == "__main__":
    unittest.main()
