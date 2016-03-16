import unittest
from scripts.segmentation import EmailSegmentation

class SimplisticTest(unittest.TestCase):

    def test(self):

    	segmenter = EmailSegmentation()
        data = open('sample_email.txt','r').read().splitlines()
        dictionary, LSI, clf = segmenter.load_models('models/')
        segs = segmenter.sample_segmented(data, clf, dictionary, LSI)

        self.failUnless(segs[0][0] in segmenter.categories)


if __name__ == '__main__':
    unittest.main()