import unittest
from model_original import MLModelHandler, DLModelHandler

text = ['음악이 주가 된, 최고의 음악영화',
        '발연기 도저히 못보겠다 진짜 이렇게 연기를 못할거라곤 상상도 못했네',
        '실화라서더욱아름답고찡하네요...많이울었어요벌써4년이란시간이흘렀네요']


class TestModelHandler(unittest.TestCase):
    def test_ml_model_handler(self):
        predicted = ['negative', 'negative', 'negative']
        ml_handler = MLModelHandler()
        for s, p in zip(text, predicted):
            result = ml_handler.handle(s)
            self.assertEqual(result[0][0], p)

    def test_dl_model_handler(self):
        predicted = ['positive', 'negative', 'positive']
        ml_handler = MLModelHandler()
        for s, p in zip(text, predicted):
            result = ml_handler.handle(s)
            self.assertEqual(result[0][0], p)


if __name__ == '__main__':
    unittest.main()
