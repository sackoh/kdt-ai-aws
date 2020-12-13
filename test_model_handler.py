import unittest
from model import MLModelHandler, DLModelHandler

text = ['음악이 주가 된, 최고의 음악영화',
        '발연기 도저히 못보겠다 진짜 이렇게 연기를 못할거라곤 상상도 못했네',
        '실화라서더욱아름답고찡하네요...많이울었어요벌써4년이란시간이흘렀네요']


class TestModelHandler(unittest.TestCase):
    def test_ml_model_handler(self):
        predicted = ['positive', 'negative', 'negative']
        ml_handler = MLModelHandler()
        results = ml_handler.handle(text)
        for label, pred in zip(results[0], predicted):
            self.assertEqual(label, pred)

    def test_dl_model_handler(self):
        predicted = ['positive', 'negative', 'positive']
        dl_handler = DLModelHandler()
        results = dl_handler.handle(text)
        for label, pred in zip(results[0], predicted):
            self.assertEqual(label, pred)


if __name__ == '__main__':
    unittest.main()
