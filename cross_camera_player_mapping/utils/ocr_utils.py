
import easyocr

class JerseyNumberRecognizer:
    def __init__(self):
        self.reader = easyocr.Reader(['en'], gpu=False)

    def recognize(self, image):
        results = self.reader.readtext(image, detail=0)
        for text in results:
            text = text.strip()
            if text.isdigit():
                return text
        return None
