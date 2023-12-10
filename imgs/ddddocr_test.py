import ddddocr

ocr = ddddocr.DdddOcr()

with open('wmQI.png', 'rb') as f:
    image_bytes = f.read()

res = ocr.classification(image_bytes)
print(res)
