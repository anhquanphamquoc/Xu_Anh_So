from flask import Flask, render_template, request
from symspellpy import SymSpell, Verbosity
import cv2
import numpy as np
from PIL import Image
import easyocr
import pkg_resources
import unidecode

app = Flask(__name__)

class SpellCorrect():
    def __init__(self, ngram_path) -> None:
        self.sym_spell = SymSpell(max_dictionary_edit_distance=2, prefix_length=7)
        
        # Đọc từ điển n-gram từ tệp tin
        ngram_dictionary_path = pkg_resources.resource_filename(    
            "symspellpy", ngram_path)
        
        self.sym_spell.load_dictionary(ngram_dictionary_path, term_index=0, count_index=1)
        
    def __call__(self, input):
        # Kiểm tra chính tả và gợi ý sửa lỗi
        suggestions = self.sym_spell.lookup_compound(
            input, max_edit_distance=2, transfer_casing=True)
        
        if suggestions:
            suggestion_term = suggestions[0].term
            if unidecode.unidecode(suggestion_term) != unidecode.unidecode(input):
                return suggestion_term

        return input

spell_corrector = SpellCorrect("symspellpy/2_gram.txt") 

# Định nghĩa hàm tiền xử lý hình ảnh
def preprocess_image(image_path):
    # Đọc hình ảnh bằng OpenCV
    image = cv2.imread(image_path)

    # Bước 1: Chuyển đổi hình ảnh sang định dạng xám (grayscale)
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Bước 2: Cân bằng độ sáng (histogram equalization)
    equalized_image = cv2.equalizeHist(gray_image)

    # Bước 3: Tăng độ tương phản (Contrast Limited Adaptive Histogram Equalization - CLAHE)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced_image = clahe.apply(equalized_image)

    return enhanced_image

reader = easyocr.Reader(['vi'])

@app.route('/', methods=['GET', 'POST'])
def index():
    recognized_text = ''

    if request.method == 'POST':
        # Nhận dữ liệu từ form
        uploaded_image = request.files['image']

        if uploaded_image.filename != '':
            # Lưu hình ảnh tạm thời lên máy chủ
            image_path = 'temp_image.jpg'
            uploaded_image.save(image_path)

            # Tiền xử lý hình ảnh
            preprocessed_image = preprocess_image(image_path)

            # Sử dụng easyocr để nhận dạng ký tự từ hình ảnh đã tiền xử lý
            results = reader.readtext(preprocessed_image)
            if uploaded_image.filename:
                # Kiểm tra chính tả và sửa lỗi
                input_text = results[0][1] if results else ''
                corrected_text = spell_corrector(input_text)

            # Lấy văn bản nhận dạng được từ easyocr
            text = ' '.join([result[1] for result in results])
            recognized_text = text

    return render_template('index.html', recognized_text=recognized_text)

if __name__ == '__main__':
    app.run(debug=True, port=5000, host='0.0.0.0', use_reloader=False)
