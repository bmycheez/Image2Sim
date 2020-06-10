import cv2
import pytesseract
import argparse
import os
import subprocess
from PIL import Image


parser = argparse.ArgumentParser(description="Image2Text")
parser.add_argument("--image_name", type=str, default="3_6_8.png", help="")
opt = parser.parse_args()


def main():
    if os.path.splitext(opt.image_name)[1] in ['.heic']:
        destination = os.path.splitext(opt.image_name)[0] + '.jpg'
        source = opt.image_name
        subprocess.call(['tifig', '-p', '-q', '100', source, destination])
        opt.image_name = destination
    img_cv = cv2.imread(opt.image_name)
    img_rgb = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
    print(pytesseract.image_to_string(img_rgb))
    """
    im_gray = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
    (thresh, im_bw) = cv2.threshold(im_gray, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    # although thresh is used below, gonna pick something suitable
    im_bw = cv2.threshold(im_gray, thresh, 255, cv2.THRESH_BINARY)[1]
    print(pytesseract.image_to_string(im_bw))
    # """


def get_captcha_text_from_captcha_image(captcha_path):
    # Preprocess the image befor OCR
    tif_file = preprocess_image_using_opencv(captcha_path)
    # Perform OCR using tesseract-ocr library
    image = Image.open(tif_file)
    ocr_text = pytesseract.image_to_string(image)
    alphanumeric_text = ''.join(e for e in ocr_text)

    return alphanumeric_text


def binarize_image_using_opencv(captcha_path, binary_image_path='input-black-n-white.jpg'):
    img = cv2.imread(captcha_path)
    im_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    (thresh, im_bw) = cv2.threshold(im_gray, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    # although thresh is used below, gonna pick something suitable
    im_bw = cv2.threshold(im_gray, thresh, 255, cv2.THRESH_BINARY)[1]
    cv2.imwrite(binary_image_path, im_bw)
    return binary_image_path


def preprocess_image_using_opencv(captcha_path):
    bin_image_path = binarize_image_using_opencv(captcha_path)

    im_bin = Image.open(bin_image_path)

    basewidth = 340  # in pixels
    wpercent = (basewidth/float(im_bin.size[0]))
    hsize = int((float(im_bin.size[1])*float(wpercent)))
    big = im_bin.resize((basewidth, hsize), Image.NEAREST)

    # tesseract-ocr only works with TIF so save the bigger image in that format
    ext = ".tif"
    tif_file = "input-NEAREST.tif"
    big.save(tif_file)

    return tif_file


if __name__ == "__main__":
    main()
    # print(get_captcha_text_from_captcha_image(opt.image_name))
