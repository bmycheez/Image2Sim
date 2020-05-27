import cv2
import pytesseract
import argparse

parser = argparse.ArgumentParser(description="Image2Text")
parser.add_argument("--image_name", type=str, default="", help="")
opt = parser.parse_args()


def main():
    img_cv = cv2.imread(opt.image_name)
    img_rgb = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
    print(pytesseract.image_to_string(img_rgb))


if __name__ == "__main__":
    main()
