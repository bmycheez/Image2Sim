import argparse

parser = argparse.ArgumentParser(description="Text2Sim")
parser.add_argument("--txt_name1", type=str, default="", help="")
parser.add_argument("--txt_name2", type=str, default="", help="")
opt = parser.parse_args()


def main():
    srcTxt = opt.txt_name1
    cmpTxt = opt.txt_name2

    f1 = open(srcTxt, "r")
    f2 = open(cmpTxt, "r")

    srcData = f1.read().split(" ")
    # print(srcData)
    cmpData = f2.read().split(" ")
    # print(cmpData)

    cnt = 0
    allcnt = len(cmpData)
    searched = []
    print("----------------")

    for cmpline in cmpData:
        for srcline in srcData:
            if srcline == cmpline:
                if cmpline not in searched:
                    searched.append(cmpline)
                    print(cmpline, end=" / ")
                    cnt += 1

    print()
    print("----------------")
    # print(cnt, type(cnt), allcnt, type(allcnt))
    print("Similarity = %.2f%%" % (cnt/allcnt*100))

    f1.close()
    f2.close()


if __name__ == "__main__":
    main()