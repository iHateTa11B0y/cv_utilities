from cvuts.coco_utils import COCO
import argparse

def main():
    parser = argparse.ArgumentParser(description='show coco json')
    parser.add_argument("--file", default="", metavar="FILE", help="path to json file")
    parser.add_argument("--imids", default="", type=str, help="image ids seprated by ,")
    parser.add_argument("--types", default="", type=str, help="type ids seprated by ,")
    args = parser.parse_args()

    cc = COCO(args.file)
    typeIds = list(map(int,args.types.replace(' ','').split(','))) if args.types else []
    imids = list(map(int,args.imids.replace(' ','').split(','))) if args.imids else []

    cc.showImgs(imgIds=imids, areaRng=[], typeIds=typeIds) 
    
if __name__=='__main__':
    main()
