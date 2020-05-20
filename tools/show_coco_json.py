from cvuts.coco_utils import COCO
import argparse

def main():
    parser = argparse.ArgumentParser(description='show coco json')
    parser.add_argument("--file", default="", metavar="FILE", help="path to json file")
    parser.add_argument("--imids", default="", type=str, help="image ids seprated by ,")
    parser.add_argument("--types", default="", type=str, help="type ids seprated by ,")
    parser.add_argument("--cats", default="", type=str, help="type ids seprated by ,")
    parser.add_argument("--skip", default=1, type=int, help="skip image")
    parser.add_argument("-v", "--vis", default=False, action='store_true', help="show or not")
    args = parser.parse_args()

    cc = COCO(args.file)
    typeIds = list(map(int,args.types.replace(' ','').split(','))) if args.types else []
    imids = list(map(int,args.imids.replace(' ','').split(','))) if args.imids else []
    cats = list(map(int,args.cats.replace(' ','').split(','))) if args.cats else []
    if args.vis:
        cc.showImgs(imgIds=imids, catIds=cats, areaRng=[], typeIds=typeIds,skip=args.skip) 
    
if __name__=='__main__':
    main()
