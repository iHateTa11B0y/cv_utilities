if __name__=='__main__':
    import sys
    cvloader = ImageLoader('opencv', 'bgr')
    piloader = ImageLoader('pil', 'bgr')

    cvimg = cvloader.load(sys.argv[1])
    pilimg = piloader.load(sys.argv[1])

    cv2.imshow('pil', np.asarray(pilimg))
    cv2.imshow('cv2', cvimg)
    cv2.imshow('diff', np.abs(np.asarray(pilimg) - cvimg))
    cv2.waitKey()

    diff = np.mean(np.abs(np.asarray(pilimg) - cvimg))
    print('diff', diff)
