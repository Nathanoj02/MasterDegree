import cv2 as cv
import sys

from algorithms import k_means, k_means_cpp, k_means_cuda, k_means_cuda_shared_mem, k_means_scikit, \
    k_means_video, k_means_video_calibration
from metrics import metrics, plot_execution_times


def test_exec_times () :
    img = cv.imread('../data/video_frame.jpg')
    plot_execution_times(img, 3, 0.5)


def test_metrics () :
    img = cv.imread('../data/video_frame.jpg')
    metrics(img, 1, 10)


def test_algorithms () :
    img = cv.imread('../data/car.jpg')
    
    # res = k_means(img, 3, 0.5)
    # cv.imwrite(f'../data/kmeans_python.jpg', res)

    res = k_means_scikit(img, 5, 0.2)
    cv.imwrite(f'../data/kmeans_scikit.jpg', res)

    res = k_means_cpp(img, 5, 0.2)
    cv.imwrite(f'../data/kmeans_cpp.jpg', res)

    res, _ = k_means_cuda(img, 5, 0.2)
    cv.imwrite(f'../data/kmeans_cuda.jpg', res)

    res, _ = k_means_cuda_shared_mem(img, 5, 0.2)
    cv.imwrite(f'../data/kmeans_cuda_shared.jpg', res)


def test_image (algorithm : callable, source_path : str, dest_path : str, k : int, stab_error : int) :
    img = cv.imread(source_path)
    res = algorithm(img, k, stab_error)

    if isinstance(res, tuple) :
        res_img, _ = res 
        cv.imwrite(dest_path, res_img)
    else :
        cv.imwrite(dest_path, res)


def test_video (source_path : str, dest_path : str, k : int, stab_error : int) :
    cap = cv.VideoCapture(source_path)
    k_means_video(cap, k, stab_error, save_path = dest_path)


def test_video_calibration (source_path : str, dest_path : str, k : int, stab_error : int) :
    cap = cv.VideoCapture(source_path)
    k_means_video_calibration (cap, k, stab_error, save_path = dest_path)


def custom_tests () :
    # test_video_calibration('../data/walking.mp4', '../data/video_res_cali.mp4', 3, 0.5)
    # test_exec_times()
    test_image(k_means_cuda_shared_mem, '../data/video_frame.jpg', '../data/res.jpg', 3, 0.5)
    pass


if __name__ == '__main__':

    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument('-o', '--operation')
    parser.add_argument('-a', '--algorithm', default='cudashared')
    parser.add_argument('-k', '--k', default=2)
    parser.add_argument('-e', '--error', default=0.5)
    parser.add_argument('-s', '--source')
    parser.add_argument('-d', '--dest')

    args = parser.parse_args()

    if len(sys.argv) == 1 :
        custom_tests()
        sys.exit(0)

    if args.operation is None :
        print('No operation passed as argument - Use -o or --operation')
        sys.exit(-1)
    
    if args.operation not in ['image', 'video'] :
        raise NotImplementedError('Operation not implemented - use \'image\' or \'video\'')
    
    if args.source is None :
        print('No input path passed as argument - Use -s or --source')
        sys.exit(-1)

    if args.dest is None :
        print('No output path passed as argument - Use -d or --dest')
        sys.exit(-1)

    k = int(args.k)
    error = float(args.error)

    if k <= 0 :
        print('K value must be greater than 0')
        sys.exit(-1)

    if error <= 0 :
        print('Error must be greater than 0')
        sys.exit(-1)
    
    match args.operation :
        case 'image' :
            match args.algorithm :
                case 'python' :
                    alg = k_means_scikit
                case 'cpp' :
                    alg = k_means_cpp
                case 'cuda' :
                    alg = k_means_cuda
                case 'cudashared' :
                    alg = k_means_cuda_shared_mem
                case _ :
                    raise NotImplementedError('Algorithm not implemented yet. Must be in [\'python\', \'cpp\', \'cuda\', \'cudashared\']')
            
            test_image(alg, args.source, args.dest, k, error)

        case 'video' :
            test_video_calibration(args.source, args.dest, k, error)
        case _ :
            raise NotImplementedError('Operation not implemented - use \'image\' or \'video\'')
    
