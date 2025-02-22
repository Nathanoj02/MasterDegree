import numpy as np
import sys
import os
import cv2 as cv
from sklearn.cluster import KMeans

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "cpp", "build", "python"))

import pysignals


def k_means (img : np.ndarray, k : int, stab_error : float, max_iterations : int = 300) -> np.ndarray :
    '''
    K-means algorithm

    Parameters
    ----------
    img : np.ndarray, 3D
        Image
    k : int
        Number of clusters
    stab_error :
        Stabilization error
    max_iterations : int
        Maximum number of iterations

    Returns
    -------
    Clustered image
    '''
    assert img is not None, 'Image not loaded correctly'

    # Create k prototypes with random values
    prototypes = [list(np.random.choice(range(256), size=3)) for _ in range(k)]

    print(f'Initial prototypes = {prototypes}', end='\n\n')

    assigned_img = np.zeros((img.shape[0], img.shape[1]), dtype=int)   # Map : pixels -> cluster number

    # Loop until prototypes are stable
    differences = np.full(shape = k, fill_value = 500)  # Array of differences wrt. last iteration

    sums = [np.zeros(3, dtype=int) for _ in range(k)]  # Array for calculating new means
    counts = np.zeros(k, dtype=int)

    iteration_count = 0
    while any(differences > stab_error) and iteration_count < max_iterations :
        print(f'Iteration {iteration_count}')
        old_values = prototypes.copy()  # Save old values for calculating differences
        # Reset sums and counts
        [arr.fill(0) for arr in sums]
        counts.fill(0)

        # Associate each pixel to nearest prototype (with Euclidian distance)
        for i, pixel in enumerate(img.reshape(-1, 3)) :
            distances = [np.linalg.norm(pixel - prot) for prot in prototypes]
            assigned_prototype_index = np.argmin(distances)
            assigned_img[i // assigned_img.shape[1], i % assigned_img.shape[1]] = assigned_prototype_index

            sums[assigned_prototype_index] += pixel
            counts[assigned_prototype_index] += 1
        
        # Update values of the prototypes to the means of the associated pixels
        for i in range(len(prototypes)) :
            if (counts[i] != 0) :
                prototypes[i] = np.divide(sums[i], counts[i])
        
        print(f'Prototypes = {prototypes}')

        differences = np.linalg.norm(np.subtract(old_values, prototypes), axis=1)

        print(f'Differences = {differences}', end='\n\n')
        iteration_count += 1

    # Substitute each pixel with the corresponding prototype value
    res = np.zeros(img.shape)
    for i in range(img.shape[0]) :
        for j in range(img.shape[1]) :
            res[i, j] = prototypes[assigned_img[i, j]].astype(np.uint8)
    
    return res


def k_means_cpp (img : np.ndarray, k : int, stab_error : float, max_iterations : int = 300) -> np.ndarray :
    '''
    K-means algorithm executed in C++

    Parameters
    ----------
    img : np.ndarray, 3D
        Image
    k : int
        Number of clusters
    stab_error : float
        Stabilization error 
    max_iterations : int
        Maximum number of iterations

    Returns
    -------
    Clustered image
    '''
    assert img is not None, 'Image not loaded correctly'

    res = pysignals.seq.k_means(img, k, stab_error, max_iterations)
    return res


def k_means_cuda (img : np.ndarray, k : int, stab_error : float, \
                  max_iterations : int = 300, prototypes : np.ndarray = None) -> np.ndarray :
    '''
    K-means algorithm executed in CUDA C++

    Parameters
    ----------
    img : np.ndarray, 3D
        Image
    k : int
        Number of clusters
    stab_error : float
        Stabilization error
    max_iterations : int
        Maximum number of iterations
    prototypes : np.ndarray, 2D
        Array of prototypes to use instead of random ones

    Returns
    -------
    Clustered image
    '''
    assert img is not None, 'Image not loaded correctly'

    dev = pysignals.par.init_k_means(img.shape[0], img.shape[1], k)
    res, updated_prototypes = k_means_cuda_exec(img, k, stab_error, dev, max_iterations, False, prototypes)
    pysignals.par.deinit_k_means(dev)
    return res, updated_prototypes


def k_means_cuda_shared_mem (img : np.ndarray, k : int, stab_error : float, \
                             max_iterations : int = 300, prototypes : np.ndarray = None) -> np.ndarray :
    '''
    K-means algorithm executed in CUDA C++

    Parameters
    ----------
    img : np.ndarray, 3D
        Image
    k : int
        Number of clusters
    stab_error : float
        Stabilization error
    max_iterations : int
        Maximum number of iterations
    prototypes : np.ndarray, 2D
        Array of prototypes to use instead of random ones

    Returns
    -------
    Clustered image
    '''
    assert img is not None, 'Image not loaded correctly'

    dev = pysignals.par.init_k_means(img.shape[0], img.shape[1], k)
    res, updated_prototypes = k_means_cuda_exec(img, k, stab_error, dev, max_iterations, True, prototypes)
    pysignals.par.deinit_k_means(dev)
    return res, updated_prototypes


def k_means_cuda_init (img : np.ndarray, k : int) :
    assert img is not None, 'Image not loaded correctly'

    dev = pysignals.par.init_k_means(img.shape[0], img.shape[1], k)
    return dev


def k_means_cuda_deinit (dev) :
    pysignals.par.deinit_k_means(dev)


def k_means_cuda_exec (img : np.ndarray, k : int, stab_error : float, dev, max_iterations : int = 300, \
                       use_shared_mem : bool = True, prototypes : np.ndarray = None) :
    use_stored_prototypes = True
    
    if prototypes is None :
        use_stored_prototypes = False
        prototypes = np.zeros((k, 3), dtype=np.int8)

    res, updated_prototypes = pysignals.par.k_means(img, k, stab_error, max_iterations, dev, use_shared_mem, prototypes, use_stored_prototypes)
    return res, updated_prototypes


def k_means_scikit (img : np.ndarray, k : int, stab_error : float, max_iterations : int = 300) :
    '''
    K-means algorithm from Scikit Learn

    Parameters
    ----------
    img : np.ndarray, 3D
        Image
    k : int
        Number of clusters
    stab_error : float
        Stabilization error
    max_iterations : int
        Maximum number of iterations

    Returns
    -------
    Clustered image
    '''
    assert img is not None, 'Image not loaded correctly'

    # Reshape image to 2D array
    img_reshaped = img.reshape((-1, 3))

    # Apply K-Means clustering
    kmeans = KMeans(n_clusters = k, init = 'random', tol = stab_error, n_init = 1, max_iter = max_iterations)
    kmeans.fit(img_reshaped)
    res = kmeans.cluster_centers_[kmeans.labels_]

    # Convert float values to uint8
    res = np.clip(res, 0, 255).astype(np.uint8)

    # Reshape back to original image shape
    return res.reshape(img.shape)


def k_means_video (cap : cv.VideoCapture, k : int, stab_error : float, save_path : str, max_iterations : int = 300) :
    '''
    K-means algorithm applied on a video

    Parameters
    ----------
    cap : cv.VideoCapture
        Source video
    k : int
        Number of clusters
    stab_error : float
        Stabilization error
    save_path : str
        Path where the clustered video is saved
    max_iterations : int
        Maximum number of iterations
    '''
    assert cap.isOpened(), 'Video not opened correctly'

    fps = cap.get(cv.CAP_PROP_FPS)
    frame_num = 1
    tot_frames = int(cap.get(cv.CAP_PROP_FRAME_COUNT))

    while cap.isOpened() :
        # Capture each frame
        ret, frame = cap.read()

        if not ret :
            break
    
        if frame_num == 1 :
            video_res = cv.VideoWriter(save_path, cv.VideoWriter_fourcc(*'mp4v'), fps, (frame.shape[1], frame.shape[0]))
            dev = k_means_cuda_init(frame, k)
        
        res, _ = k_means_cuda_exec(frame, k, stab_error, dev, max_iterations, use_shared_mem = True)
        video_res.write(res)

        print(f'\rFrame {frame_num} of {tot_frames} : {frame_num * 100 // tot_frames} %', end = '')

        frame_num += 1

    k_means_cuda_deinit(dev)
    print() # Newline

    # Save video
    video_res.release()


def k_means_video_calibration (cap : cv.VideoCapture, k : int, stab_error : float, save_path : str, \
                               max_iterations : int = 300, calibration_frames : int = 10) :
    '''
    K-means algorithm applied on a video with calibration for first frames

    Parameters
    ----------
    cap : cv.VideoCapture
        Source video
    k : int
        Number of clusters
    stab_error : float
        Stabilization error
    save_path : str
        Path where the clustered video is saved
    max_iterations : int
        Maximum number of iterations
    calibration_frames : int
        Number of frames in which calibration is operative
    '''
    assert cap.isOpened(), 'Video not opened correctly'

    fps = cap.get(cv.CAP_PROP_FPS)
    frame_num = 1
    tot_frames = int(cap.get(cv.CAP_PROP_FRAME_COUNT))

    # Set max error and best prototypes
    best_error = float("inf")
    best_prototypes = np.random.randint(0, 256, size = (k, 3))

    while cap.isOpened() :
        # Capture each frame
        ret, frame = cap.read()

        if not ret :
            break
    
        if frame_num == 1 :
            video_res = cv.VideoWriter(save_path, cv.VideoWriter_fourcc(*'mp4v'), fps, (frame.shape[1], frame.shape[0]))
            dev = k_means_cuda_init(frame, k)
        
        # During calibration
        if frame_num <= calibration_frames :
            res, prototypes = k_means_cuda_exec(frame, k, stab_error, dev, max_iterations, use_shared_mem = True)
        
            # Perform sum of squares metric
            error = np.sum((frame - res) ** 2)

            if error < best_error :
                best_error = error
                best_prototypes = np.copy(prototypes)

        # After calibration
        else :
            res, _ = k_means_cuda_exec(frame, k, stab_error, dev, max_iterations, use_shared_mem = True, prototypes = best_prototypes)

        video_res.write(res)

        print(f'\rFrame {frame_num} of {tot_frames} : {frame_num * 100 // tot_frames} %', end = '')

        frame_num += 1

    k_means_cuda_deinit(dev)
    print() # Newline

    # Save video
    video_res.release()

        