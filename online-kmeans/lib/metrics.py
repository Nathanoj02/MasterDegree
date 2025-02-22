import sys
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score
import time
import os

from algorithms import k_means_cpp, k_means_cuda, k_means_cuda_shared_mem, k_means_scikit, \
    k_means_cuda_init, k_means_cuda_deinit, k_means_cuda_exec

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "cpp", "build", "python"))


def _elbow_method (img : np.ndarray, clustered_img : np.ndarray) -> float :
    '''
    Calculate sum of squared error

    Parameters
    ----------
    img : np.ndarray, 3D
        Image
    clustered_img : np.ndarray, 3D
        Clustered image

    Returns
    -------
    Sum of squared error
    '''
    error = np.sum((img - clustered_img) ** 2)
    return error


def elbow_method (img : np.ndarray, start_k : int, end_k : int) :
    '''
    Plot elbow method

    Parameters
    ----------
    img : np.ndarray, 3D
        Image
    start_k : int
        k to begin calculating metrics
    end_k : int
        k to end calculating metrics
    '''
    error_arr = np.array([])

    for k in range (start_k, end_k + 1) :
        res = k_means_cpp(img, k, 0.01)
        error = _elbow_method(img, res)
        error_arr = np.append(error_arr, error)

    # Plot
    plt.plot(np.arange(start_k, end_k + 1), error_arr, '-o')
    plt.title('Elbow method')
    plt.xlabel('Number of clusters k')
    plt.ylabel('Total within sum of squares')
    plt.savefig('../data/stats/elbow.jpg')


def _silhouette_method (img : np.ndarray, clustered_img : np.ndarray) -> float :
    '''
    Calculate average silhouette score

    Parameters
    ----------
    img : np.ndarray, 3D
        Image
    clustered_img : np.ndarray, 3D
        Clustered image

    Returns
    -------
    Average silhouette score
    '''
    img_flat = img.reshape(-1, 3)
    clustered_img_flat = clustered_img.reshape(-1, 3)

    # Preprocess clustered image
    _, labels = np.unique(clustered_img_flat, axis=0, return_inverse=True)

    silhouette_avg = silhouette_score(img_flat, labels, metric='euclidean')

    return silhouette_avg


def silhouette_method (img : np.ndarray, start_k : int, end_k : int) :
    '''
    Plot average silhouette method

    Parameters
    ----------
    img : np.ndarray, 3D
        Image
    start_k : int
        k to begin calculating metrics
    end_k : int
        k to end calculating metrics
    '''
    sil_arr = []
    
    for k in range (start_k, end_k + 1) :
        print(k)
        sys.stdout.flush()
        res = k_means_cpp(img, k, 0.01)
        
        silhouette_avg = _silhouette_method(img, res)
        sil_arr.append(silhouette_avg)

    # Plot
    plt.plot(np.arange(start_k, end_k + 1), sil_arr, '-o')
    plt.title('Silhouette method')
    plt.xlabel('Number of clusters k')
    plt.ylabel('Average silouette width')
    plt.savefig('../data/stats/silhouette.jpg')


def metrics (img : np.ndarray, start_k : int, end_k : int) :
    '''
    Plot elbow method and average silhouette method

    Parameters
    ----------
    img : np.ndarray, 3D
        Image
    start_k : int
        k to begin calculating metrics
    end_k : int
        k to end calculating metrics
    '''
    error_arr = []
    sil_arr = []

    for k in range(start_k, end_k + 1) :
        print(f'Iteration {k} of {end_k}')
        dusk = time.time()

        res = k_means_cpp(img, k, 0.01)

        error = _elbow_method(img, res)
        silhouette_avg = _silhouette_method(img, res) if k >= 2 else 0

        error_arr.append(error)
        sil_arr.append(silhouette_avg)

        dawn = time.time()
        print('Time spent: {:.2f} mins'.format((dawn - dusk) / 60))
    
    # Plot elbow
    plt.figure()
    plt.plot(np.arange(start_k, end_k + 1), error_arr, '-o')
    plt.title('Elbow method')
    plt.xlabel('Number of clusters k')
    plt.ylabel('Total within sum of squares')
    plt.savefig('../data/stats/elbow.jpg')

    # Plot silhouette
    plt.figure()
    plt.plot(np.arange(start_k, end_k + 1), sil_arr, '-o')
    plt.title('Silhouette method')
    plt.xlabel('Number of clusters k')
    plt.ylabel('Average silhouette width')
    plt.savefig('../data/stats/silhouette.jpg')


def _execution_time_step (algorithm : callable, img : np.ndarray, k : int, stab_error : int, tries : int = 10) -> list[float] :
    '''
    Calculate execution times of the algorithm parameter

    Parameters
    ----------
    algorithm : callable
        k-means algorithm on which calculate the execution times
    img : np.ndarray, 3D
        Image
    k : int
        Number of clusters
    stab_error : float
        Stabilization error
    tries : int
        Number of repetitions

    Returns
    -------
    List of execution times
    '''
    times = []

    for i in range(tries) :
        dusk = time.time()
        algorithm(img, k, stab_error)
        dawn = time.time()
        times.append(dawn - dusk)
    
    return times


def execution_time_avg (img : np.ndarray, k : int, stab_error : int, tries : int = 10) -> tuple :
    '''
    Calculate means and standard deviations of execution times of algorithms

    Parameters
    ----------
    img : np.ndarray, 3D
        Image
    k : int
        Number of clusters
    stab_error : float
        Stabilization error
    tries : int
        Number of repetitions on which the average is calculated

    Returns
    -------
    Tuple with means and standard deviations
    '''
    # Set max error and best prototypes
    best_error = float("inf")
    best_prototypes = np.random.randint(0, 256, size = (k, 3))

    times = _execution_time_step(k_means_scikit, img, k, stab_error, tries)
    py_time_avg = np.mean(times)
    py_time_std = np.std(times)
    
    times = _execution_time_step(k_means_cpp, img, k, stab_error, tries)
    cpp_time_avg = np.mean(times)
    cpp_time_std = np.std(times)
    
    # Do once to "warm up" GPU
    k_means_cuda(img, k, stab_error)
    
    times = _execution_time_step(k_means_cuda, img, k, stab_error, tries)
    cuda_time_avg = np.mean(times)
    cuda_time_std = np.std(times)

    times = _execution_time_step(k_means_cuda_shared_mem, img, k, stab_error, tries)
    cuda_shared_time_avg = np.mean(times)
    cuda_shared_time_std = np.std(times)

    times = []
    dev = k_means_cuda_init(img, k)
    
    for i in range(tries) :
        dusk = time.time()
        k_means_cuda_exec(img, k, stab_error, dev=dev, use_shared_mem = True)
        dawn = time.time()
        times.append(dawn - dusk)
    
    k_means_cuda_deinit(dev)

    cuda_video_time_avg = np.mean(times)
    cuda_video_time_std = np.std(times)

    # Get prototypes for next algorithm
    dev = k_means_cuda_init(img, k)

    for i in range(tries) :
        res, prototypes = k_means_cuda_exec(img, k, stab_error, dev=dev, use_shared_mem = True)

        # Perform sum of squares metric
        error = np.sum((img - res) ** 2)

        if error < best_error :
            best_error = error
            best_prototypes = np.copy(prototypes)
    
    k_means_cuda_deinit(dev)

    times = []
    dev = k_means_cuda_init(img, k)

    for i in range(tries) :
        dusk = time.time()
        k_means_cuda_exec(img, k, stab_error, dev=dev, use_shared_mem = True, prototypes = best_prototypes)
        dawn = time.time()
        times.append(dawn - dusk)

    k_means_cuda_deinit(dev)

    cuda_cali_time_avg = np.mean(times)
    cuda_cali_time_std = np.std(times)

    return py_time_avg, py_time_std, \
        cpp_time_avg, cpp_time_std, \
        cuda_time_avg, cuda_time_std, \
        cuda_shared_time_avg, cuda_shared_time_std, \
        cuda_video_time_avg, cuda_video_time_std, \
        cuda_cali_time_avg, cuda_cali_time_std


def plot_execution_times (img : np.ndarray, k : int, stab_error : int, tries : int = 10) :
    '''
    Make charts on execution times, Fps and Speedups

    Parameters
    ----------
    img : np.ndarray, 3D
        Image
    k : int
        Number of clusters
    stab_error : float
        Stabilization error
    tries : int
        Number of repetitions on which the average is calculated
    '''
    py_time_avg, _, cpp_time_avg, _, cuda_time_avg, _, cuda_shared_time_avg, _, \
        cuda_video_time_avg, _, cuda_cali_time_avg, _ = execution_time_avg (img, k, stab_error, tries)

    labels = ['Scikit-learn', 'C++', 'CUDA single image', 'CUDA shared memory', 'CUDA video', 'CUDA w/ calibration']
    times = [py_time_avg, cpp_time_avg, cuda_time_avg, cuda_shared_time_avg, cuda_video_time_avg, cuda_cali_time_avg]
    times_strings = ["{:.4f}".format(t) for t in times]

    plt.figure()
    plt.bar(labels, times)
    plt.xticks(rotation=60)
    plt.gcf().set_constrained_layout(True)

    # Add labels
    for i in range (len(labels)):
        plt.text(i, times[i], times_strings[i], ha = 'center')

    plt.title('Average k-means execution time')
    plt.xlabel('Method')
    plt.ylabel('Execution time [s]')
    plt.savefig('../data/stats/times.jpg')

    # FPS chart
    fps = [1 / t for t in times]
    fps_strings = ["{:.1f}".format(f) for f in fps]

    plt.figure()
    plt.bar(labels, fps)
    plt.xticks(rotation=60)
    plt.gcf().set_constrained_layout(True)

    for i in range (len(labels)):
        plt.text(i, fps[i], fps_strings[i], ha = 'center')

    plt.title('Average k-means Frames Per Second')
    plt.xlabel('Method')
    plt.ylabel('FPS')
    plt.savefig('../data/stats/fps.jpg')

    # Speedups chart
    speedups = [times[0] / t for t in times]
    speedups_strings = ["{:.2f}".format(s) for s in speedups]

    plt.figure()
    plt.bar(labels, speedups)
    plt.xticks(rotation=60)
    plt.gcf().set_constrained_layout(True)

    for i in range (len(labels)):
        plt.text(i, speedups[i], speedups_strings[i], ha = 'center')

    plt.title('Average k-means Speed-ups')
    plt.xlabel('Method')
    plt.ylabel('Speed-up')
    plt.savefig('../data/stats/speedup.jpg')

