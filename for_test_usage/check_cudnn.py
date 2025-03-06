import ctypes

try:
    cudnn = ctypes.CDLL('libcudnn_ops.so')
    print("cuDNN loaded successfully(libcudnn_ops.so)")
except OSError:
    print("Unable to load cuDNN(libcudnn_ops.so)")


try:
    cudnn = ctypes.CDLL('libcudnn.so')
    print("cuDNN loaded successfully(libcudnn.so)")
except OSError:
    print("Unable to load cuDNN(libcudnn.so)")