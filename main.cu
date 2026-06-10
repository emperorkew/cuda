#include <iostream>
__global__ void hello() {
    printf("Hello from the GPU!\n");

}
int main() {

    hello<<<1,10>>>();
    // Kernel launches are async — the CPU doesn't wait for the GPU to finish.
    // Without a sync point, main() returns before the GPU is done and printf buffers are never flushed.
    // cudaDeviceSynchronize() is preferred over cudaDeviceReset() because it only waits for completion,
    // while cudaDeviceReset() destroys the entire CUDA context, preventing any further CUDA calls.
    cudaDeviceSynchronize();

    std::cout << "Hello, World!" << std::endl;
    return 0;
}
