#include <iostream>
#include <fstream>
#include <CL/opencl.hpp>
#include <popl/popl.hpp>
#include <limits.h>
#include <timer.hpp>

#define INT4LEN 4

static const char kernelString[] = R"CLC(
__kernel void int4matmul(__global const char *src_a,
                    __global const char *src_b,
                    __global int *dst_c,
                    int m, int n, int k)
{
    int gx = get_global_id(0);
    int gy = get_global_id(1);
    int gid = get_group_id(0);
    int lx = get_local_id(0);

    __local char tmp[8];

    if (gx < n && gy < m)
    {
        char sum = 0;
        for (int ik=0; ik<k; ik++)
        {
            char tmp_src_a = src_a[gy*k/2+ik/2];
            char tmp_src_b = src_b[ik*n/2+gx/2];
            char a, b;

            if (ik % 2 == 0)
                a = (tmp_src_a&0xf0) >> 4;
            else
                a = tmp_src_a&0x0f;

            if (gx % 2 == 0)
                b = (tmp_src_b&0xf0) >> 4;
            else
                b = tmp_src_b&0x0f;
            sum += a*b;
        }

        tmp[lx] = sum&0x0f;
        barrier(CLK_LOCAL_MEM_FENCE);

        if (lx == 0)
        {
            dst_c[(gy*n+gx)/8] = (tmp[0]<<28) | (tmp[1]<<24) | (tmp[2]<<20) | (tmp[3]<<16) |
                                (tmp[4]<<12) | (tmp[5]<<8) | (tmp[6]<<4) | (tmp[7]<<0);
        }
    }
}
)CLC";

void cpu_int4_matmul(char *src_a, char *src_b, int *dst_c, int m, int n, int k)
{
    char *dst = (char *)malloc(m*n);

    for (int im=0; im<m; im++)
    {
        for (int in=0; in<n; in++)
        {
            char sum = 0;
            for (int ik=0; ik<k; ik++)
            {
                char tmp_src_a = src_a[im*k/2+ik/2];
                char tmp_src_b = src_b[ik*n/2+in/2];
                char a, b;

                char ik_idx = 1 - (ik % 2);
                char in_idx = 1 - (in % 2);

                a = (tmp_src_a & (0x0f << (ik_idx*4))) >> (ik_idx*4);
                b = (tmp_src_b & (0x0f << (in_idx*4))) >> (in_idx*4);

                // if (ik % 2 == 0)
                //     a = (tmp_src_a&0xf0) >> 4;
                // else
                //     a = tmp_src_a&0x0f;

                // if (in % 2 == 0)
                //     b = (tmp_src_b&0xf0) >> 4;
                // else
                //     b = tmp_src_b&0x0f;

                sum += a*b;
            }
            dst[im*n+in] = sum;
        }
    }

    for (int i=0; i<m*n/8; i++)
    {
        dst_c[i] = ((int)(dst[8*i+0]&0x0f)<<28) | ((int)(dst[8*i+1]&0x0f)<<24) | ((int)(dst[8*i+2]&0x0f)<<20) | ((int)(dst[8*i+3]&0x0f)<<16) | \
                    ((int)(dst[8*i+4]&0x0f)<<12) | ((int)(dst[8*i+5]&0x0f)<<8) | ((int)(dst[8*i+6]&0x0f)<<4)  | ((int)(dst[8*i+7]&0x0f)<<0);
    }

    free(dst);
}

void print_result(int *result, int len)
{
    for (int im=0; im<len; im++)
    {
        if (im % 8 == 0)
            printf("\n");
        printf("%08x ", result[im]);
    }
    printf("\n");
}

int compare_result(int *dst, int *src, int len)
{
    for (int i=0; i<len; i++)
    {
        if (dst[i] != src[i])
            return 1;
    }
    return 0;
}


int main(int argc, char *argv[])
{
    int platformIndex = 0;
    int deviceIndex = 0;
    int matrixM = 16;
    int matrixK = 16;
    int matrixN = 16;
    CPerfCounter time;

    {
        popl::OptionParser op("This test case is used to compute matrix multiplications of type int4.\n\
            C(M, N) = A(M, K) * B(K, N)\nSupported Options");
        op.add<popl::Value<int>>("p", "platform", "Platform Index", platformIndex, &platformIndex);
        op.add<popl::Value<int>>("d", "device", "Device Index", deviceIndex, &deviceIndex);
        op.add<popl::Value<int>>("m", "matrixM", "Matrix M size", matrixM, &matrixM);
        op.add<popl::Value<int>>("k", "matrixK", "Matrix K size", matrixK, &matrixK);
        op.add<popl::Value<int>>("n", "matrixN", "Matrix N size", matrixN, &matrixN);

        bool printUsage = false;
        try {
            op.parse(argc, argv);
        } catch (std::exception& e) {
            fprintf(stderr, "Error: %s\n\n", e.what());
            printUsage = true;
        }

        if (printUsage || !op.unknown_options().empty() || !op.non_option_args().empty()) {
            fprintf(stderr,
                "Usage: copybufferkernel [options]\n"
                "%s", op.help().c_str());
            return -1;
        }
    }

    std::vector<cl::Platform> platforms;
    cl::Platform::get(&platforms);

    printf("Running on platform: %s\n",
        platforms[platformIndex].getInfo<CL_PLATFORM_NAME>().c_str() );

    std::vector<cl::Device> devices;
    platforms[platformIndex].getDevices(CL_DEVICE_TYPE_ALL, &devices);

    printf("Running on device: %s\n",
        devices[deviceIndex].getInfo<CL_DEVICE_NAME>().c_str() );

    cl::Context context{devices[deviceIndex]};
    cl::CommandQueue commandQueue = cl::CommandQueue{context, devices[deviceIndex]};

    cl::Program program{ context, kernelString };
    program.build();
    cl::Kernel kernel = cl::Kernel{ program, "int4matmul" };

    int src_a_size = INT4LEN*matrixM*matrixK/8;
    int src_b_size = INT4LEN*matrixK*matrixN/8;
    int dst_c_size = INT4LEN*matrixM*matrixN/8;

    char *src_a = (char *)malloc(src_a_size);
    char *src_b = (char *)malloc(src_b_size);
    int *gpu_dst_c = (int *)malloc(dst_c_size);
    int *cpu_dst_c = (int *)malloc(dst_c_size);
    memset(src_a, 0, src_a_size);
    memset(src_b, 0, src_b_size);
    memset(gpu_dst_c, 0, dst_c_size);
    memset(cpu_dst_c, 0, dst_c_size);

    for (int m=0; m<src_a_size; m++)
    {
        // src_a[m] = 0x12;
        src_a[m] = rand() & 0xff;
    }

    for (int k=0; k<src_b_size; k++)
    {
        // src_b[k] = 0xff;
        src_b[k] = rand() & 0xff;
    }

    time.Start();
    cpu_int4_matmul(src_a, src_b, cpu_dst_c, matrixM, matrixN, matrixK);
    time.Stop();
    // print_result(cpu_dst_c, dst_c_size/4);
    printf("int4matmul cpu tims is: %.10lf\n", time.GetElapsedTime());

    cl::Buffer bufSrcA = cl::Buffer(context, CL_MEM_READ_ONLY, src_a_size);
    cl::Buffer bufSrcB = cl::Buffer(context, CL_MEM_READ_ONLY, src_b_size);
    cl::Buffer bufDstC = cl::Buffer(context, CL_MEM_ALLOC_HOST_PTR, dst_c_size);

    commandQueue.enqueueWriteBuffer(bufSrcA, CL_TRUE, 0, src_a_size, src_a);
    commandQueue.enqueueWriteBuffer(bufSrcB, CL_TRUE, 0, src_b_size, src_b);

    kernel.setArg(0, bufSrcA);
    kernel.setArg(1, bufSrcB);
    kernel.setArg(2, bufDstC);
    kernel.setArg(3, matrixM);
    kernel.setArg(4, matrixN);
    kernel.setArg(5, matrixK);

    time.Reset();
    time.Start();
    commandQueue.enqueueNDRangeKernel(kernel, cl::NullRange, cl::NDRange{(size_t)matrixM, (size_t)matrixN}, cl::NDRange{8, 1});
    commandQueue.finish();
    time.Stop();

    commandQueue.enqueueReadBuffer(bufDstC, CL_TRUE, 0, dst_c_size, gpu_dst_c);
    // print_result(gpu_dst_c, dst_c_size/4);
    printf("int4matmul gpu tims is: %.10lf\n", time.GetElapsedTime());

    /* compare cpu & gpu result */
    int ret = compare_result(cpu_dst_c, gpu_dst_c, dst_c_size/4);

    (ret == 0) ? printf("int4 matmul: cpu & gpu results compare pass.\n") : printf("int4 matmul: cpu & gpu results compare fail.\n");

    free(src_a);
    free(src_b);
    free(gpu_dst_c);
    free(cpu_dst_c);

    return 0;
}