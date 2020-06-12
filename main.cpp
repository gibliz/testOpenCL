// main.cpp : Этот файл содержит функцию "main". Здесь начинается и заканчивается выполнение программы.
// Первое приложение для изучения основ работы с OpenCL.
// Выводит информацию об устройствах с поддержкой OpenCL, установленных в компьютере.
// Проводит тестовые вычисления (типа бенчмарка) на всех установленных устройствах OpenCL и сравнивает
// скорость выполнения этого бенчмарка на разных устройствах.

#define __CL_ENABLE_EXCEPTIONS
#pragma comment(lib, "opencl.lib")

#include <iostream>
#include <fstream>
#include <time.h>
#include <algorithm>
#include <numeric>
#include <windows.h>
#ifdef __APPLE__
#include <OpenCL/opencl.hpp>
#else
#include <CL/cl.hpp>
#endif 

using namespace std;

// global constants
const int DATA_SIZE = 20 * (1 << 20);                    // about 20M items for benchmark
const unsigned int HOST_BENCHMARK_LOOPS = 20;           // benchmark count for host
const unsigned int KERNEL_BENCHMARK_LOOPS = 200;        // benchmark count for kernel on OpenCL device


// function prototypes
void printOpenCLInfo();                                 // print OpenCL information about installed devices
string getDeviceTypeDescription(cl_device_type dt);     // get device type string description
string formatMemSizeInfo(cl_ulong ms);                  // format memory size to user-friendly string

void prepareTestData();                                 // prepare data for benchmark
void benchmarkHost();                                   // do the host benchmark
void printBenchTimes();                                 // print bench times to console
void freeTestData();                                    // free memory allocated for benchmark
float calcMath(float a, float b);                       // math calculations for benchmark
void benchmarkOpenCLDevices();                          // do all OpenCL devices benchmark
void benchmarkOpenCLDevice(cl::Device oclDevice);       // do single OpenCL device benchmark

// global variables
float* pInputVector1;               // benchmark input data 1
float* pInputVector2;               // benchmark input data 2
float* pOutputVector;               // benchmark output data
vector<double> timeValues;          // time values for current bench
double hostBenchTimeMS = 0;         // host benchmark, ms
vector <cl::Device> oclDevices;     // list of all OpenCL devices installed in the system


int main() { // ==================================================================================================
    printOpenCLInfo();

    cout << "Press Enter to start benchmarks";
    cin.get();
    prepareTestData();
    cout << "Making benchmarks..." << endl;
    benchmarkHost();
    benchmarkOpenCLDevices();
    freeTestData();

    // wait for user keypress
    cout << "Press Enter to finish";
    cin.get();
    return 0;
} //> main() =====================================================================================================

void printOpenCLInfo() {
    cout << "OpenCL info\n\n";

    oclDevices.clear();

    // get platforms
    vector<cl::Platform> platforms;
    cl::Platform::get(&platforms);
    cout << "Platforms found: " << platforms.size() << endl << endl;

    // display each platform and each device info
    for (int iPlatf = 0; iPlatf < platforms.size(); iPlatf++) {
        // plaform info
        cout << "Platform " << iPlatf + 1 << ": " << platforms[iPlatf].getInfo<CL_PLATFORM_NAME>() << endl;
        cout << "\tVendor: " << platforms[iPlatf].getInfo<CL_PLATFORM_VENDOR>() << endl;
        cout << "\tVersion: " << platforms[iPlatf].getInfo<CL_PLATFORM_VERSION>() << endl;
        //cout << "\tProfile: " << platforms[iPlatf].getInfo<CL_PLATFORM_PROFILE>() << endl;
        //cout << "\tExtensions: " << platforms[iPlatf].getInfo<CL_PLATFORM_EXTENSIONS>() << endl;

        // get and print devices info for each platform
        vector<cl::Device> devices;
        platforms[iPlatf].getDevices(CL_DEVICE_TYPE_ALL, &devices);
        cout << "\tDevices found: " << devices.size() << endl;
        for (int iDev = 0; iDev < devices.size(); iDev++) {
            // collect this device
            oclDevices.push_back(devices[iDev]);

            // print device info
            cout << "\n\t\tDevice " << iDev + 1 << ':' << endl;
            cout << "\t\tName: " << devices[iDev].getInfo<CL_DEVICE_NAME>() << endl;
            cout << "\t\tType: " << getDeviceTypeDescription(devices[iDev].getInfo<CL_DEVICE_TYPE>()) << endl;
            //cout << "\t\tVendor: " << devices[iDev].getInfo<CL_DEVICE_VENDOR>() << endl;
            cout << "\t\tMax clock freq: " << devices[iDev].getInfo<CL_DEVICE_MAX_CLOCK_FREQUENCY>() << "MHz" << endl;
            cout << "\t\tMemory size - global: " << formatMemSizeInfo(devices[iDev].getInfo<CL_DEVICE_GLOBAL_MEM_SIZE>()) << endl;
            cout << "\t\tMemory size - global cache: " << formatMemSizeInfo(devices[iDev].getInfo<CL_DEVICE_GLOBAL_MEM_CACHE_SIZE>()) << endl;
            cout << "\t\tMemory size - local: " << formatMemSizeInfo(devices[iDev].getInfo<CL_DEVICE_LOCAL_MEM_SIZE>()) << endl;
            cout << "\t\tCompute units: " << devices[iDev].getInfo<CL_DEVICE_MAX_COMPUTE_UNITS>() << endl;
            cout << "\t\tWork group size: " << devices[iDev].getInfo<CL_DEVICE_MAX_WORK_GROUP_SIZE>() << endl;
        } //> for

        cout << endl;
    } //> for
} //> printOpenCLInfo()

string getDeviceTypeDescription(cl_device_type dt) {
    switch (dt) {
    case CL_DEVICE_TYPE_CPU:
        return "CPU";
    case CL_DEVICE_TYPE_GPU:
        return "GPU";
    case CL_DEVICE_TYPE_ACCELERATOR:
        return "Accelerator";
    case CL_DEVICE_TYPE_DEFAULT:
        return "Default";
    } //> switch
    return "";
} //> getDeviceTypeDescription()

string formatMemSizeInfo(cl_ulong ms) {
    int divides = 0;
    while (ms > 1024)
        ms /= 1024, ++divides;
    string retval = to_string(ms) + ' ';
    switch (divides) {
        case 1: retval += "KiB"; break;
        case 2: retval += "MiB"; break;
        case 3: retval += "GiB"; break;
        case 4: retval += "TiB"; break;
        case 5: retval += "PiB"; break;
        case 6: retval += "EiB"; break;
        case 7: retval += "EiB"; break;
        case 8: retval += "ZiB"; break;
        case 9: retval += "YiB"; break;
        default: retval += "???";  break;
    } //> switch
    return retval;
} //> formatMemSizeInfo()

void prepareTestData() {
    cout << "Prepare memory and data...";
    pInputVector1 = new float[DATA_SIZE];
    pInputVector2 = new float[DATA_SIZE];
    pOutputVector = new float[DATA_SIZE];

    srand(static_cast<unsigned int>( time(NULL) ));
    for (int i = 0; i < DATA_SIZE; i++) {
        pInputVector1[i] = static_cast<float>(rand() * 1000.0 / RAND_MAX);
        pInputVector2[i] = static_cast<float>(rand() * 1000.0 / RAND_MAX);
    } //> for
    cout << "done\n";
} //> prepareTestData()

void benchmarkHost() {
    cout << "\n\tBenchmarking device: Host (single thread)...";
    
    // do benchmark
    timeValues.clear();
    __int64 start_count;
    __int64 end_count;
    __int64 freq;
    float* pOutputVectorHost = new float[DATA_SIZE];

    QueryPerformanceFrequency(reinterpret_cast<LARGE_INTEGER*>(&freq));
    //cout << "freq " << freq << endl;

    for (int iBench = 0; iBench < HOST_BENCHMARK_LOOPS; iBench++) {
        QueryPerformanceCounter(reinterpret_cast<LARGE_INTEGER*>(&start_count));
        for (int iJob = 0; iJob < DATA_SIZE; iJob++) {
            //Perform calculations
            pOutputVectorHost[iJob] = calcMath(pInputVector1[iJob], pInputVector2[iJob]);
        } //> for
        QueryPerformanceCounter(reinterpret_cast<LARGE_INTEGER*>(&end_count));
        double time = 1000 * (end_count - start_count) / static_cast<double>(freq);
        //cout << "start " << start_count << ", end " << end_count << ", diff " << end_count - start_count << endl;
        timeValues.push_back(time);
    } //> for
    delete[] pOutputVectorHost;
    hostBenchTimeMS = accumulate(timeValues.begin(), timeValues.end(), 0.0) / timeValues.size();
    cout << "done\n";

    printBenchTimes();
} //> benchmarkHost()

void benchmarkOpenCLDevices() {
    // check if we have no OpenCL devices
    if (oclDevices.size() == 0) {
        cout << "No OpenCL devices found\n";
        return;
    } //> if

    // bench each OpenCL device
    for (int iDev = 0; iDev < oclDevices.size(); iDev++) {
        try {
            benchmarkOpenCLDevice(oclDevices[iDev]);
            printBenchTimes();
        }
        catch (cl::Error err) {
            cout << "OpenCL error: " << err.what() << "(" << err.err() << ")\n";
        } //> catch > try

    } //> for
} //> benchmarkOpenCLDevices()

void benchmarkOpenCLDevice(cl::Device oclDevice) {
    cout << "\n\tBenchmarking device: " << oclDevice.getInfo< CL_DEVICE_NAME>() << "...";

    // create context for device
    vector<cl::Device> contextDevices;
    contextDevices.push_back(oclDevice);
    cl::Context context(contextDevices);

    // create command queue for device
    cl::CommandQueue queue(context, oclDevice);

    // clear output vector
    fill_n(pOutputVector, DATA_SIZE, static_cast<float>(0));

    // create memory buffers for device
    cl::Buffer oclInputVector1 = cl::Buffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, DATA_SIZE * sizeof(float), pInputVector1);
    cl::Buffer oclInputVector2 = cl::Buffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, DATA_SIZE * sizeof(float), pInputVector2);
    cl::Buffer oclOutputVector = cl::Buffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, DATA_SIZE * sizeof(float), pOutputVector);

    // create kernel from source file
    ifstream sourceFile("oclFile.cl");
    std::string sourceCode(std::istreambuf_iterator<char>(sourceFile), (std::istreambuf_iterator<char>()));
    cl::Program::Sources source(1, make_pair(sourceCode.c_str(), sourceCode.length() + 1));
    cl::Program program = cl::Program(context, source);
    program.build(contextDevices);
    cl::Kernel kernel(program, "TestKernel");

    // set arguments to kernel
    int iArg = 0;
    kernel.setArg(iArg++, oclInputVector1);
    kernel.setArg(iArg++, oclInputVector2);
    kernel.setArg(iArg++, oclOutputVector);
    kernel.setArg(iArg++, DATA_SIZE);

    // prepare to performance measurement
    timeValues.clear();
    __int64 start_count;
    __int64 end_count;
    __int64 freq;
    QueryPerformanceFrequency(reinterpret_cast<LARGE_INTEGER*>(&freq));

    // run the kernel on specific ND range
    for (int iBench = 0; iBench < KERNEL_BENCHMARK_LOOPS; iBench++) {
        QueryPerformanceCounter(reinterpret_cast<LARGE_INTEGER*>(&start_count));

        queue.enqueueNDRangeKernel(kernel, cl::NullRange, cl::NDRange(DATA_SIZE), cl::NDRange(128));
        queue.finish();

        QueryPerformanceCounter(reinterpret_cast<LARGE_INTEGER*>(&end_count));
        double time = 1000.0 * (end_count - start_count) / freq;
        timeValues.push_back(time);

        // read output buffer into a host
        queue.enqueueReadBuffer(oclOutputVector, CL_TRUE, 0, DATA_SIZE * sizeof(float), pOutputVector);
    } //> for
    cout << "done\n";
} //> benchmarkOpenCLDevice()

void printBenchTimes() {
    sort(timeValues.begin(), timeValues.end());
    double totalTime = accumulate(timeValues.begin(), timeValues.end(), 0.0);
    double averageTime = totalTime / timeValues.size();
    double minTime = timeValues[0];
    double maxTime = timeValues[timeValues.size() - 1];
    //double medianTime = timeValues[timeValues.size() / 2];
    cout << "\t\tBench info: " << timeValues.size() << " runs, each on " << DATA_SIZE << " items" << endl;
    cout << "\t\tAvg: " << averageTime << " ms (" << hostBenchTimeMS / averageTime << "X faster than host)" << endl;
    //cout << "Avg: " << averageTime << " ms" << endl;
    cout << "\t\tMin: " << minTime << " ms" << endl;
    cout << "\t\tMax: " << maxTime << " ms" << endl << endl;
} //> printBenchTimes()

void freeTestData() {
    delete[] pInputVector1;
    delete[] pInputVector2;
    delete[] pOutputVector;    
} //> freeTestData()

// Запуск программы: CTRL+F5 или меню "Отладка" > "Запуск без отладки"
// Отладка программы: F5 или меню "Отладка" > "Запустить отладку"

// Советы по началу работы 
//   1. В окне обозревателя решений можно добавлять файлы и управлять ими.
//   2. В окне Team Explorer можно подключиться к системе управления версиями.
//   3. В окне "Выходные данные" можно просматривать выходные данные сборки и другие сообщения.
//   4. В окне "Список ошибок" можно просматривать ошибки.
//   5. Последовательно выберите пункты меню "Проект" > "Добавить новый элемент", чтобы создать файлы кода, или "Проект" > "Добавить существующий элемент", чтобы добавить в проект существующие файлы кода.
//   6. Чтобы снова открыть этот проект позже, выберите пункты меню "Файл" > "Открыть" > "Проект" и выберите SLN-файл.
