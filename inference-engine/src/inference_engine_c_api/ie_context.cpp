// Copyright (c) 2018 Intel Corporation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

/**
 * @brief A header file that provides wrapper for IE context object
 * @file ie_context.h
 */
#include "ie_context.h"

using namespace std;

#define CHECK_INDEX(_INDEX, _SIZE)    \
  if (_INDEX < 0 || _INDEX >= _SIZE) {      \
      throw std::runtime_error("Index out of range"); \
  }


namespace InferenceEngine {

CIEContext::CIEContext()
{
    bModelLoaded = false;
    bModelCreated = false;
    targetDevice = InferenceEngine::TargetDevice::eCPU;
    inputImageSize.height = 0;
    inputImageSize.width = 0;
}

CIEContext::~CIEContext()
{

}

CIEContext::CIEContext(IEConfig * config)
{
    bModelLoaded = false;
    bModelCreated = false;
    inputImageSize.height = 0;
    inputImageSize.width = 0;
    targetDevice = InferenceEngine::TargetDevice::eCPU;
    Init(config);
    bModelLoaded = true;
}

void CIEContext::loadModel(IEConfig * config)
{
    if (bModelLoaded) return;

    std::string path("");
    if (config->pluginPath)
        path.assign(config->pluginPath);

    InferenceEngine::PluginDispatcher dispatcher({ path, "", "" });
    targetDevice = getDeviceFromId(config->targetId);
    /** Loading plugin for device **/
    plugin = dispatcher.getPluginByDevice(getDeviceName(targetDevice));
    enginePtr = plugin;
    if (nullptr == enginePtr) {
        std::cout << "Plugin path is not found!" << std::endl;
        std::cout << "Plugin Path =" << path << std::endl;
    }
    std::cout << "targetDevice:" << getDeviceName(targetDevice) << std::endl;

    /*If CPU device, load default library with extensions that comes with the product*/
    if (config->targetId == IE_CPU) {
        /**
        * cpu_extensions library is compiled from "extension" folder containing
        * custom MKLDNNPlugin layer implementations. These layers are not supported
        * by mkldnn, but they can be useful for inferring custom topologies.
        **/
        plugin.AddExtension(std::make_shared<Extensions::Cpu::CpuExtensions>());
    }

    if (nullptr != config->cpuExtPath) {
        std::string cpuExtPath(config->cpuExtPath);
        // CPU(MKLDNN) extensions are loaded as a shared library and passed as a pointer to base extension
        auto extensionPtr = InferenceEngine::make_so_pointer<InferenceEngine::IExtension>(cpuExtPath);
        plugin.AddExtension(extensionPtr);
        std::cout << "CPU Extension loaded: " << cpuExtPath << endl;
    }
    if (nullptr != config->cldnnExtPath) {
        std::string cldnnExtPath(config->cldnnExtPath);
        // clDNN Extensions are loaded from an .xml description and OpenCL kernel files
        plugin.SetConfig({ { InferenceEngine::PluginConfigParams::KEY_CONFIG_FILE, cldnnExtPath } });
        std::cout << "GPU Extension loaded: " << cldnnExtPath << endl;
    }

    /** Setting plugin parameter for collecting per layer metrics **/
    if (config->perfCounter > 0) {
//        plugin.SetConfig({ { InferenceEngine::PluginConfigParams::KEY_PERF_COUNT, InferenceEngine::PluginConfigParams::YES } });
    }

    std::string modelFileName(config->modelFileName);
    if (modelFileName.empty()) {
        std::cout << "Model file name is empty!" << endl;
        return;
    }

    xmlFile = GetFileNameNoExt(config->modelFileName) + ".xml";
    std::string networkFileName(xmlFile);
    networkReader.ReadNetwork(networkFileName);

    binFile = GetFileNameNoExt(config->modelFileName) + ".bin";
    std::string weightFileName(binFile);
    networkReader.ReadWeights(weightFileName);

    network = networkReader.getNetwork();
    inputsInfo = network.getInputsInfo();
    outputsInfo = network.getOutputsInfo();

    bModelLoaded = true;
}

void CIEContext::createModel(IEConfig * config)
{
    if (!bModelLoaded) {
        std::cout << "Please load the model firstly!" << endl;
        return;
    }

    if (bModelCreated) return;

    // prepare the input and output Blob
    // Set the precision of intput/output data provided by the user, should be called before load of the network to the plugin

    setModelInputInfo(&config->inputInfos);
    setModelOutputInfo(&config->outputInfos);

    executeNetwork = plugin.LoadNetwork(network, {});
    inferRequest = executeNetwork.CreateInferRequest();
    bModelCreated = true;
}

void CIEContext::Init(IEConfig * config)
{
    if (!bModelLoaded) {
        loadModel(config);
    }
}

void CIEContext::setTargetDevice(InferenceEngine::TargetDevice device)
{
    targetDevice = device;
}

void CIEContext::setBatchSize(const size_t size)
{
    network.setBatchSize(size);
}

size_t CIEContext::getBatchSize()
{
    return network.getBatchSize();
}

void CIEContext::forwardSync()
{
    inferRequest.Infer();
}

void CIEContext::forwardAsync()
{
    inferRequest.StartAsync();
    inferRequest.Wait(IInferRequest::WaitMode::RESULT_READY);
}

unsigned int CIEContext::getInputSize()
{
    if (!bModelLoaded) {
        std::cout << "Please load the model firstly!" << endl;
        return 0;
    }
    return inputsInfo.size();
}

unsigned int CIEContext::getOutputSize()
{
    if (!bModelLoaded) {
        std::cout << "Please load the model firstly!" << endl;
        return 0;
    }
    return outputsInfo.size();
}

void CIEContext::setInputPresion(unsigned int idx, IEPrecisionType precision)
{
    CHECK_INDEX(idx, inputsInfo.size());
    /** Iterating over all input blobs **/
    auto item = inputsInfo.begin();
    std::advance(item, idx);
    Precision inputPrecision = getPrecisionByEnum(precision);
    item->second->setPrecision(inputPrecision);
}

void CIEContext::setOutputPresion(unsigned int idx, IEPrecisionType precision)
{
    CHECK_INDEX(idx, outputsInfo.size());
    /** Iterating over all output blobs **/
    auto item = inputsInfo.begin();
    std::advance(item, idx);
    Precision outputPrecision = getPrecisionByEnum(precision);
    item->second->setPrecision(outputPrecision);
}

void CIEContext::setInputLayout(unsigned int idx, IELayoutType layout)
{
    CHECK_INDEX(idx, inputsInfo.size());
    /** Iterating over all input blobs **/
    auto item = inputsInfo.begin();
    std::advance(item, idx);
    Layout inputLayout = getLayoutByEnum(layout);
    item->second->setLayout(inputLayout);
}

void CIEContext::setOutputLayout(unsigned int idx, IELayoutType layout)
{
    CHECK_INDEX(idx, outputsInfo.size());
    /** Iterating over all output blobs **/
    auto item = inputsInfo.begin();
    std::advance(item, idx);
    Layout outputLayout = getLayoutByEnum(layout);
    item->second->setLayout(outputLayout);
}

void CIEContext::getModelInputImageSize(IEImageSize * imageSize)
{
    if (!bModelCreated) {
        std::cout << "Please create the model firstly!" << endl;
        return;
    }

    imageSize->height = inputImageSize.height;
    imageSize->width = inputImageSize.width;
    return;
}

void CIEContext::getModelInputInfo(unsigned int idx, IETensorInfo * info)
{
    CHECK_INDEX(idx, inputsInfo.size());
    /** Iterating over all input blobs **/
    auto item = inputsInfo.begin();
    std::advance(item, idx);

    info->rank = item->second->getTensorDesc().getDims().size();
    for (int i = 0; i < info->rank; i++) {
        info->dim[i] = item->second->getDims()[i];
    }
    info->precision = getEnumByPrecision(item->second->getPrecision());
    info->layout = getEnumByLayout(item->second->getLayout());
}

void CIEContext::getModelInputInfo(IEInputOutputInfo * info)
{
    if (!bModelLoaded) {
        std::cout << "Please load the model firstly!" << endl;
        return;
    }

    int id = 0;

    /* allocate the input info pointer for the tensorinfo*/
    for (auto & item : inputsInfo) {
        info->tensor[id].rank = item.second->getTensorDesc().getDims().size();
        for (int i = 0; i < info->tensor[id].rank; i++) {
            info->tensor[id].dim[i] = item.second->getDims()[i];
        }
        info->tensor[id].precision = getEnumByPrecision(item.second->getPrecision());
        info->tensor[id].layout = getEnumByLayout(item.second->getLayout());
        id++;
    }
    info->batchSize = getBatchSize();
    info->numbers = inputsInfo.size();
}

void CIEContext::setModelInputInfo(unsigned int idx, IETensorInfo * info)
{
    CHECK_INDEX(idx, inputsInfo.size());
    /** Iterating over all input blobs **/
    auto item = inputsInfo.begin();
    std::advance(item, idx);

    Precision precision = getPrecisionByEnum(info->precision);
    item->second->setPrecision(precision);
    Layout layout = getLayoutByEnum(info->layout);
    item->second->setLayout(layout);
}

void CIEContext::setModelInputInfo(IEInputOutputInfo * info)
{
    int id = 0;

    if (!bModelLoaded) {
        std::cout << "Please load the model firstly!" << endl;
        return;
    }

    if (nullptr == info->tensor) {
        std::cout << "Please get the input info firstly!" << endl;
        return;
    }

    if (info->numbers != inputsInfo.size()) {
        std::cout << "Input size is not matched with model!" << endl;
        return;
    }

    for (auto & item : inputsInfo) {
        Precision precision = getPrecisionByEnum(info->tensor[id].precision);
        item.second->setPrecision(precision);
        Layout layout = getLayoutByEnum(info->tensor[id].layout);
        item.second->setLayout(layout);

        if (info->tensor[id].dataType == IE_DATA_TYPE_IMG) {
            inputImageSize.height = inputsInfo[item.first]->getDims()[1];
            inputImageSize.width = inputsInfo[item.first]->getDims()[0];
        }

        id++;
    }
}

void CIEContext::getModelOutputInfo(unsigned int idx, IETensorInfo * info)
{
    CHECK_INDEX(idx, outputsInfo.size());
    /** Iterating over all input blobs **/
    auto item = outputsInfo.begin();
    std::advance(item, idx);

    info->rank = item->second->getTensorDesc().getDims().size();
    for (int i = 0; i < info->rank; i++) {
        info->dim[i] = item->second->getDims()[i];
    }
    info->precision = getEnumByPrecision(item->second->getPrecision());
    info->layout = getEnumByLayout(item->second->getLayout());
}

void CIEContext::getModelOutputInfo(IEInputOutputInfo * info)
{
    if (!bModelLoaded) {
        std::cout << "Please load the model firstly!" << endl;
        return;
    }

    int id = 0;

    /* allocate the input info pointer for the tensorinfo*/
    for (auto & item : outputsInfo) {
        info->tensor[id].rank = item.second->getTensorDesc().getDims().size();
        for (int i = 0; i < info->tensor[id].rank; i++) {
            info->tensor[id].dim[i] = item.second->getDims()[i];
        }
        info->tensor[id].precision = getEnumByPrecision(item.second->getPrecision());
        info->tensor[id].layout = getEnumByLayout(item.second->getLayout());
        id++;
    }
    info->batchSize = 0;
    info->numbers = outputsInfo.size();
}

void CIEContext::setModelOutputInfo(unsigned int idx, IETensorInfo * info)
{
    CHECK_INDEX(idx, outputsInfo.size());
    /** Iterating over all input blobs **/
    auto item = outputsInfo.begin();
    std::advance(item, idx);

    Precision precision = getPrecisionByEnum(info->precision);
    item->second->setPrecision(precision);
    Layout layout = getLayoutByEnum(info->layout);
    item->second->setLayout(layout);
}

void CIEContext::setModelOutputInfo(IEInputOutputInfo * info)
{
    int id = 0;

    if (!bModelLoaded) {
        std::cout << "Please load the model firstly!" << endl;
        return;
    }

    if (nullptr == info->tensor) {
        std::cout << "Please get the output info firstly!" << endl;
        return;
    }

    if (info->numbers != outputsInfo.size()) {
        std::cout << "Output size is not matched with model!" << endl;
        return;
    }

    for (auto & item : outputsInfo) {
        Precision precision = getPrecisionByEnum(info->tensor[id].precision);
        item.second->setPrecision(precision);
        Layout layout = getLayoutByEnum(info->tensor[id].layout);
        item.second->setLayout(layout);
    }
}

void CIEContext::addInput(unsigned int idx, IEData * data)
{
    unsigned int id = 0;
    std::string itemName;

    CHECK_INDEX(idx, inputsInfo.size());
    if (nullptr == data) {
        std::cout << "Input data is null pointer!" << std::endl;
        return;
    }

    /** Iterating over all input blobs **/
    auto item = inputsInfo.begin();
    std::advance(item, idx);
    itemName = item->first;

    if (itemName.empty()) {
        std::cout << "item name is empty!" << std::endl;
        return;
    }

    if (data->batchIdx > getBatchSize()) {
        std::cout << "Too many input, it is bigger than batch size!" << std::endl;
        return;
    }

    Blob::Ptr blob = inferRequest.GetBlob(itemName);
    if (data->tensor.precision == IE_FP32) {
        if(data->tensor.dataType == IE_DATA_TYPE_IMG)
            imageU8ToBlob<PrecisionTrait<Precision::FP32>::value_type>(data, blob, data->batchIdx);
        else
            nonImageToBlob<PrecisionTrait<Precision::FP32>::value_type>(data, blob, data->batchIdx);
    } else { /* IE_U8 */
        if(data->tensor.dataType == IE_DATA_TYPE_IMG)
            imageU8ToBlob<uint8_t>(data, blob, data->batchIdx);
        else
            nonImageToBlob<uint8_t>(data, blob, data->batchIdx);
    }
}

void * CIEContext::getOutput(unsigned int idx, unsigned int * size)
{
    unsigned int id = 0;
    std::string itemName;

    CHECK_INDEX(idx, outputsInfo.size());

    /** Iterating over all input blobs **/
    auto item = outputsInfo.begin();
    std::advance(item, idx);
    itemName = item->first;

    if (itemName.empty()) {
        std::cout << "item name is empty!" << std::endl;
        return nullptr;
    }

    const Blob::Ptr blob = inferRequest.GetBlob(itemName);
    float* outputResult = static_cast<PrecisionTrait<Precision::FP32>::value_type*>(blob->buffer());

    *size = blob->byteSize();
    return (reinterpret_cast<void *>(outputResult));
}

unsigned int CIEContext::getOutput(unsigned int idx, IEData * data)
{
    unsigned int id = 0;
    unsigned int size = 0;
    std::string itemName;

    CHECK_INDEX(idx, outputsInfo.size());
    if (nullptr == data) {
        std::cout << "Output data is null pointer!" << std::endl;
        return size;
    }

    /** Iterating over all input blobs **/
    auto item = outputsInfo.begin();
    std::advance(item, idx);
    itemName = item->first;

    if (itemName.empty()) {
        std::cout << "item name is empty!" << std::endl;
        return size;
    }

    const Blob::Ptr blob = inferRequest.GetBlob(itemName);
    float* outputResult = static_cast<PrecisionTrait<Precision::FP32>::value_type*>(blob->buffer());
    data->header.bufSize = blob->byteSize();
    data->header.bufId = (uintptr_t)(outputResult);

    return data->header.bufSize;
}

InferenceEngine::TargetDevice CIEContext::getDeviceFromString(const std::string &deviceName) {
    return InferenceEngine::TargetDeviceInfo::fromStr(deviceName);
}

InferenceEngine::TargetDevice CIEContext::getDeviceFromId(IETargetDeviceType device) {
    switch (device) {
    case IE_Default:
        return InferenceEngine::TargetDevice::eDefault;
    case IE_Balanced:
        return InferenceEngine::TargetDevice::eBalanced;
    case IE_CPU:
        return InferenceEngine::TargetDevice::eCPU;
    case IE_GPU:
        return InferenceEngine::TargetDevice::eGPU;
    case IE_FPGA:
        return InferenceEngine::TargetDevice::eFPGA;
    case IE_MYRIAD:
        return InferenceEngine::TargetDevice::eMYRIAD;
    case IE_HETERO:
        return InferenceEngine::TargetDevice::eHETERO;
    default:
        return InferenceEngine::TargetDevice::eCPU;
    }
}

InferenceEngine::Layout CIEContext::estimateLayout(const int chNum)
{
    if (chNum == 4)
        return InferenceEngine::Layout::NCHW;
    else if (chNum == 2)
        return InferenceEngine::Layout::NC;
    else if (chNum == 3)
        return InferenceEngine::Layout::CHW;
    else
        return InferenceEngine::Layout::ANY;
}

InferenceEngine::Layout CIEContext::getLayoutByEnum(IELayoutType layout)
{
    switch (layout) {
    case IE_NCHW:
        return InferenceEngine::Layout::NCHW;
    case IE_NHWC:
        return InferenceEngine::Layout::NHWC;
    case IE_OIHW:
        return InferenceEngine::Layout::OIHW;
    case IE_C:
        return InferenceEngine::Layout::C;
    case IE_CHW:
        return InferenceEngine::Layout::CHW;
    case IE_HW:
        return InferenceEngine::Layout::HW;
    case IE_NC:
        return InferenceEngine::Layout::NC;
    case IE_CN:
        return InferenceEngine::Layout::CN;
    case IE_BLOCKED:
        return InferenceEngine::Layout::BLOCKED;
    case IE_ANY:
        return InferenceEngine::Layout::ANY;
    default:
        return InferenceEngine::Layout::ANY;
    }

}

IELayoutType CIEContext::getEnumByLayout(InferenceEngine::Layout layout)
{
    switch (layout) {
    case InferenceEngine::Layout::NCHW:
        return IE_NCHW;
    case InferenceEngine::Layout::NHWC:
        return IE_NHWC;
    case InferenceEngine::Layout::OIHW:
        return IE_OIHW;
    case InferenceEngine::Layout::C:
        return IE_C;
    case InferenceEngine::Layout::CHW:
        return IE_CHW;
    case InferenceEngine::Layout::HW:
        return IE_HW;
    case InferenceEngine::Layout::NC:
        return IE_NC;
    case InferenceEngine::Layout::CN:
        return IE_CN;
    case InferenceEngine::Layout::BLOCKED:
        return IE_BLOCKED;
    case InferenceEngine::Layout::ANY:
        return IE_ANY;
    default:
        return IE_ANY;
    }
}

InferenceEngine::Precision CIEContext::getPrecisionByEnum(IEPrecisionType precision)
{
    switch (precision) {
    case IE_MIXED:
        return InferenceEngine::Precision::MIXED;
    case IE_FP32:
        return InferenceEngine::Precision::FP32;
    case IE_FP16:
        return InferenceEngine::Precision::FP16;
    case IE_Q78:
        return InferenceEngine::Precision::Q78;
    case IE_I16:
        return InferenceEngine::Precision::I16;
    case IE_U8:
        return InferenceEngine::Precision::U8;
    case IE_I8:
        return InferenceEngine::Precision::I8;
    case IE_U16:
        return InferenceEngine::Precision::U16;
    case IE_I32:
        return InferenceEngine::Precision::I32;
    case IE_CUSTOM:
        return InferenceEngine::Precision::CUSTOM;
    case IE_UNSPECIFIED:
        return InferenceEngine::Precision::UNSPECIFIED;
    default:
        return InferenceEngine::Precision::UNSPECIFIED;
    }
}

IEPrecisionType CIEContext::getEnumByPrecision(InferenceEngine::Precision precision)
{
    switch (precision) {
    case InferenceEngine::Precision::MIXED:
        return IE_MIXED;
    case InferenceEngine::Precision::FP32:
        return IE_FP32;
    case InferenceEngine::Precision::FP16:
        return IE_FP16;
    case InferenceEngine::Precision::Q78:
        return IE_Q78;
    case InferenceEngine::Precision::I16:
        return IE_I16;
    case InferenceEngine::Precision::U8:
        return IE_U8;
    case InferenceEngine::Precision::I8:
        return IE_I8;
    case InferenceEngine::Precision::U16:
        return IE_U16;
    case InferenceEngine::Precision::I32:
        return IE_I32;
    case InferenceEngine::Precision::CUSTOM:
        return IE_CUSTOM;
    case InferenceEngine::Precision::UNSPECIFIED:
        return IE_UNSPECIFIED;
    default:
        return IE_UNSPECIFIED;
    }
}

std::string CIEContext::GetFileNameNoExt(const std::string &filePath) {
    auto pos = filePath.rfind('.');
    if (pos == std::string::npos) return filePath;
    return filePath.substr(0, pos);
}

template <typename T> void CIEContext::imageU8ToBlob(const IEData * data, Blob::Ptr& blob, int batchIndex)
{
    SizeVector blobSize = blob.get()->dims();
    const size_t width = blobSize[0];
    const size_t height = blobSize[1];
    const size_t channels = blobSize[2];
    const size_t inputImageSize = width * height;
    unsigned char * buffer = (unsigned char *)data->header.bufId;
    T* blob_data = blob->buffer().as<T*>();
    const float mean_val = 127.5f;
    const float std_val = 0.0078125f;

    if (width != data->tensor.dim[0] || height!= data->tensor.dim[1]) {
        std::cout << "Input Image size is not matched with model!" << endl;
        return;
    }

    int batchOffset = batchIndex * height* width * channels;

    if (IE_IMAGE_BGR_PLANAR == data->imageFormat) {
        // B G R planar input image
        /** Filling input dim with images. First b channel, then g and r channels **/
        size_t imageStrideSize = data->tensor.dimStride[0] * data->tensor.dimStride[1];

        if (data->tensor.dim[0] == data->tensor.dimStride[0] &&
            data->tensor.dim[1] == data->tensor.dimStride[1]) {
            std::memcpy(blob_data + batchOffset, buffer, inputImageSize * channels);
        }
        else if (data->tensor.dim[0] == data->tensor.dimStride[0]) {
            for (size_t ch = 0; ch < channels; ++ch) {
                std::memcpy(blob_data + batchOffset + ch * inputImageSize, buffer + ch * imageStrideSize, inputImageSize);
            }
        }
        else {
            /** Iterate over all pixel in image (b,g,r) **/
            for (size_t ch = 0; ch < channels; ++ch) {
                for (size_t h = 0; h < height; h++) {
                    std::memcpy(blob_data + batchOffset + ch * inputImageSize + h * width, buffer + ch * imageStrideSize + h * data->tensor.dimStride[0], width);
                }
            }
        }
    }
    else if (IE_IMAGE_BGR_PACKED == data->imageFormat) {
        // B G R packed input image
        size_t imageStrideSize = data->tensor.dim[2] * data->tensor.dimStride[0];

        if (data->tensor.precision == IE_FP32) {
            /** Iterate over all pixel in image (b,g,r) **/
            for (size_t h = 0; h < height; h++)
                for (size_t w = 0; w < width; w++)
                    /** Iterate over all channels **/
                    for (size_t ch = 0; ch < channels; ++ch)
                        blob_data[batchOffset + ch * inputImageSize + h * width + w] = float((buffer[h * imageStrideSize + w * data->tensor.dim[2] + ch] - mean_val) * std_val);
        } else {
            /** Iterate over all pixel in image (b,g,r) **/
            for (size_t h = 0; h < height; h++)
                for (size_t w = 0; w < width; w++)
                    /** Iterate over all channels **/
                    for (size_t ch = 0; ch < channels; ++ch)
                        blob_data[batchOffset + ch * inputImageSize + h * width + w] = buffer[h * imageStrideSize + w * data->tensor.dim[2] + ch];
        }
    }
    else if (IE_IMAGE_RGB_PLANAR == data->imageFormat) {
        // R G B planar input image, switch the R and B plane. TBD

    }
    else if (IE_IMAGE_RGB_PACKED == data->imageFormat) {
        // R G B packed input image, wwitch the R and B packed value. TBD

    }
    else if (IE_IMAGE_GRAY_PLANAR == data->imageFormat) {

    }
}

template <typename T> void CIEContext::nonImageToBlob(const IEData * data, Blob::Ptr& blob, int batchIndex)
{
    SizeVector blobSize = blob.get()->dims();
    T * buffer = (T *)data->header.bufId;
    T* blob_data = blob->buffer().as<T*>();

    unsigned int size = 1;

    for (int i = 0; i < data->tensor.rank; i++)
        size = data->tensor.dim[i];

    int batchOffset = batchIndex * size;

    memcpy(blob_data + batchOffset, buffer, size);
}

void CIEContext::printPerformanceCounts(const std::map<std::string, InferenceEngine::InferenceEngineProfileInfo>& performanceMap, std::ostream &stream, bool bshowHeader)
{
    long long totalTime = 0;
    // Print performance counts
    if (bshowHeader) {
        stream << std::endl << "performance counts:" << std::endl << std::endl;
    }
    for (const auto & it : performanceMap) {
        std::string toPrint(it.first);
        const int maxLayerName = 30;

        if (it.first.length() >= maxLayerName) {
            toPrint = it.first.substr(0, maxLayerName - 4);
            toPrint += "...";
        }


        stream << std::setw(maxLayerName) << std::left << toPrint;
        switch (it.second.status) {
        case InferenceEngine::InferenceEngineProfileInfo::EXECUTED:
            stream << std::setw(15) << std::left << "EXECUTED";
            break;
        case InferenceEngine::InferenceEngineProfileInfo::NOT_RUN:
            stream << std::setw(15) << std::left << "NOT_RUN";
            break;
        case InferenceEngine::InferenceEngineProfileInfo::OPTIMIZED_OUT:
            stream << std::setw(15) << std::left << "OPTIMIZED_OUT";
            break;
        }
        stream << std::setw(30) << std::left << "layerType: " + std::string(it.second.layer_type) + " ";
        stream << std::setw(20) << std::left << "realTime: " + std::to_string(it.second.realTime_uSec);
        stream << std::setw(20) << std::left << " cpu: " + std::to_string(it.second.cpu_uSec);
        stream << " execType: " << it.second.exec_type << std::endl;
        if (it.second.realTime_uSec > 0) {
            totalTime += it.second.realTime_uSec;
        }
    }
    stream << std::setw(20) << std::left << "Total time: " + std::to_string(totalTime) << " microseconds" << std::endl;
}

void CIEContext::printLog(unsigned int flag)
{
    std::map<std::string, InferenceEngine::InferenceEngineProfileInfo> perfomanceMap;

    if (flag == IE_LOG_LEVEL_NONE)
        return;

    if (flag&IE_LOG_LEVEL_ENGINE) {
        enginePtr->GetPerformanceCounts(perfomanceMap, nullptr);
        printPerformanceCounts(perfomanceMap, std::cout, true);
    }

    if (flag&IE_LOG_LEVEL_LAYER) {
        perfomanceMap = inferRequest.GetPerformanceCounts();
        printPerformanceCounts(perfomanceMap, std::cout, true);
    }
}

}  // namespace InferenceEngine
