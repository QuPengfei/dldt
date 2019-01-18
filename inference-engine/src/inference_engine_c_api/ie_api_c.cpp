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

#include "ie_api_c.h"
#include "ie_context.h"


#ifdef __cplusplus
extern "C" {
#endif

using namespace std;
using namespace InferenceEngine;

int32_t IESizeOfContext()
{
    return sizeof(InferenceEngine::CIEContext);
}

void * IEAllocateContext()
{
    InferenceEngine::CIEContext * context = new InferenceEngine::CIEContext();
    return (reinterpret_cast<void *>(context));
}

void * IEAllocateContextWithConfig(IEConfig * config)
{
    InferenceEngine::CIEContext * context = new InferenceEngine::CIEContext(config);
    IEAllocateInputOutputInfo(context, config);
    return (reinterpret_cast<void *>(context));
}

void IEFreeContext(void * contextPtr, IEConfig * config)
{
    IEFreeInputOutputInfo(contextPtr, config);
    delete(reinterpret_cast<InferenceEngine::CIEContext *>(contextPtr));
}

void IEAllocateInputOutputInfo(void * contextPtr, IEConfig * config)
{
    InferenceEngine::CIEContext * context = reinterpret_cast<InferenceEngine::CIEContext *>(contextPtr);
    uint32_t size = 0;
    IEInputOutputInfo * info = nullptr;

    size = context->getInputSize();
    info = &(config->inputInfos);

    if (nullptr == info->tensor) {
        info->tensor = (IETensorInfo *)malloc(size * sizeof(IETensorInfo));
        info->header.bufSize = size * sizeof(IETensorInfo);
        info->header.bufId = (uintptr_t)(info->tensor);
        memset((void *)info->tensor, 0, info->header.bufSize);
    }
    else if (size != info->numbers) {
        free(info->tensor);
        info->header.bufSize = 0;
        info->header.bufId = 0;
        info->tensor = (IETensorInfo *)malloc(size * sizeof(IETensorInfo));
        info->header.bufSize = size * sizeof(IETensorInfo);
        info->header.bufId = (uintptr_t)(info->tensor);
        memset((void *)info->tensor, 0, info->header.bufSize);
    }

    size = context->getOutputSize();
    info = &(config->outputInfos);

    if (nullptr == info->tensor) {
        info->tensor = (IETensorInfo *)malloc(size * sizeof(IETensorInfo));
        info->header.bufSize = size * sizeof(IETensorInfo);
        info->header.bufId = (uintptr_t)(info->tensor);
        memset((void *)info->tensor, 0, info->header.bufSize);
    }
    else if (size != info->numbers) {
        free(info->tensor);
        info->header.bufSize = 0;
        info->header.bufId = 0;
        info->tensor = (IETensorInfo *)malloc(size * sizeof(IETensorInfo));
        info->header.bufSize = size * sizeof(IETensorInfo);
        info->header.bufId = (uintptr_t)(info->tensor);
        memset((void *)info->tensor, 0, info->header.bufSize);
    }
}

void IEFreeInputOutputInfo(void * contextPtr, IEConfig * config)
{
    IEInputOutputInfo * info = nullptr;

    info = &(config->inputInfos);
    if (nullptr != info->tensor) {
        free(info->tensor);
        info->header.bufSize = 0;
        info->header.bufId = 0;
    }
    info = &(config->outputInfos);
    if (nullptr != info->tensor) {
        free(info->tensor);
        info->header.bufSize = 0;
        info->header.bufId = 0;
    }
}

void IELoadModel(void * contextPtr, IEConfig * config)
{
    InferenceEngine::CIEContext * context = reinterpret_cast<InferenceEngine::CIEContext *>(contextPtr);
    context->loadModel(config);
}

void IECreateModel(void * contextPtr, IEConfig * config)
{
    InferenceEngine::CIEContext * context = reinterpret_cast<InferenceEngine::CIEContext *>(contextPtr);
    context->createModel(config);
}

void IEGetInputImageSize(void * contextPtr, IEImageSize * size)
{
    InferenceEngine::CIEContext * context = reinterpret_cast<InferenceEngine::CIEContext *>(contextPtr);
    context->getModelInputImageSize(size);
}

void IEGetInputInfo(void * contextPtr, IEInputOutputInfo * info)
{
    InferenceEngine::CIEContext * context = reinterpret_cast<InferenceEngine::CIEContext *>(contextPtr);
    context->getModelInputInfo(info);
}

void IESetInputInfo(void * contextPtr, IEInputOutputInfo * info)
{
    InferenceEngine::CIEContext * context = reinterpret_cast<InferenceEngine::CIEContext *>(contextPtr);
    context->setModelInputInfo(info);
}

void IEGetOutputInfo(void * contextPtr, IEInputOutputInfo * info)
{
    InferenceEngine::CIEContext * context = reinterpret_cast<InferenceEngine::CIEContext *>(contextPtr);
    context->getModelOutputInfo(info);
}

void IESetOutputInfo(void * contextPtr, IEInputOutputInfo * info)
{
    InferenceEngine::CIEContext * context = reinterpret_cast<InferenceEngine::CIEContext *>(contextPtr);
    context->setModelOutputInfo(info);
}

void IEForward(void * contextPtr, IEInferMode aSyncMode)
{
    InferenceEngine::CIEContext * context = reinterpret_cast<InferenceEngine::CIEContext *>(contextPtr);

    if (IE_INFER_MODE_SYNC == aSyncMode) {
        context->forwardSync();
    }
    else {
        context->forwardAsync();
    }
}

void IESetInput(void * contextPtr, uint32_t idx, IEData * data)
{
    InferenceEngine::CIEContext * context = reinterpret_cast<InferenceEngine::CIEContext *>(contextPtr);
    context->addInput(idx, data);
}

void * IEGetResult(void * contextPtr, uint32_t idx, uint32_t * size)
{
    InferenceEngine::CIEContext * context = reinterpret_cast<InferenceEngine::CIEContext *>(contextPtr);
    return context->getOutput(idx, size);
}

void IEPrintLog(void * contextPtr, uint32_t flag)
{
    InferenceEngine::CIEContext * context = reinterpret_cast<InferenceEngine::CIEContext *>(contextPtr);
    return context->printLog(flag);
}

void IESetBatchSize(void *contextPtr, int32_t size)
{
    InferenceEngine::CIEContext * context = reinterpret_cast<InferenceEngine::CIEContext *>(contextPtr);
    return context->setBatchSize(size);
}

#ifdef __cplusplus
}
#endif
