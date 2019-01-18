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
 * @brief This is a header file with common C API of the inference engine definitions.
 * @file ie_api_c.h
 */

#ifndef _IE_API_C_H_
#define _IE_API_C_H_

#include <ie_common_c.h>

#if defined(_WIN32)
#ifdef IMPLEMENT_IE_EXTERN_API
#define IE_EXTERN(type) __declspec(dllexport) type __cdecl
#else
#define IE_EXTERN(type) __declspec(dllimport) type __cdecl
#endif
#else
#define IE_EXTERN(TYPE) extern "C" TYPE
#endif

#ifdef __cplusplus
extern "C" {
#endif

/*
 * @brief The API IEAllocateContextWithConfig, which allocate the Inference engine context
 * @param the IEConfig as input
 * @return the void pointer to the IE context
 */
IE_EXTERN(void *) IEAllocateContextWithConfig(IEConfig * config);

/*
 * @brief The API IEAllocateContext, which allocate the Inference engine context defaultly
 * @return the void pointer to the IE context
 */
IE_EXTERN(void *) IEAllocateContext(void);

/*
 * @brief The API IEFreeContext,which release the inferencce engine context
 * @param the pointer to inference engine context
 * @param the IEConfig pointer
 */
IE_EXTERN(void) IEFreeContext(void * contextPtr, IEConfig * config);

/*
* @brief The API IEAllocateInputOutputInfo, which allocate the input/ouput info for the IEConfig
* @param the pointer to inference engine context
* @param the IEConfig pointer
*/
IE_EXTERN(void) IEAllocateInputOutputInfo(void * contextPtr, IEConfig * config);

/*
* @brief The API IEFreeInputOutputInfo, which free the input/ouput info for the IEConfig
* @param the pointer to inference engine context
* @param the IEConfig pointer
*/
IE_EXTERN(void) IEFreeInputOutputInfo(void * contextPtr, IEConfig * config);

/*
* @brief The API IELoadModel, which load the model from IR format
* @param the pointer to inference engine context
* @param the IEConfig as input with IR file name(absolute path)
*/
IE_EXTERN(void) IELoadModel(void * contextPtr, IEConfig * config);

/*
 * @brief The API IECreateModel, which create the model/network and deploy it on the target device.
 * this need to fill in the input info and output info before this API.
 * @param the pointer to inference engine context
 * @param the IEConfig as input with IR file name(absolute path)
 */
IE_EXTERN(void) IECreateModel(void * contextPtr, IEConfig * config);

/*
 * @brief The API IESizeOfContext, which get the size of inference engine context
 * @return the size of context
 */
IE_EXTERN(int32_t) IESizeOfContext(void);

/*
 * @brief The API IEGetInputImageSize, which get the size of Input Image
 * @param the pointer to inference engine context
 * @param the Image size structure
 */
IE_EXTERN(void) IEGetInputImageSize(void * contextPtr, IEImageSize * size);

/*
 * @brief The API IEGetInputInfo, which get the info of model Input
 * @param the pointer to inference engine context
 * @return the input info of the model
 */
IE_EXTERN(void) IEGetInputInfo(void * contextPtr, IEInputOutputInfo * info);

/*
 * @brief The API IESetInputInfo, which set the info of model Input
 * @param the pointer to inference engine context
 * @return the input info of the model
 */
IE_EXTERN(void) IESetInputInfo(void * contextPtr, IEInputOutputInfo * info);

/*
 * @brief The API IEGetOutputInfo, which get the info of model output
 * @param the pointer to inference engine context
 * @return the output info of the model
 */
IE_EXTERN(void) IEGetOutputInfo(void * contextPtr, IEInputOutputInfo * info);

/*
* @brief The API IESetOutputInfo, which set the info of model output
* @param the pointer to inference engine context
* @return the output info of the model
*/
IE_EXTERN(void) IESetOutputInfo(void * contextPtr, IEInputOutputInfo * info);

/*
 * @brief The API IEForward, which execute the model in the sync/async mode. Call the IESetInput firstly.
 * @param the pointer to inference engine context
 * @param the sync or async mode. 1: async mode; 0: sync mode. default is 0
 */
IE_EXTERN(void) IEForward(void * contextPtr, IEInferMode mode);

/*
 * @brief The API IESetInput, which feed the input image to the model
 * @param the pointer to inference engine context
 * @param the index of input, idx can be get from the IEGetModelInputInfo()
 * @param the image to be processed
 */
IE_EXTERN(void) IESetInput(void * contextPtr, uint32_t idx, IEData * data);

/*
 * @brief The API IEGetResul, which get the result pointer after the execution
 * @param  the pointer to inference engine context
 * @param  the index of output, idx can be get from the IEGetModelOutputInfo()
 * @param  the size of result
 */
IE_EXTERN(void) * IEGetResult(void * contextPtr, uint32_t idx, uint32_t * size);

/*
 * @brief The API IEPrintLog, which print the log info after the execution
 * @param the pointer to inference engine context
 * @param the flag of log level
 */
IE_EXTERN(void) IEPrintLog(void * contextPtr, uint32_t flag);

/*
 * @brief The API IESetBatchSize, which set batch size
 * @param the pointer to inference engine context
 * @param the batch size value to be set
 */
IE_EXTERN(void) IESetBatchSize(void *contextPtr, int32_t size);

#ifdef __cplusplus
}
#endif
#endif
