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
 * @brief This is a header file with common C inference engine definitions.
 * @file ie_common_c.h
 */

#ifndef _IE_COMMON_C_H_
#define _IE_COMMON_C_H_

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @enum IETargetDeviceType
 * @brief This describes known device/plugin types
 */
typedef enum tagIETargetDeviceType {
    IE_Default = 0,
    IE_Balanced = 1,
    IE_CPU = 2,
    IE_GPU = 3,
    IE_FPGA = 4,
    IE_MYRIAD = 5,
    IE_HETERO = 8
}IETargetDeviceType;

/**
 * @enum IEPrecisionType
 * @brief This define data or operation Precision types
 */
typedef enum tagIEPrecisinType {
    IE_UNSPECIFIED = 255, /**< Unspecified value. Used by default */
    IE_MIXED = 0,  /**< Mixed value. Can be received from network. No applicable for tensors */
    IE_FP32 = 10,  /**< 32bit floating point value */
    IE_FP16 = 11,  /**< 16bit floating point value */
    IE_Q78 = 20,   /**< 16bit specific signed fixed point precision */
    IE_I16 = 30,   /**< 16bit signed integer value */
    IE_U8 = 40,    /**< 8bit unsigned integer value */
    IE_I8 = 50,    /**< 8bit signed integer value */
    IE_U16 = 60,   /**< 16bit unsigned integer value */
    IE_I32 = 70,   /**< 32bit signed integer value */
    IE_CUSTOM = 80 /**< custom precision has it's own name and size of elements */
}IEPrecisionType;

/**
 * @enum IELayoutType
 * @brief This define the input data Layouts that the inference engine supports
 */
typedef enum tagIELayoutType {
    IE_ANY = 0,/* "any" layout */
    /* I/O data layouts */
    IE_NCHW = 1,
    IE_NHWC = 2,
    /* weight layouts */
    IE_OIHW = 64,
    /* bias layouts */
    IE_C = 96,
    /* Single image layout (for mean image) */
    IE_CHW = 128,
    /* 2D */
    IE_HW = 192,
    IE_NC = 193,
    IE_CN = 194,
    IE_BLOCKED = 200,
}IELayoutType;

/**
 * @enum IEMemoryType
 * @brief It define memory type of input data
 */
typedef enum tagIEMemoryType {
    IE_DEVICE_DEFAULT = 0,
    IE_DEVICE_HOST = 1,
    IE_DEVICE_GPU = 2,
    IE_DEVICE_MYRIAD = 3,
    IE_DEVICE_SHARED = 4,
}IEMemoryType;

/**
 * @enum IEImageFormatType
 * @brief This define the Image Channel order type,now the BGR is used in the most model.
 */
typedef enum tagIEImageFormatType {
    IE_IMAGE_FORMAT_UNKNOWN = -1,
    IE_IMAGE_BGR_PACKED,
    IE_IMAGE_BGR_PLANAR,
    IE_IMAGE_RGB_PACKED,
    IE_IMAGE_RGB_PLANAR,
    IE_IMAGE_GRAY_PLANAR,
    IE_IMAGE_GENERIC_1D,
    IE_IMAGE_GENERIC_2D,
}IEImageFormatType;

/**
 * @enum IEInferMode
 * @brief IE forward mode: sync/async
 */
typedef enum tagIEInferMode {
    IE_INFER_MODE_SYNC = 0,
    IE_INFER_MODE_ASYNC = 1,
}IEInferMode;

/**
 * @enum IEDataType
 * @brief IE data mode: image/non-image
 */
typedef enum tagIEDataType {
    IE_DATA_TYPE_NON_IMG = 0,
    IE_DATA_TYPE_IMG = 1,
}IEDataType;

/**
 * @enum IELogLevel
 * @brief This describe the IE log level
 */
typedef enum tagIELogLevel {
    IE_LOG_LEVEL_NONE = 0x0,
    IE_LOG_LEVEL_ENGINE = 0x1,
    IE_LOG_LEVEL_LAYER = 0x2,
}IELogLevel;

/**
* @struct IEBufferExt
* @brief This describe common header for the buffer
*/
typedef struct tagIEExtBuf {
    uint32_t bufSize;
    uintptr_t bufId;
}IEExtBuf;

/**
 * @struct IEImageSize
 * @brief This define the image Size, the BGR is used in the most model.
 */
typedef struct tagIEImageSize {
    uint32_t width;
    uint32_t height;
}IEImageSize;

/**
 * @brief this define the number of model input/output tensor info.
 */
#define IE_TENSOR_MAX_RANK 12

typedef struct tagTensorInfo {
    uint32_t rank;
    uint32_t dim[IE_TENSOR_MAX_RANK];       /* [0]: width, [1]:height, [2]:channels, etc. */
    uint32_t dimStride[IE_TENSOR_MAX_RANK]; /* [0]: width, [1]:height, [2]:channels, etc. */
    IEPrecisionType precision;              /* IE_FP32:IE_FP16:IE_U8 etc. */
    IELayoutType layout;
    IEDataType dataType;
}IETensorInfo;

/**
 * @struct IEInputOutputInfo
 * @brief this define the structure of model input/output info.
 */
typedef struct tagIEInputOutputInfo {
    IEExtBuf header;        /* bufId is the pointer to the IETensorInfo */
    IETensorInfo * tensor; /* model input/output info pointer */
    uint32_t batchSize;
    uint32_t numbers;          /* model input/output number */
}IEInputOutputInfo;

/**
 * @struct IEData
 * @brief This describe inference engine Input Data: Image(BGR format) or Non-Image type
 */
typedef struct tagIEData {
    IEExtBuf header;           /* bufId is the pointer to the input data buffer(image or non image) */
    IETensorInfo tensor;
    uint32_t batchIdx;
    IEMemoryType memType;
    IEImageFormatType imageFormat;
}IEData;

/**
 * @struct IEConfig
 * @brief This describe the Inference Engine Context Configuration
 */
typedef struct tagIEConfig {
    IETargetDeviceType targetId;
    IEInputOutputInfo inputInfos;
    IEInputOutputInfo outputInfos;

    char * pluginPath;
    char * cpuExtPath;      /* extension file name for CPU */
    char * cldnnExtPath;    /* extension file name for GPU */
    char * modelFileName;   /* Bin/xml file name */
    uint32_t perfCounter;   /* performance measurement flag */
    uint32_t inferReqNum;   /* it work with async mode and value is 1 in default. */
}IEConfig;

#ifdef __cplusplus
}
#endif
#endif
