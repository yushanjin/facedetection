/**
* Copyright 2020 Huawei Technologies Co., Ltd
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at

* http://www.apache.org/licenses/LICENSE-2.0

* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.

* File sample_process.cpp
* Description: handle acl resource
*/
#include <iostream>

#include "acl/acl.h"
#include "model_process.h"
#include "utils.h"
#include "face_detect.h"
using namespace std;

namespace {
 const static std::vector<std::string> ssdLabel = { "background", "face"};

const uint32_t kBBoxDataBufId = 1;
const uint32_t kBoxNumDataBufId = 0;

enum BBoxIndex { EMPTY = 0, LABEL,SCORE,TOPLEFTX, TOPLEFTY, BOTTOMRIGHTX, BOTTOMRIGHTY};
}

FaceDetect::FaceDetect(const char* modelPath, uint32_t modelWidth, uint32_t modelHeight)
:deviceId_(0), context_(nullptr), stream_(nullptr), imageDataBuf_(nullptr), modelWidth_(modelWidth), modelHeight_(modelHeight), channel_(nullptr), isInited_(false){
    modelPath_ = modelPath;
    imageDataSize_ = RGBU8_IMAGE_SIZE(modelWidth, modelHeight);
}

FaceDetect::~FaceDetect() {
    DestroyResource();
}

Result FaceDetect::InitResource() {
    // ACL init
    const char *aclConfigPath = "../src/acl.json";
    aclError ret = aclInit(aclConfigPath);
    if (ret != ACL_ERROR_NONE) {
        ERROR_LOG("acl init failed\n");
        return FAILED;
    }
    INFO_LOG("acl init success\n");

    // open device
    ret = aclrtSetDevice(deviceId_);
    if (ret != ACL_ERROR_NONE) {
        ERROR_LOG("acl open device %d failed\n", deviceId_);
        return FAILED;
    }
    INFO_LOG("open device %d success\n", deviceId_);

    // create context (set current)
    ret = aclrtCreateContext(&context_, deviceId_);
    if (ret != ACL_ERROR_NONE) {
        ERROR_LOG("acl create context failed\n");
        return FAILED;
    }
    INFO_LOG("create context success");

    // create stream
    ret = aclrtCreateStream(&stream_);
    if (ret != ACL_ERROR_NONE) {
        ERROR_LOG("acl create stream failed\n");
        return FAILED;
    }
    INFO_LOG("create stream success");

    ret = aclrtGetRunMode(&runMode_);
    if (ret != ACL_ERROR_NONE) {
        ERROR_LOG("acl get run mode failed\n");
        return FAILED;
    }

    return SUCCESS;
}

Result FaceDetect::InitModel(const char* omModelPath) {
    Result ret = model_.LoadModelFromFileWithMem(omModelPath);
    if (ret != SUCCESS) {
        ERROR_LOG("execute LoadModelFromFileWithMem failed\n");
        return FAILED;
    }

    ret = model_.CreateDesc();
    if (ret != SUCCESS) {
        ERROR_LOG("execute CreateDesc failed\n");
        return FAILED;
    }

    ret = model_.CreateOutput();
    if (ret != SUCCESS) {
        ERROR_LOG("execute CreateOutput failed\n");
        return FAILED;
    }

    return SUCCESS;
}

Result FaceDetect::CreateModelInputdDataset()
{
    //Request image data memory for input model
    aclError aclRet = aclrtMalloc(&imageDataBuf_, imageDataSize_, ACL_MEM_MALLOC_HUGE_FIRST);
    if (aclRet != ACL_ERROR_NONE) {
        ERROR_LOG("malloc device data buffer failed, aclRet is %d", aclRet);
        return FAILED;
    }
    //Use the applied memory to create the model and input dataset. After creation, only update the memory data for each frame of inference, instead of creating the input dataset every time
    Result ret = model_.CreateInput(imageDataBuf_, imageDataSize_);
    if (ret != SUCCESS) {
        ERROR_LOG("Create mode input dataset failed");
        return FAILED;
    }

    return SUCCESS;
}

Result FaceDetect::OpenPresenterChannel(string channel_name) {
    OpenChannelParam param;
    param.host_ip = "192.168.1.123";  //IP address of Presenter Server
    param.port = 7008;  //port of present service
    //    param.channel_name = "video";
    param.channel_name = channel_name;
    param.content_type = ContentType::kVideo;  //content type is Video
    INFO_LOG("OpenChannel start");
    PresenterErrorCode errorCode = OpenChannel(channel_, param);
    INFO_LOG("OpenChannel param");
    if (errorCode != PresenterErrorCode::kNone) {
        ERROR_LOG("OpenChannel failed %d", static_cast<int>(errorCode));
        return FAILED;
    }

    return SUCCESS;
}

Result FaceDetect::Init(string channel_name) {
    if (isInited_) {
        INFO_LOG("FaceDetect instance is initied already!\n");
        return SUCCESS;
    }

    Result ret = InitResource();
    if (ret != SUCCESS) {
        ERROR_LOG("Init acl resource failed\n");
        return FAILED;
    }

    ret = InitModel(modelPath_);
    if (ret != SUCCESS) {
        ERROR_LOG("Init model failed\n");
        return FAILED;
    }

    ret = CreateModelInputdDataset();
    if (ret != SUCCESS) {
        ERROR_LOG("Create image info buf failed\n");
        return FAILED;
    }

    //Connect the presenter server
    ret = OpenPresenterChannel(channel_name);
    if (ret != SUCCESS) {
        ERROR_LOG("Open presenter channel failed");
        return FAILED;
    }

    isInited_ = true;
    return SUCCESS;
}

Result FaceDetect::Preprocess(cv::Mat& frame) {
    //Scale the frame image to the desired size of the model
    cv::Mat reiszeMat;
    cv::resize(frame, reiszeMat, cv::Size(modelWidth_, modelHeight_));
    if (reiszeMat.empty()) {
        ERROR_LOG("Resize image failed");
        return FAILED;
    }
//    Copy the data into the cache of the input dataset
    aclrtMemcpyKind policy = (runMode_ == ACL_HOST)?ACL_MEMCPY_HOST_TO_DEVICE:ACL_MEMCPY_DEVICE_TO_DEVICE;
    aclError ret = aclrtMemcpy(imageDataBuf_, imageDataSize_,reiszeMat.ptr<uint8_t>(), imageDataSize_, policy);
    if (ret != ACL_ERROR_NONE) {
        ERROR_LOG("Copy resized image data to device failed.");
        return FAILED;
    }

    return SUCCESS;
}

Result FaceDetect::Inference(aclmdlDataset*& inferenceOutput) {
    //Perform reasoning
    Result ret = model_.Execute();
    if (ret != SUCCESS) {
        ERROR_LOG("Execute model inference failed");
        return FAILED;
    }
    //Get inference output
    inferenceOutput = model_.GetModelOutputData();

    return SUCCESS;
}

Result FaceDetect::Postprocess(cv::Mat& frame, aclmdlDataset* modelOutput) {
    uint32_t dataSize = 0;
    float* detectData = (float *)GetInferenceOutputItem(dataSize, modelOutput, kBBoxDataBufId);
    if (detectData == nullptr)
        return FAILED;
    printf("detectData:%f \n", *detectData);

    uint32_t* boxNum = (uint32_t *)GetInferenceOutputItem(dataSize, modelOutput, kBoxNumDataBufId);
    if (boxNum == nullptr)
        return FAILED;
    printf("boxNum:%d \n", *boxNum);

    uint32_t totalBox = boxNum[0];
    printf("totalBox:%d \n", totalBox);

//    float widthScale = (float)(frame.cols) / modelWidth_;
//    float heightScale = (float)(frame.rows) / modelHeight_;

    vector<DetectionResult> detectResults;

    for (uint32_t i = 0; i < totalBox; i++) {

        INFO_LOG("object: area %f, %f, %f, %f,  %f  %f %f %f  \n",
        detectData[i*8+0],  detectData[i*8+1], detectData[i*8+2],
        detectData[i*8+3],  detectData[i*8+4], detectData[i*8+5], detectData[i*8+6], detectData[i*8+7]);

        DetectionResult oneResult;
        Point point_lt, point_rb;
        uint32_t score = uint32_t(detectData[SCORE + i*8] * 100);
        if (score < 70)
            break ;
        printf("score:%d \n", score);
        point_lt.x = detectData[ TOPLEFTX + i*8] * (float)(frame.cols);
        point_lt.y = detectData[ TOPLEFTY + i*8] *  (float)(frame.rows);
        point_rb.x = detectData[ BOTTOMRIGHTX + i*8] * (float)(frame.cols);
        point_rb.y = detectData[ BOTTOMRIGHTY + i*8] *  (float)(frame.rows);

        uint32_t objIndex = (uint32_t)detectData[LABEL + i*8];
        oneResult.lt = point_lt;
        oneResult.rb = point_rb;
        oneResult.result_text = ssdLabel[objIndex] + ":" + std::to_string(score) + "\%";
        INFO_LOG("%d %d %d %d %d %s\n", objIndex,point_lt.x, point_lt.y, point_rb.x, point_rb.y, oneResult.result_text.c_str());

        detectResults.emplace_back(oneResult);
    }
    //If it is the host side, the data is copied from the device and the memory used by the copy is freed
    if (runMode_ == ACL_HOST) {
        delete[]((uint8_t*)detectData);
        delete[]((uint8_t*)boxNum);
    }

    //Sends inference results and images to presenter Server for display
    SendImage(detectResults, frame);

    return SUCCESS;
}

void* FaceDetect::GetInferenceOutputItem(uint32_t& itemDataSize, aclmdlDataset* inferenceOutput, uint32_t idx)
{
    aclDataBuffer* dataBuffer = aclmdlGetDatasetBuffer(inferenceOutput, idx);
    if (dataBuffer == nullptr) {
        ERROR_LOG("Get the %dth dataset buffer from model "
                  "inference output failed\n", idx);
        return nullptr;
    }

    void* dataBufferDev = aclGetDataBufferAddr(dataBuffer);
    if (dataBufferDev == nullptr) {
        ERROR_LOG("Get the %dth dataset buffer address "
                  "from model inference output failed\n", idx);
        return nullptr;
    }

    size_t bufferSize = aclGetDataBufferSize(dataBuffer);
    if (bufferSize == 0) {
        ERROR_LOG("The %d   th dataset buffer size of "
                  "model inference output is 0\n", idx);
        return nullptr;
    }

    void* data = nullptr;
    if (runMode_ == ACL_HOST) {
        data = Utils::CopyDataDeviceToLocal(dataBufferDev, bufferSize);
        if (data == nullptr) {
            ERROR_LOG("Copy inference output to host failed");
            return nullptr;
        }
    }
    else {
        data = dataBufferDev;
    }

    itemDataSize = bufferSize;
    return data;
}

void FaceDetect::EncodeImage(vector<uint8_t>& encodeImg, cv::Mat& origImg)
{
    vector<int> param = vector<int>(2);
    param[0] = CV_IMWRITE_JPEG_QUALITY;
    param[1] = 95;//default(95) 0-100
    //Jpeg images must serialize the Proto message before they can be sent
    cv::imencode(".jpg", origImg, encodeImg, param);
}

Result FaceDetect::SendImage(vector<DetectionResult>& detectionResults, cv::Mat& origImg)
{
    vector<uint8_t> encodeImg;
    EncodeImage(encodeImg, origImg);

    ImageFrame imageParam;
    imageParam.format = ImageFormat::kJpeg;
    imageParam.width = origImg.cols;
    imageParam.height = origImg.rows;
    imageParam.size = encodeImg.size();
    imageParam.data = reinterpret_cast<uint8_t*>(encodeImg.data());
    imageParam.detection_results = detectionResults;
    //Sends the detected object frame information and frame image to the Presenter Server for display
    PresenterErrorCode errorCode = PresentImage(channel_, imageParam);
    if (errorCode != PresenterErrorCode::kNone) {
        ERROR_LOG("PresentImage failed %d", static_cast<int>(errorCode));
        return FAILED;
    }

    return SUCCESS;
}

void FaceDetect::DestroyResource()
{
    aclrtFree(imageDataBuf_);

    delete channel_;

    //The ACL resource held by the model instance must be released before the ACL exits or ABORT will be torn down
    model_.DestroyResource();

    aclError ret = aclrtResetDevice(deviceId_);
    if (ret != ACL_ERROR_NONE) {
        ERROR_LOG("reset device failed\n");
    }
    INFO_LOG("end to reset device is %d\n", deviceId_);

    ret = aclFinalize();
    if (ret != ACL_ERROR_NONE) {
        ERROR_LOG("finalize acl failed\n");
    }
    INFO_LOG("end to finalize acl");
}