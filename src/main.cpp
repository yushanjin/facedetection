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

* File main.cpp
* Description: dvpp sample main func
*/

#include <iostream>
#include <stdlib.h>
#include <dirent.h>
#include <thread>
#include <future>
#include <chrono>
#include <queue>

#include "face_detect.h"
#include "model_process.h"
#include "utils.h"

using namespace std::chrono;
using namespace std;

namespace {
const uint32_t kModelWidth3 = 304;
const uint32_t kModelHeight3 = 300;
const char* kModelPath3 = "../model/face_detection.om";
}

string videoFile = "../data/person.mp4";

int main(int argc, char *argv[]) {
    //检查应用程序执行时的输入,程序执行的参数为输入视频文件路径
    if((argc < 2) || (argv[1] == nullptr)){
        ERROR_LOG("Please input: ./main <image_dir>");
        return FAILED;
    }

    //使用opencv打开视频流
    videoFile = string(argv[1]);
    printf("open %s\n", videoFile.c_str());
    cv::VideoCapture capture(videoFile);
    if (!capture.isOpened()) {
        cout << "Movie open Error" << endl;
        return FAILED;
    }

    //获取视频流信息（分别率、fps等）
    cout << "width = " << capture.get(3) << endl;
    cout << "height = " << capture.get(4) << endl;
    cout << "frame_fps = " << capture.get(5) << endl;
    cout << "frame_nums = " << capture.get(7) << endl;

    FaceDetect face_detect(kModelPath3, kModelWidth3, kModelHeight3);
    //Initializes the ACL resource for categorical reasoning, loads the model and requests the memory used for reasoning input
    Result ret1 = face_detect.Init("face_detection_video_rtsp");
    if (ret1 != SUCCESS) {
        ERROR_LOG("FaceDetection Init resource failed");
        return FAILED;
    }
    //Frame by frame reasoning
    while(1) {
        // 1 Read a frame of an image
        cv::Mat frame;
        if (!capture.read(frame)) {
            INFO_LOG("Video capture return false");
            break;
        }
        //对帧图片进行预处理
        high_resolution_clock::time_point start = high_resolution_clock::now();
        Result ret = face_detect.Preprocess(frame);
        if (ret != SUCCESS) {
            ERROR_LOG("Read file %s failed, continue to read next",
            videoFile.c_str());
            continue;
        }
        high_resolution_clock::time_point end = high_resolution_clock::now();
        duration < double, std::milli > time_span = end - start;
        cout << "\nPreProcess time " << time_span.count() << "ms" << endl;

        //将预处理的图片送入模型推理,并获取推理结果
        aclmdlDataset* inferenceOutput = nullptr;
        start = high_resolution_clock::now();
        ret = face_detect.Inference(inferenceOutput);
        if ((ret != SUCCESS) || (inferenceOutput == nullptr)) {
            ERROR_LOG("Inference model inference output data failed");
            return FAILED;
        }
        end = high_resolution_clock::now();
        time_span = end - start;
        cout << "Inference time " << time_span.count() << "ms" << endl;

        //解析推理输出,并将推理得到的物体类别,置信度和图片送到presenter server显示
        start = high_resolution_clock::now();
        ret = face_detect.Postprocess(frame, inferenceOutput);
        if (ret != SUCCESS) {
            ERROR_LOG("Process model inference output data failed");
            return FAILED;
        }
        end = high_resolution_clock::now();
        time_span = end - start;
        cout << "PostProcess time " << time_span.count() << "ms" << endl;
    }

    INFO_LOG("Execute video face detection success");

    return SUCCESS;

}
