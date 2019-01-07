//Tencent is pleased to support the open source community by making FeatherCNN available.

//Copyright (C) 2019 THL A29 Limited, a Tencent company. All rights reserved.

//Licensed under the BSD 3-Clause License (the "License"); you may not use this file except
//in compliance with the License. You may obtain a copy of the License at
//
//https://opensource.org/licenses/BSD-3-Clause
//
//Unless required by applicable law or agreed to in writing, software distributed
//under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
//CONDITIONS OF ANY KIND, either express or implied. See the License for the
//specific language governing permissions and limitations under the License.
#pragma once
#include <stdlib.h>
#include <cstring>
#include <fstream>
#include <functional>
#include <limits>
#include <map>
#include <string>
#include <unordered_map>
#include <vector>

#include "common.h"

namespace clhpp_feather
{

inline bool IsTuning()
{
    const char *tuning = getenv("FEATHER_TUNING");
    return tuning != nullptr && strlen(tuning) == 1 && tuning[0] == '1';
}
inline bool IsTunned()
{
    const char *tuning = getenv("FEATHER_TUNING");
    return tuning != nullptr && strlen(tuning) == 1 && tuning[0] == '9';
}
inline bool IsTunerInProcess()
{
    const char *tuning = getenv("FEATHER_TUNING");
    return tuning != nullptr && strlen(tuning) == 1 && tuning[0] == '2';
}


template <typename param_type>
class Tuner
{
    public:

        Tuner()
        {
            tuned_param_file_path_.clear();
            path_ = nullptr;
            frame_ = 0;
            tuner_size_ = 10; //default
        }

        //net forward finish call
        void SetTunerPram() {
            if(clhpp_feather::IsTunerInProcess()) 
            {
                frame_++;
                if(frame_ >= tuner_size_)
                {
                    setenv("FEATHER_TUNING", "9", 1);
                    LOGI("SET FEATHER_TUNING = 9");
                }
                //LOGI("frame_ is %d", frame_);
            }
            if(clhpp_feather::IsTuning())
            {
                setenv("FEATHER_TUNING", "9", 1);
                //LOGI("SET FEATHER_TUNING = 9");
            }
        }

        int IsTunerInProcess(size_t kwg_size,
                 const size_t& height,
                 const size_t& width,
                 std::vector<param_type> gws,
                 std::vector<param_type> lws,
                 std::vector<std::vector<param_type> >& gws_list,
                 std::vector<std::vector<param_type> >& lws_list)
        {
            int index = frame_ % tuner_size_;
            std::vector<std::vector<param_type> > gws_list_tm;
            std::vector<std::vector<param_type> > lws_list_tm;
            TunerArry(kwg_size, height, width, gws, lws, gws_list_tm, lws_list_tm);
            gws_list.push_back(gws_list_tm[index]);
            lws_list.push_back(lws_list_tm[index]);
            return 0;
        }

        int TunerArry(size_t kwg_size,
                 const size_t& height,
                 const size_t& width,
                 std::vector<param_type> gws,
                 std::vector<param_type> lws,
                 std::vector<std::vector<param_type> >& gws_list,
                 std::vector<std::vector<param_type> >& lws_list)
        {
            //gws HWC
            //lws  HWC
            size_t lws_c = lws[2];
            //LOGI("kwg_size %d lws_c %d", kwg_size, lws_c);
            std::vector<std::vector<size_t>> candidates =
            {
                {kwg_size / 2 / lws_c, 2, lws_c},     {kwg_size / 4 / lws_c, 4, lws_c},
                {kwg_size / 8 / lws_c, 8, lws_c},     {kwg_size / 16 / lws_c, 16, lws_c},
                {kwg_size / 32 / lws_c, 32, lws_c},   {kwg_size / 64 / lws_c, 64, lws_c},
                {kwg_size / 128 / lws_c, 128, lws_c}, {kwg_size / 256 / lws_c, 256, lws_c},
                {kwg_size / lws_c, 1, lws_c},         {1, kwg_size / lws_c, lws_c}
            };
            for (int i = 0; i < candidates.size() ; i++)
            {
                if (candidates[i][0] < 1 || candidates[i][1] < 1)
                    continue;
                lws[0] = candidates[i][0];
                lws[1] = candidates[i][1];
                gws[0] = (height / lws[0] + !!(height % lws[0])) * lws[0];
                gws[1] = (width / lws[1]  + !!(width % lws[1])) * lws[1];
                lws_list.push_back({lws[0], lws[1], lws[2]});
                gws_list.push_back({gws[0], gws[1], gws[2]});
            }
            tuner_size_ = tuner_size_ < gws_list.size() ? gws_list.size() : tuner_size_;
            return 0;
        }

        inline void WriteRunParameters()
        {
            if (path_ != nullptr)
            {
                LOGI("Write tuning result to %s", path_);
                std::ofstream ofs(path_, std::ios::binary | std::ios::out);
                if (ofs.is_open())
                {
                    int64_t num_pramas = param_table_.size();
                    ofs.write(reinterpret_cast<char *>(&num_pramas), sizeof(num_pramas));
                    for (auto &kp : param_table_)
                    {
                        int32_t key_size = kp.first.size();
                        ofs.write(reinterpret_cast<char *>(&key_size), sizeof(key_size));
                        ofs.write(kp.first.c_str(), key_size);

                        auto &params = kp.second;
                        int32_t params_size = params.size() * sizeof(param_type);
                        ofs.write(reinterpret_cast<char *>(&params_size),
                                  sizeof(params_size));

                        LOGI("Write tuning param: %s", kp.first.c_str());
                        for (auto &param : params)
                        {
                            ofs.write(reinterpret_cast<char *>(&param), sizeof(params_size));
                        }
                    }
                    ofs.close();
                }
                else
                {
                    LOGE("Write run parameter file failed.");
                }
            }
        }

        inline void ReadRunParamters()
        {
            if (!tuned_param_file_path_.empty())
            {
                std::ifstream ifs(tuned_param_file_path_,
                                  std::ios::binary | std::ios::in);
                if (ifs.is_open())
                {
                    int64_t num_params = 0;
                    ifs.read(reinterpret_cast<char *>(&num_params), sizeof(num_params));
                    while (num_params--)
                    {
                        int32_t key_size = 0;
                        ifs.read(reinterpret_cast<char *>(&key_size), sizeof(key_size));
                        std::string key(key_size, ' ');
                        ifs.read(&key[0], key_size);

                        int32_t params_size = 0;
                        ifs.read(reinterpret_cast<char *>(&params_size), sizeof(params_size));
                        int32_t params_count = params_size / sizeof(unsigned int);
                        std::vector<unsigned int> params(params_count);
                        for (int i = 0; i < params_count; ++i)
                        {
                            ifs.read(reinterpret_cast<char *>(&params[i]),
                                     sizeof(unsigned int));
                        }
                        param_table_.emplace(key, params);
                    }
                    ifs.close();
                }
                else
                {
                    LOGE("Read OpenCL tuned parameters file failed.");
                }
            }
            else
            {
                LOGI("There is no tuned parameters.");
            }
        }

        bool set_layer_kernel_wks(const std::string &key, std::vector<param_type>& value, double time = 1.0)
        {
            if (this->param_time_.find(key) != this->param_time_.end())
            {
                if(this->param_time_[key] > time)
                {
                    this->param_table_[key] = value;
                    this->param_time_[key] = time; 
                    LOGE("Tuner param_table_ set [%s] %f", key.c_str(), time);
                }
            }
            else {
                this->param_table_[key] = value;
                this->param_time_[key] = time;
                LOGE("Tuner param_table_ set [%s] %f", key.c_str(), time);
            }
            return true;
        }

        bool get_layer_kernel_wks(const std::string &key, std::vector<param_type>& value)
        {
            if (param_table_.find(key) == param_table_.end())
            {
                LOGE("Tuner can not find %s", key.c_str());
                return false;
            }
            value = param_table_[key];
            return true;
        }

    private:
        std::string tuned_param_file_path_;
        char *path_;
        std::unordered_map<std::string, std::vector<param_type>> param_table_;
        std::unordered_map<std::string, double> param_time_;
        int frame_;
        int tuner_size_;
};

}  // namespace clhpp_feather
