/* * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

#include "migraphx_ops.h"

#include <algorithm>
#include <cmath>
#include <memory>
#include <migraphx/instruction.hpp>
#include <migraphx/make_op.hpp>
#include <migraphx/ranges.hpp>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace tvm {
namespace runtime {
namespace contrib {

using namespace migraphx;

void check_attr_sizes(size_t kdims, size_t attr_size, const std::string& error_msg) {
  if (kdims != attr_size) {
    MIGRAPHX_THROW(error_msg + " k-dims: " + std::to_string(kdims) +
                   " attribute size: " + std::to_string(attr_size));
  }
}

MIGraphXOpConverter::MIGraphXOpConverter(std::string op_name,
                                         const std::vector<InputType>& input_types,
                                         bool variable_input_count)
    : op_name(std::move(op_name)),
      input_types(input_types),
      variable_input_count(variable_input_count) {}

// class ActivationOpConverter : public MIGraphXOpConverter {
//  public:
//   explicit ActivationOpConverter(std::string op_name)
//       : MIGraphXOpConverter(std::move(op_name)) {}
//   ~ActivationOpConverter() = default;

//   void Convert(MIGraphXOpConverterParams* params) const {
// //     static const std::unordered_map<std::string, nvinfer1::ActivationType> op_map = {
// // //       {"nn.relu", nvinfer1::ActivationType::kRELU},
// // //       {"sigmoid", nvinfer1::ActivationType::kSIGMOID},
// // //       {"tanh", nvinfer1::ActivationType::kTANH},
// //     };
// //     auto it = op_map.find(op_name);
// //     ICHECK(it != op_map.end()) << "Unsupported activation type " << op_name;

// //     ICHECK(act_layer != nullptr);
// //     params->outputs.push_back(act_layer->getOutput(0));
//   }
// };

// class Conv2DOpConverter : public MIGraphXOpConverter {
//  public:
//   explicit Conv2DOpConverter(std::string op_name)
//       : MIGraphXOpConverter(std::move(op_name), {kTensor, kWeight}) {}
//   ~Conv2DOpConverter() = default;

//   void Convert(MIGraphXOpConverterParams* params) const {
//     auto input_tensor = params->inputs.at(0).tensor;
//     auto input_shape = input_tensor->get_shape();
//     auto input_lens  = input_shape.max_lens();
//     assert(input_lens.size() > 2);
//     auto kdims = input_lens.size() - 2;
//     // auto input_dims = TrtDimsToVector(input_tensor->getDimensions());
//     auto weight_shape = params->inputs.at(1).weight_shape;
//     ICHECK_EQ(params->node.GetAttr<std::vector<std::string>>("data_layout")[0], "NCHW");
//     ICHECK(params->node.GetAttr<std::vector<std::string>>("out_layout")[0] == "" ||
//            params->node.GetAttr<std::vector<std::string>>("out_layout")[0] == "NCHW");
//     ICHECK_EQ(params->node.GetAttr<std::vector<std::string>>("kernel_layout")[0], "OIHW");
//     auto str_strides = params->node.GetAttr<std::vector<std::string>>("strides");
//     auto str_dilation = params->node.GetAttr<std::vector<std::string>>("dilation");
//     auto str_padding = params->node.GetAttr<std::vector<std::string>>("padding");
//     int groups = std::stoi(params->node.GetAttr<std::vector<std::string>>("groups")[0]);
//     int channels = weight_shape[0];
//     if (params->node.HasAttr("channels") &&
//         !params->node.GetAttr<std::vector<std::string>>("channels")[0].empty()) {
//       channels = std::stoi(params->node.GetAttr<std::vector<std::string>>("channels")[0]);
//     }

//     VLOG(1) << "TODO op_name: " << op_name << "\n";
//     auto op = make_op("convolution");
//     auto values   = op.to_value();
//     ICHECK_EQ(str_strides.size(), 2);
//     const std::vector<int> strides = {std::stoi(str_strides[0]), std::stoi(str_strides[1])};
//     ICHECK_EQ(str_dilation.size(), 2);
//     const std::vector<int> dilation = {std::stoi(str_dilation[0]), std::stoi(str_dilation[1])};

//     values["stride"].clear();
//     copy(strides, std::back_inserter(values["stride"]));
//     check_attr_sizes(kdims, values["stride"].size(), "PARSE_CONV: inconsistent strides");
//     values["dilation"].clear();
//     copy(dilation, std::back_inserter(values["dilation"]));
//     check_attr_sizes(
//         kdims, values["dilation"].size(), "PARSE_CONV: inconsistent dilations");

//     // const auto kernel_size = nvinfer1::DimsHW(weight_shape[2], weight_shape[3]);
//     // const nvinfer1::DataType weight_type = params->inputs.at(1).weight.type;
//     // nvinfer1::Weights bias{weight_type, nullptr, 0};
//     // auto conv_layer = params->network->addConvolution(*input_tensor, channels, kernel_size,
//     //                                                   params->inputs.at(1).weight, bias);
//     // ICHECK(conv_layer != nullptr);
//     // conv_layer->setName(params->LayerName().c_str());
//     // conv_layer->setPadding(prepadding);
//     // ICHECK_EQ(str_strides.size(), 2);
//     // const auto strides = nvinfer1::DimsHW(std::stoi(str_strides[0]),
//     std::stoi(str_strides[1]));
//     // conv_layer->setStride(strides);
//     // ICHECK_EQ(str_dilation.size(), 2);
//     // const auto dilation = nvinfer1::DimsHW(std::stoi(str_dilation[0]),
//     std::stoi(str_dilation[1]));
//     // conv_layer->setDilation(dilation);
//     // conv_layer->setNbGroups(groups);
//     // params->outputs.push_back(conv_layer->getOutput(0));
//   }
// };

class MatmulOpConverter : public MIGraphXOpConverter {
 public:
  explicit MatmulOpConverter(std::string op_name)
      : MIGraphXOpConverter(std::move(op_name), {kTensor, kTensor}) {}
  ~MatmulOpConverter() = default;

  void Convert(MIGraphXOpConverterParams* params) const {
    auto a0 = params->inputs.at(0).tensor;
    auto a1 = params->inputs.at(1).tensor;
    const auto s0 = a0->get_shape();
    const auto s1 = a1->get_shape();
    instruction_ref dot_res;
    bool is_a_prepended = false;
    bool is_b_appended = false;
    if (s0.ndim() == 1) {
      is_a_prepended = true;
      a0 = params->module->add_instruction(make_op("unsqueeze", {{"axes", {0}}}), a0);
    }
    if (s1.ndim() == 1) {
      is_b_appended = true;
      a1 = params->module->add_instruction(make_op("unsqueeze", {{"axes", {1}}}), a1);
    }
    if (s0.dynamic() or s1.dynamic()) {
      auto s0_dds = a0->get_shape().to_dynamic().dyn_dims();
      auto s1_dds = a1->get_shape().to_dynamic().dyn_dims();
      VLOG(1) << "TODO s0: " << s0 << "\n";
      VLOG(1) << "TODO s1: " << s1 << "\n";
      VLOG(1) << "TODO s0_dds: " << s0.to_dynamic() << "\n";
      VLOG(1) << "TODO s1_dds: " << s1.to_dynamic() << "\n";

      // TODO: handling this case requires a new multibroadcast mode
      if (not std::equal(s0_dds.rbegin() + 2, s0_dds.rend(), s1_dds.rbegin() + 2, s1_dds.rend())) {
        MIGRAPHX_THROW("PARSE_MATMUL: dynamic shape broadcasting not supported");
      }

      dot_res = params->module->add_instruction(make_op("dot"), a0, a1);
    } else {
      auto s0_lens = a0->get_shape().lens();
      auto s1_lens = a1->get_shape().lens();
      instruction_ref ba0 = a0;
      instruction_ref ba1 = a1;
      // try broadcasting if dimensions other than last two do not match
      if (not std::equal(s0_lens.rbegin() + 2, s0_lens.rend(), s1_lens.rbegin() + 2,
                         s1_lens.rend())) {
        auto l0_it = s0_lens.begin() + s0_lens.size() - 2;
        std::vector<std::size_t> l0_broadcasted_lens(s0_lens.begin(), l0_it);
        auto l1_it = s1_lens.begin() + s1_lens.size() - 2;
        std::vector<std::size_t> l1_broadcasted_lens(s1_lens.begin(), l1_it);
        auto output_lens = compute_broadcasted_lens(l0_broadcasted_lens, l1_broadcasted_lens);
        l0_broadcasted_lens = output_lens;
        l0_broadcasted_lens.insert(l0_broadcasted_lens.end(), l0_it, s0_lens.end());
        l1_broadcasted_lens = output_lens;
        l1_broadcasted_lens.insert(l1_broadcasted_lens.end(), l1_it, s1_lens.end());
        if (s0_lens != l0_broadcasted_lens) {
          ba0 = params->module->add_instruction(
              make_op("multibroadcast", {{"out_lens", l0_broadcasted_lens}}), a0);
        }
        if (s1_lens != l1_broadcasted_lens) {
          ba1 = params->module->add_instruction(
              make_op("multibroadcast", {{"out_lens", l1_broadcasted_lens}}), a1);
        }
      }
      dot_res = params->module->add_instruction(make_op("dot"), ba0, ba1);
    }

    // squeeze the appended or prepended dimensions
    int64_t num_axis = dot_res->get_shape().ndim();
    if (is_a_prepended) {
      dot_res =
          params->module->add_instruction(make_op("squeeze", {{"axes", {num_axis - 2}}}), dot_res);
      --num_axis;
    }
    if (is_b_appended) {
      dot_res =
          params->module->add_instruction(make_op("squeeze", {{"axes", {num_axis - 1}}}), dot_res);
    }

    params->outputs.push_back(dot_res);
  }
};

const std::unordered_map<std::string, std::unique_ptr<MIGraphXOpConverter>>& GetOpConverters() {
  static const std::unordered_map<std::string, std::unique_ptr<MIGraphXOpConverter>>* map = []() {
    std::vector<std::unique_ptr<MIGraphXOpConverter>> all_converters;
    // all_converters.emplace_back(std::make_unique<ActivationOpConverter>("clip"));
    // all_converters.emplace_back(std::make_unique<Conv2DOpConverter>("nn.conv2d"));
    all_converters.emplace_back(std::make_unique<MatmulOpConverter>("matmul"));
    auto* map = new std::unordered_map<std::string, std::unique_ptr<MIGraphXOpConverter>>();
    for (auto& converter : all_converters) {
      map->emplace("migraphx." + converter->op_name, std::move(converter));
    }
    return map;
  }();
  return *map;
}

}  // namespace contrib
}  // namespace runtime
}  // namespace tvm
