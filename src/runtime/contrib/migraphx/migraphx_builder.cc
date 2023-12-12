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

#include <tvm/runtime/ndarray.h>

#include <memory>
#include <string>

#include "migraphx_builder.h"

namespace tvm {
namespace runtime {
namespace contrib {

MIGraphXBuilder::MIGraphXBuilder(const std::vector<const DLTensor*>& data_entry,
                                 bool use_fp16)
    : 
      p_(),
      m_ptr_(p_.get_main_module()),
      data_entry_(data_entry),
      use_fp16_(use_fp16) {}

migraphx::shape::type_t DLDataType2MIGraphXDataType(DLDataType data_type) {
  ICHECK(data_type.code == kDLFloat && (data_type.bits == 16 || data_type.bits == 32))
      << "Invalid input Tensor type. Only float16 and float32 are supported";
  return (data_type.bits == 16) ? migraphx::shape::half_type : migraphx::shape::float_type;
}

void MIGraphXBuilder::AddInput(int nid, uint32_t entry_id, const JSONGraphNode& node) {
  auto node_name = node.GetOpName();
  auto shapes = node.GetOpShape();
  auto dtypes = node.GetOpDataType();
  ICHECK_EQ(shapes.size(), dtypes.size());
  node_output_map_[nid] = {};
  for (size_t i = 0; i < shapes.size(); ++i) {
    const std::string name = node_name + "_" + std::to_string(i);
    auto shape = shapes[i];
    bool is_dynamic = false;
    for (const auto& dim: shape) {
      if (dim == -1) {
        is_dynamic = true;
        break;
      }
    }
    const auto dtype = DLDataType2MIGraphXDataType(dtypes[i]);
    migraphx::shape s;
    if (is_dynamic) {
      std::vector<migraphx::shape::dynamic_dimension> dyn_dims;
      for (const auto& dim: shape) {
        if (dim == -1) {
          dyn_dims.push_back({1, 1});
        } else {
          dyn_dims.push_back({dim, dim, {dim}});
        }
      }
      s = {dtype, dyn_dims};
    } else {
      s = {dtype, shape};
    }
    auto input_tensor = m_ptr_->add_parameter(name, s);
    node_output_map_[nid].push_back(MIGraphXOpInput{input_tensor});
    network_input_names_.push_back(name);
    // entry_id_map_[name] = entry_id + i;
    network_input_names_shape_map_[name] = s;
  }
}

std::unordered_map<std::string, migraphx::shape> MIGraphXBuilder::GetInputNameToShapeMap() {
  return network_input_names_shape_map_;
}

void MIGraphXBuilder::AddConstant(int nid, const DLTensor* data) {
  migraphx::literal weight = GetDLTensorAsWeights(data, kDLCPU);
  std::vector<int> shape(data->shape, data->shape + data->ndim);
  node_output_map_[nid] = {MIGraphXOpInput(weight, shape)};
}

void MIGraphXBuilder::AddOutput(const JSONGraphNodeEntry& node, uint32_t entry_id) {
  auto it = node_output_map_.find(node.id_);
  ICHECK(it != node_output_map_.end()) << "Output was not found.";
  auto out_tensor = it->second[node.index_].tensor;
  network_output_tensors_.push_back(out_tensor);
  // std::string name = "migraphx_output_" + std::to_string(network_output_names_.size());
  // // If the network is already marked as an input or output, make a copy to avoid TRT crash.
  // if (out_tensor->isNetworkOutput()) {
  //   LOG(WARNING) << name << " is a duplicate output.";
  //   out_tensor = network_->addIdentity(*out_tensor)->getOutput(0);
  // } else if (out_tensor->isNetworkInput()) {
  //   LOG(WARNING) << name << " is both an input and an output.";
  //   out_tensor = network_->addIdentity(*out_tensor)->getOutput(0);
  // }
  // out_tensor->setName(name.c_str());
  // network_->markOutput(*out_tensor);
  // network_output_names_.push_back(name);
  // entry_id_map_[name] = entry_id;
}

void MIGraphXBuilder::AddLayer(int nid, const JSONGraphNode& node) {
  MIGraphXOpConverterParams params(m_ptr_, nid, node, &weights_);
  // Look up converter.
  const std::unordered_map<std::string, std::unique_ptr<MIGraphXOpConverter>>& map =
      GetOpConverters();
  auto it = map.find(params.op_name);
  ICHECK(it != map.end()) << params.op_name << ": Unsupported operator";
  const MIGraphXOpConverter& converter = *it->second;
  if (!converter.variable_input_count) {
    ICHECK_EQ(node.GetInputs().size(), converter.input_types.size())
        << params.op_name << ": Mismatched input sizes";
  }
  // Get inputs.
  for (size_t i = 0; i < node.GetInputs().size(); ++i) {
    VLOG(1) << "TODO for loop node inputs i: " << i << "\n";
    auto in_node = node.GetInputs()[i];
    auto it = node_output_map_.find(in_node.id_);
    ICHECK(it != node_output_map_.end()) << params.op_name << ": Input was not found";
    auto input = it->second[in_node.index_];
    if (!converter.variable_input_count) {
      if (converter.input_types[i] == kTensor && input.type == kWeight) {
        input = MIGraphXOpInput(GetInputAsTensor(input));
      } else if (converter.input_types[i] == kWeight && input.type == kTensor) {
        LOG(FATAL) << params.op_name << ": Input " << i << " must be a constant.";
      }
    }
    params.inputs.push_back(input);
  }

  // Convert op to TRT.
  VLOG(1) << "TODO Convert:" << params.op_name << "\n";
  converter.Convert(&params);

  // Get outputs.
  node_output_map_[nid] = {};
  std::vector<DLDataType> dtype = node.GetOpDataType();
  ICHECK_EQ(params.outputs.size(), dtype.size()) << params.op_name << ": Mismatched output sizes";
  for (size_t i = 0; i < params.outputs.size(); ++i) {
    auto out = params.outputs[i];
    // out->setType(DLDataType2MIGraphXDataType(dtype[i]));
    node_output_map_[nid].push_back(MIGraphXOpInput(out));
  }
}

void MIGraphXBuilder::AddReturn() {
  m_ptr_->add_return(network_output_tensors_);
}

migraphx::literal MIGraphXBuilder::CreateLiteral(migraphx::shape::type_t shape_type, const std::vector<size_t>& dims, const char* data)
{
    // empty input
    auto elem_num =
        std::accumulate(dims.begin(), dims.end(), std::size_t(1), std::multiplies<std::size_t>());
    if(elem_num == 0)
    {
        return migraphx::literal{shape_type};
    }

    // in case of scalar constants in onnx file, use dims=1 to fill initializer data
    if(dims.empty())
        return migraphx::literal{{shape_type}, data};
    return migraphx::literal{{shape_type, dims}, data};
}

// template <class T, MIGRAPHX_REQUIRES(not std::is_pointer<T>{})>
// migraphx::literal MIGraphXBuilder::CreateLiteral(migraphx::shape::type_t shape_type, const std::vector<size_t>& dims, T data)
// {
//     // empty input
//     auto elem_num =
//         std::accumulate(dims.begin(), dims.end(), std::size_t(1), std::multiplies<std::size_t>());
//     if(elem_num == 0)
//     {
//         return migraphx::literal{shape_type};
//     }

//     // scalar input
//     if(dims.empty())
//         return migraphx::literal{{shape_type}, data.begin(), data.end()};
//     return migraphx::literal{{shape_type, dims}, data.begin(), data.end()};
// }

migraphx::literal MIGraphXBuilder::GetDLTensorAsWeights(const DLTensor* dptr,
                                                        DLDeviceType src_device) {
  ICHECK_EQ(dptr->device.device_type, src_device);
  ICHECK((dptr->dtype.bits != 16 || dptr->dtype.bits != 32))
      << "Invalid input Tensor type. Float16 and Float32 are supported";
  const auto dtype = (static_cast<int>(dptr->dtype.bits) == 16) ? migraphx::shape::half_type
                                                                    : migraphx::shape::float_type;

  const size_t weight_bytes = GetDataSize(*dptr);
  std::vector<size_t> dims(dptr->ndim);
  for (tvm_index_t i = 0; i < dptr->ndim; ++i) {
    VLOG(1) << "TODO dims: " << i << " dim: " << dptr->shape[i] << "\n";
    dims[i] = dptr->shape[i];
  }
  size_t shape_size = std::accumulate(dims.begin(), dims.end(), 1, std::multiplies<size_t>());
  migraphx::literal weight;
  VLOG(1) << "TODO bind constant dptr->ndim: " << dptr->ndim << " weight_bytes: " << weight_bytes << " shape size: " << shape_size << " \n";
  switch(dtype) {
      VLOG(1) << "dtype half\n";
    case migraphx::shape::half_type: {
      std::vector<uint16_t> data_vals(shape_size);
      ICHECK_EQ(TVMArrayCopyToBytes(const_cast<DLTensor*>(dptr), const_cast<void*>(static_cast<void*>(data_vals.data())),
                                      weight_bytes),
                  0)
          << TVMGetLastError();
      std::vector<migraphx::half> data_half;
      std::transform(data_vals.begin(),
                      data_vals.end(),
                      std::back_inserter(data_half),
                      [](uint16_t raw_val) { return *reinterpret_cast<migraphx::half*>(&raw_val); });
      weight = CreateLiteral(dtype, dims, data_half);
      break;
    }
    case migraphx::shape::float_type: {
      VLOG(1) << "dtype float\n";
      std::vector<float> data_vals(shape_size);
      ICHECK_EQ(TVMArrayCopyToBytes(const_cast<DLTensor*>(dptr), const_cast<void*>(static_cast<void*>(data_vals.data())),
                                      weight_bytes),
                  0)
          << TVMGetLastError();
      VLOG(1) << "dtype float CreateLiteral\n";
      weight = CreateLiteral(dtype, dims, data_vals);
      break;
    }
    default:
      VLOG(1) << "dtype not supported\n";
        throw std::runtime_error("");
  }
  VLOG(1) << "bind constant end\n";

  weights_.push_back(weight);
  return weight;
}

migraphx::instruction_ref MIGraphXBuilder::GetInputAsTensor(const MIGraphXOpInput& input) {
  if (input.type == kTensor) return input.tensor;
  auto shape = input.weight_shape;
  return m_ptr_->add_literal(input.weight);
}

MIGraphXContext MIGraphXBuilder::Compile() {
  std::string target_name = "gpu";
  auto target = migraphx::make_target(target_name);
  auto options = migraphx::compile_options{};
  options.offload_copy = false;
  options.fast_math = true;
  options.exhaustive_tune = false;
  VLOG(1) << "Migraphx program: " << p_ << std::endl;
  p_.compile(target, options);
  VLOG(1) << "Compiled migraphx program: " << p_ << std::endl;

  std::vector<std::string> parameter_names = p_.get_parameter_names();
  for (const auto &param_name: parameter_names) {
    VLOG(1) << "TODO params: " << param_name << "\n";
    if (!migraphx::contains(network_input_names_, param_name)) {
      VLOG(1) << "TODO output params: " << param_name << "\n";
      network_output_names_.push_back(param_name);
    }
  }
  assert(network_output_names_.size() == network_output_tensors_.size());
  for (size_t i = 0; i < network_output_names_.size(); i++) {
    auto n = network_output_names_[i];
    auto s = p_.get_parameter_shape(n);
    if (s != network_output_tensors_[i]->get_shape()) {
      std::stringstream err_msg;
      err_msg << "Output shape mismatch, name: " << n << " expected: " << network_output_tensors_[i]->get_shape() << " actual: " << s << std::endl;
      throw std::runtime_error(err_msg.str());
    }
    network_output_names_shape_map_[n] = s;
  }
  MIGraphXContext migraphx_context = {
    network_input_names_shape_map_,
    network_output_names_shape_map_,
    network_input_names_,
    network_output_names_,
    p_,
  };
  return migraphx_context;
}

void MIGraphXBuilder::CleanUp() {
  VLOG(1) << "Destroying MIGraphX weights";
  weights_.clear();
}

}  // namespace contrib
}  // namespace runtime
}  // namespace tvm
