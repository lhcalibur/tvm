/*
 * Licensed to the Apache Software Foundation (ASF) under one
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

/*!
 * \file src/runtime/contrib/migraphx/migraphx_runtime.cc
 * \brief JSON runtime implementation for TensorRT.
 */

#include <dmlc/parameter.h>
#include <tvm/runtime/ndarray.h>
#include <tvm/runtime/registry.h>

#include <fstream>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "../../file_utils.h"
#include "../json/json_node.h"
#include "../json/json_runtime.h"

#include "migraphx_builder.h"


namespace tvm {
namespace runtime {
namespace contrib {


using namespace tvm::runtime::json;

class MIGraphXRuntime : public JSONRuntimeBase {
 public:
  /*!
   * \brief The MIGraphx runtime module. Deserialize the provided functions
   * on creation and store in the layer cache.
   *
   * \param symbol_name The name of the function.
   * \param graph_json serialized JSON representation of a sub-graph.
   * \param const_names The names of each constant in the sub-graph.
   */
  explicit MIGraphXRuntime(const std::string& symbol_name, const std::string& graph_json,
                           const Array<String>& const_names)
      : JSONRuntimeBase(symbol_name, graph_json, const_names),
        use_fp16_(false) {
        }

  /*!
   * \brief The type key of the module.
   *
   * \return module type key.
   */
  const char* type_key() const final { return "migraphx"; }

  /*! \brief Get the property of the runtime module .*/
  int GetPropertyMask() const final {
    return ModulePropertyMask::kBinarySerializable | ModulePropertyMask::kRunnable;
  }

  /*!
   * \brief Initialize runtime. Create TensorRT layer from JSON
   * representation.
   *
   * \param consts The constant params from compiled model.
   */
  void Init(const Array<NDArray>& consts) override {
    VLOG(1) << "Init MIGraphX runtime";
    ICHECK_EQ(consts.size(), const_idx_.size())
        << "The number of input constants must match the number of required.";
    LoadGlobalAttributes();
    SetupConstants(consts);
    BuildFromJson();
  }

  void LoadGlobalAttributes() {
    // These settings are global to the entire subgraph. Codegen will add them as attributes to all
    // op nodes. Read from first one.
    for (size_t i = 0; i < nodes_.size(); ++i) {
      if (nodes_[i].HasAttr("use_fp16")) {
        use_fp16_ = std::stoi(nodes_[i].GetAttr<std::vector<std::string>>("use_fp16")[0]);
      }
    }
  }

  /*! \brief Destroy engines and contexts. */
  void DestroyEngines() {
  }

  ~MIGraphXRuntime() override {
    VLOG(1) << "Destroying MIGraphX runtime";
    DestroyEngines();
    VLOG(1) << "Destroyed MIGraphX runtime";
  }

  /*! \brief Run inference using built engine. */
  void Run() override {
    migraphx::parameter_map m;
    for (size_t i = 0; i < input_nodes_.size(); ++i) {
      auto nid = input_nodes_[i];
      if (nodes_[nid].GetOpType() == "input") {
        for (size_t j = 0; j < nodes_[nid].GetOpShape().size(); ++j) {
          uint32_t eid = EntryID(nid, j);
          const std::string name = nodes_[nid].GetOpName() + "_" + std::to_string(j);
          VLOG(1) << "name: " << name << " TODO device_type: " <<  data_entry_[eid]->device.device_type << "\n";
          auto s = migraphx_context.network_input_names_shape_map[name];
          if (data_entry_[eid]->device.device_type == kDLROCM) {
            auto p = migraphx::argument(s, data_entry_[eid]->data);
            m[name] = p;
          } else {
            throw std::runtime_error("not implemented");
            // auto device_buffer = GetOrAllocateDeviceBuffer(eid, binding_index);
            // device_buffer.CopyFrom(data_entry_[eid]);
            // bindings[binding_index] = device_buffer->data;
          }

          // auto dims = engine->getBindingDimensions(binding_index);
          // int num_elements = 1;
          // for (int i = 0; i < dims.nbDims; ++i) num_elements *= dims.d[i];
          // binding_sizes[binding_index] = num_elements;
        }
      }
    }

    // Setup output bindings.
    for (size_t i = 0; i < outputs_.size(); ++i) {
      uint32_t eid = EntryID(outputs_[i]);
      const std::string& name = migraphx_context.network_output_names[i];
      const auto& s = migraphx_context.network_output_names_shape_map[name];
      if (data_entry_[eid]->device.device_type == kDLROCM) {
        auto p = migraphx::argument(s, data_entry_[eid]->data);
        m[name] = p;
      } else {
            throw std::runtime_error("not implemented");
        // auto device_buffer = GetOrAllocateDeviceBuffer(eid, binding_index);
        // bindings[binding_index] = device_buffer->data;
      }
    }
    auto out = migraphx_context.p.eval(m);
    migraphx_context.p.finish();
    std::vector<migraphx::argument> result(out.size());
    std::string target_name = "gpu";
    auto t = migraphx::make_target(target_name);
    std::transform(out.begin(), out.end(), result.begin(), [&](auto& argu) {
        return t.copy_from(argu);
    });

      VLOG(1) << "TODO output: \n";
    for (size_t i=0; i<10;i++) {
      VLOG(1) << result[0].element(i) <<", ";
      i++;
    }
      VLOG(1) << "\n";
  }

 private:
  void BuildFromJson() {
    VLOG(1) << "Build MIGraphX from json";
    const bool use_fp16 = dmlc::GetEnv("TVM_MIGRAPHX_USE_FP16", false) || use_fp16_;
    MIGraphXBuilder builder(data_entry_, use_fp16);
    for (size_t i = 0; i < input_nodes_.size(); ++i) {
      VLOG(1) << "TODO for loop input i: " << i << std::endl;
      auto nid = input_nodes_[i];
      const auto& node = nodes_[nid];
      std::string name = node.GetOpName();
      if (node.GetOpType() == "input") {
        VLOG(1) << "TODO add input op: " << name << std::endl;
        builder.AddInput(nid, EntryID(nid, 0), node);
      } else {
        ICHECK_EQ(node.GetOpType(), "const");
        VLOG(1) << "TODO add constant op: " << name << std::endl;
        uint32_t eid = EntryID(nid, 0);
        builder.AddConstant(nid, data_entry_[eid]);
      }
    }

    // Add layers.
    for (size_t nid = 0; nid < nodes_.size(); ++nid) {
      const auto& node = nodes_[nid];
      if (node.GetOpType() != "kernel") continue;
      builder.AddLayer(nid, node);
    }

    // Add outputs.
    for (size_t i = 0; i < outputs_.size(); ++i) {
      builder.AddOutput(outputs_[i], EntryID(outputs_[i]));
    }
    builder.AddReturn();
    migraphx_context = builder.Compile();
  };

  MIGraphXContext migraphx_context;

  /*! \brief Use auto-conversion to fp16 */
  bool use_fp16_;
};

runtime::Module MIGraphXRuntimeCreate(const String& symbol_name, const String& graph_json,
                                      const Array<String>& const_names) {
  auto n = make_object<MIGraphXRuntime>(symbol_name, graph_json, const_names);
  return runtime::Module(n);
}

TVM_REGISTER_GLOBAL("runtime.migraphx_runtime_create").set_body_typed(MIGraphXRuntimeCreate);

TVM_REGISTER_GLOBAL("runtime.module.loadbinary_migraphx")
    .set_body_typed(JSONRuntimeBase::LoadFromBinary<MIGraphXRuntime>);

}  // namespace contrib
}  // namespace runtime
}  // namespace tvm
