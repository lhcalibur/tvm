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

#ifndef TVM_RUNTIME_CONTRIB_MIGRAPHX_MIGRAPHX_BUILDER_H_
#define TVM_RUNTIME_CONTRIB_MIGRAPHX_MIGRAPHX_BUILDER_H_

#include <tvm/runtime/ndarray.h>

#include <string>

#include <migraphx/program.hpp>
#include <migraphx/shape.hpp>
#include <migraphx/instruction_ref.hpp>
#include <migraphx/literal.hpp>
#include <migraphx/requires.hpp>
#include <migraphx/register_target.hpp>
#include <migraphx/ranges.hpp>

#include "migraphx_ops.h"

namespace tvm {
namespace runtime {
namespace contrib {

using JSONGraphNode = tvm::runtime::json::JSONGraphNode;
using JSONGraphNodeEntry = tvm::runtime::json::JSONGraphNodeEntry;


struct MIGraphXContext {
  /*! \brief Map input names to shape*/
  std::unordered_map<std::string, migraphx::shape> network_input_names_shape_map;

  /*! \brief Map output names to shape*/
  std::unordered_map<std::string, migraphx::shape> network_output_names_shape_map;

  /*! \brief Input names. */
  std::vector<std::string> network_input_names;

  /*! \brief Output names. */
  std::vector<std::string> network_output_names;

  /*! \brief Compiled MIGraphX program*/
  migraphx::program p;
};

class MIGraphXBuilder {
  public:
    MIGraphXBuilder(const std::vector<const DLTensor*>& data_entry, bool use_fp16);
    
    void AddInput(int nid, uint32_t entry_id, const JSONGraphNode& node);

    void AddConstant(int nid, const DLTensor* data);

    void AddOutput(const JSONGraphNodeEntry& node, uint32_t entry_id);
    
    void AddLayer(int nid, const JSONGraphNode& node);

    std::unordered_map<std::string, migraphx::shape> GetInputNameToShapeMap();

    void AddReturn();
    
    MIGraphXContext Compile();

  private:
    static migraphx::literal CreateLiteral(migraphx::shape::type_t shape_type, const std::vector<size_t>& dims, const char* data);
    template <class T, MIGRAPHX_REQUIRES(not std::is_pointer<T>{})>
    static migraphx::literal CreateLiteral(migraphx::shape::type_t shape_type, const std::vector<size_t>& dims, T data) {
      // empty input
      auto elem_num =
          std::accumulate(dims.begin(), dims.end(), std::size_t(1), std::multiplies<std::size_t>());
      if(elem_num == 0)
      {
          return migraphx::literal{shape_type};
      }

      // scalar input
      if(dims.empty())
          return migraphx::literal{{shape_type}, data.begin(), data.end()};
      return migraphx::literal{{shape_type, dims}, data.begin(), data.end()};
    }

    migraphx::literal GetDLTensorAsWeights(const DLTensor* dptr,
                                                            DLDeviceType src_device);

    migraphx::instruction_ref GetInputAsTensor(const MIGraphXOpInput& input);

    void CleanUp();
    migraphx::program p_;
    migraphx::module* m_ptr_;

  /*! \brief Maps a node to its outputs. */
  std::unordered_map<int, std::vector<MIGraphXOpInput>> node_output_map_;

  /*! \brief List of all weights held in memory. */
  std::vector<migraphx::literal> weights_;

  /*! \brief Input and output tensors from TVM. */
  const std::vector<const DLTensor*>& data_entry_;

  /*! \brief Map TensorRT binding name to index in data_entry_. */
  // std::unordered_map<std::string, uint32_t> entry_id_map_;

  /*! \brief Map input names to shape*/
  std::unordered_map<std::string, migraphx::shape> network_input_names_shape_map_;

  /*! \brief Map output names to shape*/
  std::unordered_map<std::string, migraphx::shape> network_output_names_shape_map_;

  /*! \brief Whether to automatically convert model to 16-bit floating point precision. */
  bool use_fp16_;

  /*! \brief Input names. */
  std::vector<std::string> network_input_names_;

  /*! \brief Output names. */
  std::vector<std::string> network_output_names_;

  /*! \brief Output tensors. */
  std::vector<migraphx::instruction_ref> network_output_tensors_;
};


}  // namespace contrib
}  // namespace runtime
}  // namespace tvm

#endif  // TVM_RUNTIME_CONTRIB_MIGRAPHX_MIGRAPHX_BUILDER_H_