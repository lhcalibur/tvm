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

/*!
 * \file runtime/contrib/tensorrt/tensorrt_ops.h
 * \brief Converters from Relay ops into MIGraphX layers. Converters should
 * inherit from MIGraphXOpConverter and implement the Convert() method.
 */

#ifndef TVM_RUNTIME_CONTRIB_MIGRAPHX_MIGRAPHX_OPS_H_
#define TVM_RUNTIME_CONTRIB_MIGRAPHX_MIGRAPHX_OPS_H_

#include <algorithm>
#include <cmath>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "../json/json_node.h"
#include <migraphx/program.hpp>
#include <migraphx/shape.hpp>
#include <migraphx/instruction_ref.hpp>
#include <migraphx/instruction.hpp>
#include <migraphx/literal.hpp>
#include <migraphx/common.hpp>


namespace tvm {
namespace runtime {
namespace contrib {

using JSONGraphNode = tvm::runtime::json::JSONGraphNode;

enum InputType { kTensor, kWeight, kIgnored };

struct MIGraphXOpInput {
  /*! \brief If type is kTensor, will store input tensor. */
  migraphx::instruction_ref tensor;

  /*! \brief If type is kWeight, will store input weight. */
  migraphx::literal weight;

  /*! \brief Whether the input is in tensor or weight. */
  InputType type;

  /*! \brief If type is kWeight, will store weight shape. */
  std::vector<int> weight_shape;

  explicit MIGraphXOpInput(migraphx::instruction_ref tensor)
      : tensor(tensor), weight{tensor->get_shape().type()}, type(kTensor) {}
  MIGraphXOpInput(migraphx::literal weight, const std::vector<int>& shape)
      : tensor(nullptr), weight(weight), type(kWeight), weight_shape(shape) {}
};

/*! \brief Parameters to convert an Op from Relay to MIGraphX. */
struct MIGraphXOpConverterParams {
  /*! \brief The MIGraphX module that the new layer should be added to. */
  migraphx::module* module;
  /*! \brief Index of JSON node. */
  int nid;
  /*! \brief The corresponding JSON node. */
  const JSONGraphNode& node;
  /*! \brief The type of op. */
  std::string op_name;
  /*! \brief Inputs to the op. */
  std::vector<MIGraphXOpInput> inputs;
  /*! \brief Outputs of the op should be populated here during Convert(). */
// TODO
//   std::vector<migraphx::instruction_ref*> outputs;
  std::vector<migraphx::instruction_ref> outputs;
  /*! \brief Any newly allocated weights should be stored here also. */
  std::vector<migraphx::literal>* weights;

  MIGraphXOpConverterParams(migraphx::module* module, int nid,
                            const JSONGraphNode& node, std::vector<migraphx::literal>* weights)
      : module(module), nid(nid), node(node), weights(weights) {
    op_name = node.GetOpName();
  }

  std::string LayerName() const { return op_name + "(" + std::to_string(nid) + ")"; }
};



/*! \brief Base class for an op converter from Relay to TRT. */
class MIGraphXOpConverter {
 public:
  virtual ~MIGraphXOpConverter() = default;

  /*! \brief Operator name. */
  std::string op_name;
  /*! \brief Used to specify whether each input is tensor or weight. */
  const std::vector<InputType> input_types;
  /*! \brief If set to true, any number of tensor inputs can be used for the op. */
  const bool variable_input_count;

  /*!
   * \brief Converter subclasses should call this constructor to set
   * input_types or variable_input_count.
   * \param input_types For each input to the op, there should be a
   * corresponding entry in input_types to determine whether that input should
   * be a tensor or a weight. MIGraphXBuilder will prepare inputs in
   * MIGraphXOpConverter according to this.
   * \param variable_input_count If the op can have multiple inputs, set this to
   * true. input_types vector will be ignored and any number of input tensors
   * can be used for this op. All inputs will be tensors and not weights.
   */
  MIGraphXOpConverter(std::string op_name, const std::vector<InputType>& input_types,
                      bool variable_input_count = false);

  /*!
   * \brief Convert to TRT. Implementation should use inputs and attributes
   * from the CallNode to add the corresponding TRT layers to network. Outputs
   * should be pushed to outputs vector.
   * \param params Parameters for this op.
   */
  virtual void Convert(MIGraphXOpConverterParams* params) const = 0;

};

/*!
 * \brief Get the map of available MIGraphXOpConverters, where the key is the name of the relay op.
 * \return Map of MIGraphXOpConverters.
 */
const std::unordered_map<std::string, std::unique_ptr<MIGraphXOpConverter>>& GetOpConverters();

}  // namespace contrib
}  // namespace runtime
}  // namespace tvm

#endif  // TVM_RUNTIME_CONTRIB_MIGRAPHX_MIGRAPHX_OPS_H_
