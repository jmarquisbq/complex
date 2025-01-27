#pragma once

#include "SimplnxCore/SimplnxCore_export.hpp"

#include "simplnx/Filter/Arguments.hpp"
#include "simplnx/Filter/FilterTraits.hpp"
#include "simplnx/Filter/IFilter.hpp"
#include "simplnx/Filter/Parameters.hpp"

namespace nx::core
{
/**
 * @class WriteNodesAndElementsFilesFilter
 * @brief The WriteNodesAndElementsFilesFilter is an IFilter class designed to export the
 * DataStructure to a target HDF5 file.
 */
class SIMPLNXCORE_EXPORT WriteNodesAndElementsFilesFilter : public IFilter
{
public:
  WriteNodesAndElementsFilesFilter() = default;
  ~WriteNodesAndElementsFilesFilter() noexcept override = default;

  WriteNodesAndElementsFilesFilter(const WriteNodesAndElementsFilesFilter&) = delete;
  WriteNodesAndElementsFilesFilter(WriteNodesAndElementsFilesFilter&&) noexcept = delete;

  WriteNodesAndElementsFilesFilter& operator=(const WriteNodesAndElementsFilesFilter&) = delete;
  WriteNodesAndElementsFilesFilter& operator=(WriteNodesAndElementsFilesFilter&&) noexcept = delete;

  // Parameter Keys
  static inline constexpr StringLiteral k_SelectedGeometry = "selected_geometry_path";
  static inline constexpr StringLiteral k_WriteNodeFile = "write_node_file";
  static inline constexpr StringLiteral k_NumberNodes = "number_nodes";
  static inline constexpr StringLiteral k_IncludeNodeFileHeader = "include_node_file_header";
  static inline constexpr StringLiteral k_NodeFilePath = "node_file_path";
  static inline constexpr StringLiteral k_WriteElementFile = "write_element_file";
  static inline constexpr StringLiteral k_NumberElements = "number_elements";
  static inline constexpr StringLiteral k_IncludeElementFileHeader = "include_element_file_header";
  static inline constexpr StringLiteral k_ElementFilePath = "element_file_path";

  /**
   * @brief Reads SIMPL json and converts it simplnx Arguments.
   * @param json
   * @return Result<Arguments>
   */
  static Result<Arguments> FromSIMPLJson(const nlohmann::json& json);

  /**
   * @brief Returns the name of the filter class.
   * @return std::string
   */
  std::string name() const override;

  /**
   * @brief Returns the C++ classname of this filter.
   * @return
   */
  std::string className() const override;

  /**
   * @brief Returns the WriteNodesAndElementsFilesFilter class's UUID.
   * @return Uuid
   */
  Uuid uuid() const override;

  /**
   * @brief Returns the human readable name of the filter.
   * @return std::string
   */
  std::string humanName() const override;

  /**
   * @brief Returns the default tags for this filter.
   * @return
   */
  std::vector<std::string> defaultTags() const override;

  /**
   * @brief Returns a collection of the filter's parameters (i.e. its inputs)
   * @return Parameters
   */
  Parameters parameters() const override;

  /**
   * @brief Returns parameters version integer.
   * Initial version should always be 1.
   * Should be incremented everytime the parameters change.
   * @return VersionType
   */
  VersionType parametersVersion() const override;

  /**
   * @brief Returns a copy of the filter as a std::unique_ptr.
   * @return UniquePointer
   */
  UniquePointer clone() const override;

protected:
  /**
   * @brief Classes that implement IFilter must provide this function for preflight.
   * Runs after the filter runs the checks in its parameters.
   * @param dataStructure
   * @param args
   * @param messageHandler
   * @return Result<OutputActions>
   */
  PreflightResult preflightImpl(const DataStructure& dataStructure, const Arguments& args, const MessageHandler& messageHandler, const std::atomic_bool& shouldCancel) const override;

  /**
   * @brief Classes that implement IFilter must provide this function for execute.
   * Runs after the filter applies the OutputActions from preflight.
   * @param dataStructure
   * @param args
   * @param pipelineNode
   * @param messageHandler
   * @return Result<>
   */
  Result<> executeImpl(DataStructure& dataStructure, const Arguments& args, const PipelineFilter* pipelineNode, const MessageHandler& messageHandler,
                       const std::atomic_bool& shouldCancel) const override;
};
} // namespace nx::core

SIMPLNX_DEF_FILTER_TRAITS(nx::core, WriteNodesAndElementsFilesFilter, "8c563174-0183-45fe-8ef8-756104f215d5");
