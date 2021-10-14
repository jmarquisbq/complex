#pragma once

#include "complex/Filter/FilterTraits.hpp"
#include "complex/Filter/IFilter.hpp"
#include "complex/complex_export.hpp"

namespace complex
{
/**
 * @class Export3dSolidMesh
 * @brief This filter will ....
 */
class COMPLEX_EXPORT Export3dSolidMesh : public IFilter
{
public:
  Export3dSolidMesh() = default;
  ~Export3dSolidMesh() noexcept override = default;

  Export3dSolidMesh(const Export3dSolidMesh&) = delete;
  Export3dSolidMesh(Export3dSolidMesh&&) noexcept = delete;

  Export3dSolidMesh& operator=(const Export3dSolidMesh&) = delete;
  Export3dSolidMesh& operator=(Export3dSolidMesh&&) noexcept = delete;

  // Parameter Keys
  static inline constexpr StringLiteral k_MeshingPackage_Key = "MeshingPackage";
  static inline constexpr StringLiteral k_outputPath_Key = "outputPath";
  static inline constexpr StringLiteral k_PackageLocation_Key = "PackageLocation";
  static inline constexpr StringLiteral k_NetgenSTLFileName_Key = "NetgenSTLFileName";
  static inline constexpr StringLiteral k_GmshSTLFileName_Key = "GmshSTLFileName";
  static inline constexpr StringLiteral k_SurfaceMeshFaceLabelsArrayPath_Key = "SurfaceMeshFaceLabelsArrayPath";
  static inline constexpr StringLiteral k_FeatureEulerAnglesArrayPath_Key = "FeatureEulerAnglesArrayPath";
  static inline constexpr StringLiteral k_FeaturePhasesArrayPath_Key = "FeaturePhasesArrayPath";
  static inline constexpr StringLiteral k_FeatureCentroidArrayPath_Key = "FeatureCentroidArrayPath";
  static inline constexpr StringLiteral k_MeshFileFormat_Key = "MeshFileFormat";
  static inline constexpr StringLiteral k_RefineMesh_Key = "RefineMesh";
  static inline constexpr StringLiteral k_MaxRadiusEdgeRatio_Key = "MaxRadiusEdgeRatio";
  static inline constexpr StringLiteral k_MinDihedralAngle_Key = "MinDihedralAngle";
  static inline constexpr StringLiteral k_OptimizationLevel_Key = "OptimizationLevel";
  static inline constexpr StringLiteral k_MeshSize_Key = "MeshSize";
  static inline constexpr StringLiteral k_LimitTetrahedraVolume_Key = "LimitTetrahedraVolume";
  static inline constexpr StringLiteral k_MaxTetrahedraVolume_Key = "MaxTetrahedraVolume";
  static inline constexpr StringLiteral k_IncludeHolesUsingPhaseID_Key = "IncludeHolesUsingPhaseID";
  static inline constexpr StringLiteral k_PhaseID_Key = "PhaseID";
  static inline constexpr StringLiteral k_TetDataContainerName_Key = "TetDataContainerName";
  static inline constexpr StringLiteral k_VertexAttributeMatrixName_Key = "VertexAttributeMatrixName";
  static inline constexpr StringLiteral k_CellAttributeMatrixName_Key = "CellAttributeMatrixName";

  /**
   * @brief Returns the name of the filter.
   * @return
   */
  std::string name() const override;

  /**
   * @brief Returns the C++ classname of this filter.
   * @return
   */
  std::string className() const override;

  /**
   * @brief Returns the uuid of the filter.
   * @return
   */
  Uuid uuid() const override;

  /**
   * @brief Returns the human readable name of the filter.
   * @return
   */
  std::string humanName() const override;

  /**
   * @brief Returns the parameters of the filter (i.e. its inputs)
   * @return
   */
  Parameters parameters() const override;

  /**
   * @brief Returns a copy of the filter.
   * @return
   */
  UniquePointer clone() const override;

protected:
  /**
   * @brief Takes in a DataStructure and checks that the filter can be run on it with the given arguments.
   * Returns any warnings/errors. Also returns the changes that would be applied to the DataStructure.
   * Some parts of the actions may not be completely filled out if all the required information is not available at preflight time.
   * @param ds The input DataStructure instance
   * @param filterArgs These are the input values for each parameter that is required for the filter
   * @param messageHandler The MessageHandler object
   * @return Returns a Result object with error or warning values if any of those occurred during execution of this function
   */
  Result<OutputActions> preflightImpl(const DataStructure& ds, const Arguments& filterArgs, const MessageHandler& messageHandler) const override;

  /**
   * @brief Applies the filter's algorithm to the DataStructure with the given arguments. Returns any warnings/errors.
   * On failure, there is no guarantee that the DataStructure is in a correct state.
   * @param ds The input DataStructure instance
   * @param filterArgs These are the input values for each parameter that is required for the filter
   * @param messageHandler The MessageHandler object
   * @return Returns a Result object with error or warning values if any of those occurred during execution of this function
   */
  Result<> executeImpl(DataStructure& ds, const Arguments& filterArgs, const MessageHandler& messageHandler) const override;
};
} // namespace complex

COMPLEX_DEF_FILTER_TRAITS(complex, Export3dSolidMesh, "e7f02408-6c01-5b56-b970-7813e64c12e2");
