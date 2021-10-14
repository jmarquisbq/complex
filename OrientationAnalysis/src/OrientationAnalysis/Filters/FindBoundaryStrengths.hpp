#pragma once

#include "complex/Filter/FilterTraits.hpp"
#include "complex/Filter/IFilter.hpp"
#include "complex/complex_export.hpp"

namespace complex
{
/**
 * @class FindBoundaryStrengths
 * @brief This filter will ....
 */
class COMPLEX_EXPORT FindBoundaryStrengths : public IFilter
{
public:
  FindBoundaryStrengths() = default;
  ~FindBoundaryStrengths() noexcept override = default;

  FindBoundaryStrengths(const FindBoundaryStrengths&) = delete;
  FindBoundaryStrengths(FindBoundaryStrengths&&) noexcept = delete;

  FindBoundaryStrengths& operator=(const FindBoundaryStrengths&) = delete;
  FindBoundaryStrengths& operator=(FindBoundaryStrengths&&) noexcept = delete;

  // Parameter Keys
  static inline constexpr StringLiteral k_Loading_Key = "Loading";
  static inline constexpr StringLiteral k_SurfaceMeshFaceLabelsArrayPath_Key = "SurfaceMeshFaceLabelsArrayPath";
  static inline constexpr StringLiteral k_AvgQuatsArrayPath_Key = "AvgQuatsArrayPath";
  static inline constexpr StringLiteral k_FeaturePhasesArrayPath_Key = "FeaturePhasesArrayPath";
  static inline constexpr StringLiteral k_CrystalStructuresArrayPath_Key = "CrystalStructuresArrayPath";
  static inline constexpr StringLiteral k_SurfaceMeshF1sArrayName_Key = "SurfaceMeshF1sArrayName";
  static inline constexpr StringLiteral k_SurfaceMeshF1sptsArrayName_Key = "SurfaceMeshF1sptsArrayName";
  static inline constexpr StringLiteral k_SurfaceMeshF7sArrayName_Key = "SurfaceMeshF7sArrayName";
  static inline constexpr StringLiteral k_SurfaceMeshmPrimesArrayName_Key = "SurfaceMeshmPrimesArrayName";

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

COMPLEX_DEF_FILTER_TRAITS(complex, FindBoundaryStrengths, "1fe81a18-79fd-5236-8f20-b0a93d2de9a4");
