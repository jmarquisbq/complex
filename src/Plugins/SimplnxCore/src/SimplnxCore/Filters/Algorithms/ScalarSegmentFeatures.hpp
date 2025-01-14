#pragma once

#include "SimplnxCore/SimplnxCore_export.hpp"

#include "simplnx/DataStructure/DataArray.hpp"
#include "simplnx/DataStructure/DataPath.hpp"
#include "simplnx/DataStructure/DataStructure.hpp"
#include "simplnx/DataStructure/IDataArray.hpp"
#include "simplnx/Filter/IFilter.hpp"
#include "simplnx/Utilities/DataArrayUtilities.hpp"
#include "simplnx/Utilities/SegmentFeatures.hpp"

#include <random>
#include <vector>

namespace nx::core
{

struct SIMPLNXCORE_EXPORT ScalarSegmentFeaturesInputValues
{
  int ScalarTolerance = 0;
  bool UseMask;
  bool RandomizeFeatureIds;
  DataPath ImageGeometryPath;
  DataPath InputDataPath;
  DataPath MaskArrayPath;
  DataPath FeatureIdsArrayPath;
  DataPath CellFeatureAttributeMatrixPath;
  DataPath ActiveArrayPath;
};

/**
 * @brief The ScalarSegmentFeatures class
 */
class SIMPLNXCORE_EXPORT ScalarSegmentFeatures : public SegmentFeatures
{
public:
  using FeatureIdsArrayType = Int32Array;
  using GoodVoxelsArrayType = BoolArray;

  ScalarSegmentFeatures(DataStructure& dataStructure, ScalarSegmentFeaturesInputValues* inputValues, const std::atomic_bool& shouldCancel, const IFilter::MessageHandler& mesgHandler);
  ~ScalarSegmentFeatures() noexcept override;

  ScalarSegmentFeatures(const ScalarSegmentFeatures&) = delete;
  ScalarSegmentFeatures(ScalarSegmentFeatures&&) noexcept = delete;
  ScalarSegmentFeatures& operator=(const ScalarSegmentFeatures&) = delete;
  ScalarSegmentFeatures& operator=(ScalarSegmentFeatures&&) noexcept = delete;

  Result<> operator()();

protected:
  /**
   * @brief
   * @param data
   * @param args
   * @param gnum
   * @param nextSeed
   * @return int64
   */
  int64_t getSeed(int32 gnum, int64 nextSeed) const override;

  /**
   * @brief
   * @param data
   * @param args
   * @param referencePoint
   * @param neighborPoint
   * @param gnum
   * @return bool
   */
  bool determineGrouping(int64 referencePoint, int64 neighborPoint, int32 gnum) const override;

private:
  const ScalarSegmentFeaturesInputValues* m_InputValues = nullptr;
  FeatureIdsArrayType* m_FeatureIdsArray = nullptr;
  GoodVoxelsArrayType* m_GoodVoxelsArray = nullptr;
  std::shared_ptr<SegmentFeatures::CompareFunctor> m_CompareFunctor;
  std::unique_ptr<MaskCompare> m_GoodVoxels = nullptr;
};
} // namespace nx::core
