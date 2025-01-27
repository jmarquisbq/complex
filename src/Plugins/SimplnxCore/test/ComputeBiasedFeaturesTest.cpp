#include <catch2/catch.hpp>

#include "simplnx/UnitTest/UnitTestCommon.hpp"

#include "SimplnxCore/Filters/ComputeBiasedFeaturesFilter.hpp"
#include "SimplnxCore/SimplnxCore_test_dirs.hpp"

using namespace nx::core;

namespace
{
const std::string k_ComputedBiasedFeaturesName = "ComputedBiasedFeatures";
const std::string k_ExemplarBiasedFeaturesName = "BiasedFeatures";
const DataPath k_GeometryPath({Constants::k_SmallIN100});
const DataPath k_Geometry2DPath({Constants::k_SmallIN1002});
const DataPath k_CentroidsPath({Constants::k_SmallIN100, Constants::k_Grain_Data, Constants::k_Centroids});
const DataPath k_SurfaceFeaturesPath({Constants::k_SmallIN100, Constants::k_Grain_Data, Constants::k_SurfaceFeatures});
const DataPath k_PhasesPath({Constants::k_SmallIN100, Constants::k_Grain_Data, Constants::k_Phases});
const DataPath k_Centroids2DPath({Constants::k_SmallIN1002, Constants::k_Grain_Data, Constants::k_Centroids});
const DataPath k_SurfaceFeatures2DPath({Constants::k_SmallIN1002, Constants::k_Grain_Data, Constants::k_SurfaceFeatures});
const DataPath k_Phases2DPath({Constants::k_SmallIN1002, Constants::k_Grain_Data, Constants::k_Phases});
const DataPath k_ExemplarBiasedFeaturesPath({Constants::k_SmallIN100, Constants::k_Grain_Data, k_ExemplarBiasedFeaturesName});
const DataPath k_ComputedBiasedFeaturesPath({Constants::k_SmallIN100, Constants::k_Grain_Data, k_ComputedBiasedFeaturesName});
const DataPath k_ExemplarBiasedFeatures2DPath({Constants::k_SmallIN1002, Constants::k_Grain_Data, k_ExemplarBiasedFeaturesName});
const DataPath k_ComputedBiasedFeatures2DPath({Constants::k_SmallIN1002, Constants::k_Grain_Data, k_ComputedBiasedFeaturesName});
const DataPath k_WrongSurfaceFeaturePath({Constants::k_SmallIN100, Constants::k_EbsdScanData, Constants::k_FeatureIds});
} // namespace

TEST_CASE("SimplnxCore::ComputeBiasedFeaturesFilter: Valid filter execution", "[ComputeBiasedFeaturesFilter]")
{
  const nx::core::UnitTest::TestFileSentinel testDataSentinel(nx::core::unit_test::k_CMakeExecutable, nx::core::unit_test::k_TestFilesDir, "6_6_find_biased_features.tar.gz",
                                                              "6_6_find_biased_features.dream3d");

  // Read Exemplar DREAM3D File Filter
  auto exemplarFilePath = fs::path(fmt::format("{}/6_6_find_biased_features.dream3d", unit_test::k_TestFilesDir));
  DataStructure dataStructure = UnitTest::LoadDataStructure(exemplarFilePath);

  // Instantiate the filter, a DataStructure object and an Arguments Object
  ComputeBiasedFeaturesFilter filter;
  Arguments args;

  args.insertOrAssign(ComputeBiasedFeaturesFilter::k_CalcByPhase_Key, std::make_any<bool>(true));
  args.insertOrAssign(ComputeBiasedFeaturesFilter::k_BiasedFeaturesArrayName_Key, std::make_any<std::string>(k_ComputedBiasedFeaturesName));

  SECTION("3D")
  {
    args.insertOrAssign(ComputeBiasedFeaturesFilter::k_GeometryPath_Key, std::make_any<DataPath>(k_GeometryPath));
    args.insertOrAssign(ComputeBiasedFeaturesFilter::k_CentroidsArrayPath_Key, std::make_any<DataPath>(k_CentroidsPath));
    args.insertOrAssign(ComputeBiasedFeaturesFilter::k_SurfaceFeaturesArrayPath_Key, std::make_any<DataPath>(k_SurfaceFeaturesPath));
    args.insertOrAssign(ComputeBiasedFeaturesFilter::k_PhasesArrayPath_Key, std::make_any<DataPath>(k_PhasesPath));

    // Preflight the filter and check result
    auto preflightResult = filter.preflight(dataStructure, args);
    SIMPLNX_RESULT_REQUIRE_VALID(preflightResult.outputActions)

    // Execute the filter and check the result
    auto executeResult = filter.execute(dataStructure, args);
    SIMPLNX_RESULT_REQUIRE_VALID(executeResult.result)

    UnitTest::CompareArrays<bool>(dataStructure, k_ExemplarBiasedFeaturesPath, k_ComputedBiasedFeaturesPath);
  }
  SECTION("2D")
  {
    args.insertOrAssign(ComputeBiasedFeaturesFilter::k_GeometryPath_Key, std::make_any<DataPath>(k_Geometry2DPath));
    args.insertOrAssign(ComputeBiasedFeaturesFilter::k_CentroidsArrayPath_Key, std::make_any<DataPath>(k_Centroids2DPath));
    args.insertOrAssign(ComputeBiasedFeaturesFilter::k_SurfaceFeaturesArrayPath_Key, std::make_any<DataPath>(k_SurfaceFeatures2DPath));
    args.insertOrAssign(ComputeBiasedFeaturesFilter::k_PhasesArrayPath_Key, std::make_any<DataPath>(k_Phases2DPath));

    // Preflight the filter and check result
    auto preflightResult = filter.preflight(dataStructure, args);
    SIMPLNX_RESULT_REQUIRE_VALID(preflightResult.outputActions)

    // Execute the filter and check the result
    auto executeResult = filter.execute(dataStructure, args);
    SIMPLNX_RESULT_REQUIRE_VALID(executeResult.result)

    UnitTest::CompareArrays<bool>(dataStructure, k_ExemplarBiasedFeatures2DPath, k_ComputedBiasedFeatures2DPath);
  }
}

TEST_CASE("SimplnxCore::ComputeBiasedFeaturesFilter: Invalid filter execution", "[ComputeBiasedFeaturesFilter]")
{
  const nx::core::UnitTest::TestFileSentinel testDataSentinel(nx::core::unit_test::k_CMakeExecutable, nx::core::unit_test::k_TestFilesDir, "6_6_find_biased_features.tar.gz",
                                                              "6_6_find_biased_features.dream3d");

  // Read Exemplar DREAM3D File Filter
  auto exemplarFilePath = fs::path(fmt::format("{}/6_6_find_biased_features.dream3d", unit_test::k_TestFilesDir));
  DataStructure dataStructure = UnitTest::LoadDataStructure(exemplarFilePath);

  // Instantiate the filter, a DataStructure object and an Arguments Object
  ComputeBiasedFeaturesFilter filter;
  Arguments args;

  // Create default Parameters for the filter.
  args.insertOrAssign(ComputeBiasedFeaturesFilter::k_CalcByPhase_Key, std::make_any<bool>(true));
  args.insertOrAssign(ComputeBiasedFeaturesFilter::k_GeometryPath_Key, std::make_any<DataPath>(k_GeometryPath));
  args.insertOrAssign(ComputeBiasedFeaturesFilter::k_CentroidsArrayPath_Key, std::make_any<DataPath>(k_CentroidsPath));
  args.insertOrAssign(ComputeBiasedFeaturesFilter::k_SurfaceFeaturesArrayPath_Key, std::make_any<DataPath>(k_WrongSurfaceFeaturePath));
  args.insertOrAssign(ComputeBiasedFeaturesFilter::k_PhasesArrayPath_Key, std::make_any<DataPath>(k_PhasesPath));
  args.insertOrAssign(ComputeBiasedFeaturesFilter::k_BiasedFeaturesArrayName_Key, std::make_any<std::string>(k_ComputedBiasedFeaturesName));

  // Preflight the filter and check result
  auto preflightResult = filter.preflight(dataStructure, args);
  SIMPLNX_RESULT_REQUIRE_INVALID(preflightResult.outputActions)

  // Execute the filter and check the result
  auto executeResult = filter.execute(dataStructure, args);
  SIMPLNX_RESULT_REQUIRE_INVALID(executeResult.result)
}
