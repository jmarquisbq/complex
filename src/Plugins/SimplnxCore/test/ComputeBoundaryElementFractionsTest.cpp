#include <catch2/catch.hpp>

#include "simplnx/UnitTest/UnitTestCommon.hpp"

#include "SimplnxCore/Filters/ComputeBoundaryElementFractionsFilter.hpp"
#include "SimplnxCore/SimplnxCore_test_dirs.hpp"

using namespace nx::core;

namespace
{
const std::string k_BCFName = "Boundary Cell Fractions";

const DataPath k_FeatureDataAMPath = DataPath({Constants::k_SmallIN100, Constants::k_Grain_Data});

const DataPath k_ExemplarBCFPath = k_FeatureDataAMPath.createChildPath(" Surface Element Fractions");
const DataPath k_GeneratedBCFPath = k_FeatureDataAMPath.createChildPath(k_BCFName);
} // namespace

TEST_CASE("SimplnxCore::ComputeBoundaryElementFractionsFilter: Valid Filter Execution", "[SimplnxCore][ComputeBoundaryElementFractionsFilter]")
{
  const nx::core::UnitTest::TestFileSentinel testDataSentinel(nx::core::unit_test::k_CMakeExecutable, nx::core::unit_test::k_TestFilesDir, "6_6_find_feature_boundary_element_fractions.tar.gz",
                                                              "6_6_find_feature_boundary_element_fractions");

  DataStructure dataStructure =
      UnitTest::LoadDataStructure(fs::path(fmt::format("{}/6_6_find_feature_boundary_element_fractions/6_6_find_feature_boundary_element_fractions.dream3d", unit_test::k_TestFilesDir)));

  {
    // Instantiate the filter, a DataStructure object and an Arguments Object
    ComputeBoundaryElementFractionsFilter filter;
    Arguments args;

    // Create default Parameters for the filter.
    args.insertOrAssign(ComputeBoundaryElementFractionsFilter::k_FeatureIdsArrayPath_Key,
                        std::make_any<DataPath>(DataPath({Constants::k_SmallIN100, Constants::k_EbsdScanData, Constants::k_FeatureIds})));
    args.insertOrAssign(ComputeBoundaryElementFractionsFilter::k_BoundaryCellsArrayPath_Key, std::make_any<DataPath>(DataPath({Constants::k_SmallIN100, Constants::k_EbsdScanData, "BoundaryCells"})));
    args.insertOrAssign(ComputeBoundaryElementFractionsFilter::k_FeatureDataAMPath_Key, std::make_any<DataPath>(k_FeatureDataAMPath));
    args.insertOrAssign(ComputeBoundaryElementFractionsFilter::k_BoundaryCellFractionsArrayName_Key, std::make_any<std::string>(::k_BCFName));

    // Preflight the filter and check result
    auto preflightResult = filter.preflight(dataStructure, args);
    REQUIRE(preflightResult.outputActions.valid());

    // Execute the filter and check the result
    auto executeResult = filter.execute(dataStructure, args);
    REQUIRE(executeResult.result.valid());
  }

  UnitTest::CompareArrays<float32>(dataStructure, k_ExemplarBCFPath, k_GeneratedBCFPath);
}
