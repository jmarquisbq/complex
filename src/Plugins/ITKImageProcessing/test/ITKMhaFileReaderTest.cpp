#include <catch2/catch.hpp>

#include "ITKImageProcessing/Filters/ITKImageReaderFilter.hpp"
#include "ITKImageProcessing/Filters/ITKMhaFileReaderFilter.hpp"
#include "ITKImageProcessing/ITKImageProcessing_test_dirs.hpp"

#include "simplnx/UnitTest/UnitTestCommon.hpp"

#include <filesystem>

namespace fs = std::filesystem;

using namespace nx::core;

TEST_CASE("ITKImageProcessing::ITKMhaFileReaderFilter: Read 2D & 3D Image Data", "[ITKImageProcessing][ITKMhaFileReaderFilter]")
{
  const nx::core::UnitTest::TestFileSentinel testDataSentinel(nx::core::unit_test::k_CMakeExecutable, nx::core::unit_test::k_TestFilesDir, "ITKMhaFileReaderTest_v3.tar.gz", "ITKMhaFileReaderTest_v3");

  // Load plugins (this is needed because ITKMhaFileReaderFilter needs access to the SimplnxCore plugin)
  Application::GetOrCreateInstance()->loadPlugins(unit_test::k_BuildDir.view(), true);

  // Test reading 2D & 3D image data
  const fs::path exemplaryFilePath = fs::path(unit_test::k_TestFilesDir.view()) / "ITKMhaFileReaderTest_v3/ExemplarySmallIN100.dream3d";
  fs::path filePath;
  std::string exemplaryGeomName;
  SECTION("Test 2D Image Data")
  {
    filePath = fs::path(unit_test::k_TestFilesDir.view()) / "ITKMhaFileReaderTest_v3/SmallIN100_073.mha";
    exemplaryGeomName = "ExemplarySmallIN100_073";
  }
  SECTION("Test 3D Image Data")
  {
    filePath = fs::path(unit_test::k_TestFilesDir.view()) / "ITKMhaFileReaderTest_v3/SmallIN100.mha";
    exemplaryGeomName = "ExemplarySmallIN100";
  }

  const ITKMhaFileReaderFilter filter;
  DataStructure dataStructure = UnitTest::LoadDataStructure(exemplaryFilePath);
  Arguments args;

  const std::string geomName = "ImageGeom";
  const std::string amName = "Cell Data";
  const std::string arrName = "ImageData";
  const std::string tMatrixName = "TransformationMatrix";
  const DataPath geomPath{{geomName}};
  const DataPath arrayPath{{geomName, amName, arrName}};
  const DataPath tMatrixPath{{geomName, tMatrixName}};

  const std::string exemplaryAMName = "Cell Data";
  const std::string exemplaryArrName = "ImageData";
  const std::string exemplaryTMatrixName = "TransformationMatrix";
  const DataPath exemplaryGeomPath{{exemplaryGeomName}};
  const DataPath exemplaryArrayPath{{exemplaryGeomName, exemplaryAMName, exemplaryArrName}};
  const DataPath exemplaryTMatrixPath{{exemplaryGeomName, exemplaryTMatrixName}};

  args.insertOrAssign(ITKImageReaderFilter::k_FileName_Key, filePath);
  args.insertOrAssign(ITKImageReaderFilter::k_ImageGeometryPath_Key, geomPath);
  args.insertOrAssign(ITKImageReaderFilter::k_CellDataName_Key, amName);
  args.insertOrAssign(ITKImageReaderFilter::k_ImageDataArrayPath_Key, arrName);
  args.insertOrAssign(ITKMhaFileReaderFilter::k_ApplyImageTransformation, true);
  args.insertOrAssign(ITKMhaFileReaderFilter::k_SaveImageTransformationAsArray, true);
  args.insertOrAssign(ITKMhaFileReaderFilter::k_TransformationMatrixDataArrayPathKey, tMatrixPath);

  auto preflightResult = filter.preflight(dataStructure, args);
  SIMPLNX_RESULT_REQUIRE_VALID(preflightResult.outputActions)

  auto executeResult = filter.execute(dataStructure, args);
  SIMPLNX_RESULT_REQUIRE_VALID(executeResult.result)

  const auto* imageGeomPtr = dataStructure.getDataAs<ImageGeom>(geomPath);
  REQUIRE(imageGeomPtr != nullptr);

  const auto* exemplaryImageGeomPtr = dataStructure.getDataAs<ImageGeom>(exemplaryGeomPath);
  REQUIRE(exemplaryImageGeomPtr != nullptr);

  REQUIRE(imageGeomPtr->getDimensions() == exemplaryImageGeomPtr->getDimensions());

  auto calcOrigin = imageGeomPtr->getOrigin();
  auto exemplarOrigin = exemplaryImageGeomPtr->getOrigin();

  REQUIRE(imageGeomPtr->getOrigin() == exemplaryImageGeomPtr->getOrigin());
  REQUIRE(imageGeomPtr->getSpacing() == exemplaryImageGeomPtr->getSpacing());

  const auto* dataArrayPtr = dataStructure.getDataAs<Float32Array>(arrayPath);
  REQUIRE(dataArrayPtr != nullptr);

  const auto* exemplaryDataArrayPtr = dataStructure.getDataAs<Float32Array>(exemplaryArrayPath);
  REQUIRE(exemplaryDataArrayPtr != nullptr);

  REQUIRE(dataArrayPtr->getTupleShape() == exemplaryDataArrayPtr->getTupleShape());
  REQUIRE(dataArrayPtr->getComponentShape() == exemplaryDataArrayPtr->getComponentShape());
  REQUIRE(std::equal(dataArrayPtr->begin(), dataArrayPtr->end(), exemplaryDataArrayPtr->begin(), exemplaryDataArrayPtr->end()));

  const auto* tMatrixPtr = dataStructure.getDataAs<Float32Array>(tMatrixPath);
  REQUIRE(tMatrixPtr != nullptr);

  const auto* exemplaryTMatrixPtr = dataStructure.getDataAs<Float32Array>(exemplaryTMatrixPath);
  REQUIRE(exemplaryTMatrixPtr != nullptr);

  REQUIRE(tMatrixPtr->getTupleShape() == exemplaryTMatrixPtr->getTupleShape());
  REQUIRE(tMatrixPtr->getComponentShape() == exemplaryTMatrixPtr->getComponentShape());
  REQUIRE(std::equal(tMatrixPtr->begin(), tMatrixPtr->end(), exemplaryTMatrixPtr->begin(), exemplaryTMatrixPtr->end()));
}
