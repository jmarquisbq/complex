#include <catch2/catch.hpp>

#include "ITKImageProcessing/Common/sitkCommon.hpp"
#include "ITKImageProcessing/Filters/ITKBinaryDilateImageFilter.hpp"
#include "ITKImageProcessing/ITKImageProcessing_test_dirs.hpp"
#include "ITKTestBase.hpp"

#include "simplnx/Parameters/BoolParameter.hpp"
#include "simplnx/Parameters/ChoicesParameter.hpp"
#include "simplnx/Parameters/DataObjectNameParameter.hpp"
#include "simplnx/Parameters/NumberParameter.hpp"
#include "simplnx/Parameters/VectorParameter.hpp"
#include "simplnx/UnitTest/UnitTestCommon.hpp"

#include <filesystem>
namespace fs = std::filesystem;

using namespace nx::core;
using namespace nx::core::Constants;
using namespace nx::core::UnitTest;

TEST_CASE("ITKImageProcessing::ITKBinaryDilateImageFilter(BinaryDilate)", "[ITKImageProcessing][ITKBinaryDilateImage][BinaryDilate]")
{
  DataStructure dataStructure;
  const ITKBinaryDilateImageFilter filter;

  const DataPath inputGeometryPath({ITKTestBase::k_ImageGeometryPath});
  const DataPath cellDataPath = inputGeometryPath.createChildPath(ITKTestBase::k_ImageCellDataName);
  const DataPath inputDataPath = cellDataPath.createChildPath(ITKTestBase::k_InputDataName);
  const DataObjectNameParameter::ValueType outputArrayName = ITKTestBase::k_OutputDataPath;

  { // Start Image Comparison Scope
    const fs::path inputFilePath = fs::path(unit_test::k_SourceDir.view()) / unit_test::k_DataDir.view() / "JSONFilters" / "Input/STAPLE1.png";
    Result<> imageReadResult = ITKTestBase::ReadImage(dataStructure, inputFilePath, inputGeometryPath, ITKTestBase::k_ImageCellDataName, ITKTestBase::k_InputDataName);
    SIMPLNX_RESULT_REQUIRE_VALID(imageReadResult)
  } // End Image Comparison Scope

  Arguments args;
  args.insertOrAssign(ITKBinaryDilateImageFilter::k_InputImageGeomPath_Key, std::make_any<DataPath>(inputGeometryPath));
  args.insertOrAssign(ITKBinaryDilateImageFilter::k_InputImageDataPath_Key, std::make_any<DataPath>(inputDataPath));
  args.insertOrAssign(ITKBinaryDilateImageFilter::k_OutputImageArrayName_Key, std::make_any<DataObjectNameParameter::ValueType>(outputArrayName));
  args.insertOrAssign(ITKBinaryDilateImageFilter::k_KernelRadius_Key, std::make_any<VectorParameter<uint32>::ValueType>(std::vector<uint32>{1, 1, 1}));
  args.insertOrAssign(ITKBinaryDilateImageFilter::k_KernelType_Key, std::make_any<ChoicesParameter::ValueType>(itk::simple::sitkBall));
  args.insertOrAssign(ITKBinaryDilateImageFilter::k_ForegroundValue_Key, std::make_any<Float64Parameter::ValueType>(255.0));

  auto preflightResult = filter.preflight(dataStructure, args);
  SIMPLNX_RESULT_REQUIRE_VALID(preflightResult.outputActions)

  auto executeResult = filter.execute(dataStructure, args);
  SIMPLNX_RESULT_REQUIRE_VALID(executeResult.result)

  const std::string md5Hash = ITKTestBase::ComputeMd5Hash(dataStructure, cellDataPath.createChildPath(outputArrayName));
  REQUIRE(md5Hash == "9eef659f21dab5eb49e0f715a5d9a21b");
}

TEST_CASE("ITKImageProcessing::ITKBinaryDilateImageFilter(BinaryDilateVectorRadius)", "[ITKImageProcessing][ITKBinaryDilateImage][BinaryDilateVectorRadius]")
{
  DataStructure dataStructure;
  const ITKBinaryDilateImageFilter filter;

  const DataPath inputGeometryPath({ITKTestBase::k_ImageGeometryPath});
  const DataPath cellDataPath = inputGeometryPath.createChildPath(ITKTestBase::k_ImageCellDataName);
  const DataPath inputDataPath = cellDataPath.createChildPath(ITKTestBase::k_InputDataName);
  const DataObjectNameParameter::ValueType outputArrayName = ITKTestBase::k_OutputDataPath;

  { // Start Image Comparison Scope
    const fs::path inputFilePath = fs::path(unit_test::k_SourceDir.view()) / unit_test::k_DataDir.view() / "JSONFilters" / "Input/STAPLE1.png";
    Result<> imageReadResult = ITKTestBase::ReadImage(dataStructure, inputFilePath, inputGeometryPath, ITKTestBase::k_ImageCellDataName, ITKTestBase::k_InputDataName);
    SIMPLNX_RESULT_REQUIRE_VALID(imageReadResult)
  } // End Image Comparison Scope

  Arguments args;
  args.insertOrAssign(ITKBinaryDilateImageFilter::k_InputImageGeomPath_Key, std::make_any<DataPath>(inputGeometryPath));
  args.insertOrAssign(ITKBinaryDilateImageFilter::k_InputImageDataPath_Key, std::make_any<DataPath>(inputDataPath));
  args.insertOrAssign(ITKBinaryDilateImageFilter::k_OutputImageArrayName_Key, std::make_any<DataObjectNameParameter::ValueType>(outputArrayName));
  args.insertOrAssign(ITKBinaryDilateImageFilter::k_KernelRadius_Key, std::make_any<VectorParameter<uint32>::ValueType>(std::vector<uint32>{20, 1, 1}));
  args.insertOrAssign(ITKBinaryDilateImageFilter::k_KernelType_Key, std::make_any<ChoicesParameter::ValueType>(itk::simple::sitkBox));
  args.insertOrAssign(ITKBinaryDilateImageFilter::k_ForegroundValue_Key, std::make_any<Float64Parameter::ValueType>(255));

  auto preflightResult = filter.preflight(dataStructure, args);
  SIMPLNX_RESULT_REQUIRE_VALID(preflightResult.outputActions)

  auto executeResult = filter.execute(dataStructure, args);
  SIMPLNX_RESULT_REQUIRE_VALID(executeResult.result)

  const std::string md5Hash = ITKTestBase::ComputeMd5Hash(dataStructure, cellDataPath.createChildPath(outputArrayName));
  REQUIRE(md5Hash == "99108c735fe9727bca09ca28a42827d3");
}
