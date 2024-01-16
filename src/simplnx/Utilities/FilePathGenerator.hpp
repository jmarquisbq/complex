#pragma once

#include "simplnx/Common/Types.hpp"
#include "simplnx/simplnx_export.hpp"

#include <string_view>
#include <utility>
#include <vector>

namespace nx::core
{
namespace FilePathGenerator
{
enum class Ordering : uint8
{
  LowToHigh = 0,
  HighToLow = 1
};

/**
 * @brief GenerateAndValidateFileList This method will generate a file list in the correct order of the files that should
 * be imported to an h5ebsd file. Also returns if there are missing files.
 * @param start Z Slice Start
 * @param end S Slice End
 * @param increment How much to increment each item
 * @param stackLowToHigh How should the file list be generated by starting at the start value and increasing or starting at the endIndex and deccreasing
 * @param inputPath Example File Name
 * @param filePrefix The start of the file name
 * @param fileSuffix the end of the file name
 * @param fileExtension The file extension (including the '.' char)
 * @param paddingDigits the number of padding digits to use when generating the integer index value
 * @return
 */
SIMPLNX_EXPORT std::pair<std::vector<std::string>, bool> GenerateAndValidateFileList(int32 start, int32 end, int32 increment, Ordering order, std::string_view inputPath, std::string_view filePrefix,
                                                                                     std::string_view fileSuffix, std::string_view fileExtension, uint32 paddingDigits, bool failFast = true);

/**
 * @brief GenerateFileList This method will generate a file list in the correct order of the files that should
 * be imported to an h5ebsd file.
 * @param start Z Slice Start
 * @param end S Slice End
 * @param increment How much to increment each item
 * @param stackLowToHigh How should the file list be generated by starting at the start value and increasing or starting at the endIndex and deccreasing
 * @param inputPath Example File Name
 * @param filePrefix The start of the file name
 * @param fileSuffix the end of the file name
 * @param fileExtension The file extension (including the '.' char)
 * @param paddingDigits the number of padding digits to use when generating the integer index value
 * @return
 */
SIMPLNX_EXPORT std::vector<std::string> GenerateFileList(int32 start, int32 end, int32 increment, Ordering order, std::string_view inputPath, std::string_view filePrefix, std::string_view fileSuffix,
                                                         std::string_view fileExtension, uint32 paddingDigits);
} // namespace FilePathGenerator
} // namespace nx::core