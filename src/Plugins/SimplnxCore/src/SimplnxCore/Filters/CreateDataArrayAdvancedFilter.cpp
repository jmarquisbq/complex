#include "CreateDataArrayAdvancedFilter.hpp"

#include "SimplnxCore/Filters/Algorithms/InitializeData.hpp"

#include "simplnx/Common/TypesUtility.hpp"
#include "simplnx/Filter/Actions/CreateArrayAction.hpp"
#include "simplnx/Parameters/ArrayCreationParameter.hpp"
#include "simplnx/Parameters/BoolParameter.hpp"
#include "simplnx/Parameters/DataObjectNameParameter.hpp"
#include "simplnx/Parameters/DataStoreFormatParameter.hpp"
#include "simplnx/Parameters/DynamicTableParameter.hpp"
#include "simplnx/Parameters/NumberParameter.hpp"
#include "simplnx/Parameters/NumericTypeParameter.hpp"
#include "simplnx/Parameters/StringParameter.hpp"
#include "simplnx/Utilities/DataArrayUtilities.hpp"
#include "simplnx/Utilities/FilterUtilities.hpp"
#include "simplnx/Utilities/SIMPLConversion.hpp"

#include <stdexcept>

using namespace nx::core;

namespace
{
struct CreateAndInitArrayFunctor
{
  template <class T>
  void operator()(IDataArray* iDataArray, const std::string& initValue)
  {
    Result<T> result = ConvertTo<T>::convert(initValue);

    auto* dataStore = iDataArray->template getIDataStoreAs<AbstractDataStore<T>>();
    dataStore->fill(result.value());
  }
};
} // namespace

namespace nx::core
{
//------------------------------------------------------------------------------
std::string CreateDataArrayAdvancedFilter::name() const
{
  return FilterTraits<CreateDataArrayAdvancedFilter>::name;
}

//------------------------------------------------------------------------------
std::string CreateDataArrayAdvancedFilter::className() const
{
  return FilterTraits<CreateDataArrayAdvancedFilter>::className;
}

//------------------------------------------------------------------------------
Uuid CreateDataArrayAdvancedFilter::uuid() const
{
  return FilterTraits<CreateDataArrayAdvancedFilter>::uuid;
}

//------------------------------------------------------------------------------
std::string CreateDataArrayAdvancedFilter::humanName() const
{
  return "Create Data Array (Advanced)";
}

//------------------------------------------------------------------------------
std::vector<std::string> CreateDataArrayAdvancedFilter::defaultTags() const
{
  return {className(), "Create", "Data Structure", "Data Array", "Initialize", "Make"};
}

//------------------------------------------------------------------------------
Parameters CreateDataArrayAdvancedFilter::parameters() const
{
  Parameters params;

  params.insertSeparator(Parameters::Separator{"Output Data Array"});
  params.insert(std::make_unique<ArrayCreationParameter>(k_DataPath_Key, "Created Array", "Array storing the data", DataPath({"Data"})));
  params.insert(std::make_unique<NumericTypeParameter>(k_NumericType_Key, "Output Numeric Type", "Numeric Type of data to create", NumericType::int32));
  params.insert(std::make_unique<DataStoreFormatParameter>(k_DataFormat_Key, "Data Format",
                                                           "This value will specify which data format is used by the array's data store. An empty string results in in-memory data store.", ""));

  params.insertSeparator(Parameters::Separator{"Tuple Dimensions"});
  params.insertLinkableParameter(std::make_unique<BoolParameter>(
      k_AdvancedOptions_Key, "Set Tuple Dimensions [not required if creating inside an existing Attribute Matrix]",
      "This allows the user to set the tuple dimensions directly rather than just inheriting them. This option is NOT required if you are creating the Data Array in an Attribute Matrix", true));

  {
    DynamicTableInfo tableInfo;
    tableInfo.setRowsInfo(DynamicTableInfo::StaticVectorInfo(1));
    tableInfo.setColsInfo(DynamicTableInfo::DynamicVectorInfo(1, "TUPLE DIM {}"));
    const DynamicTableInfo::TableDataType defaultTable{{1.0F}};
    params.insert(std::make_unique<DynamicTableParameter>(k_TupleDims_Key, "Data Array Tuple Dimensions (Slowest to Fastest Dimensions)",
                                                          "Slowest to Fastest Dimensions. Note this might be opposite displayed by an image geometry.", defaultTable, tableInfo));
  }

  params.insertSeparator(Parameters::Separator{"Component Dimensions"});
  {
    DynamicTableInfo tableInfo;
    tableInfo.setRowsInfo(DynamicTableInfo::StaticVectorInfo(1));
    tableInfo.setColsInfo(DynamicTableInfo::DynamicVectorInfo(1, "COMP DIM {}"));
    const DynamicTableInfo::TableDataType defaultTable{{1.0F}};
    params.insert(std::make_unique<DynamicTableParameter>(k_CompDims_Key, "Data Array Component Dimensions (Slowest to Fastest Dimensions)", "Slowest to Fastest Component Dimensions.", defaultTable,
                                                          tableInfo));
  }

  params.insertSeparator(Parameters::Separator{"Initialization Options"});
  params.insertLinkableParameter(std::make_unique<ChoicesParameter>(k_InitType_Key, "Initialization Type", "Method for determining the what values of the data in the array should be initialized to",
                                                                    static_cast<ChoicesParameter::ValueType>(0),
                                                                    ChoicesParameter::Choices{"Fill Value", "Incremental", "Random", "Random With Range"})); // sequence dependent DO NOT REORDER

  params.insert(std::make_unique<StringParameter>(k_InitValue_Key, "Fill Values [Seperated with ;]",
                                                  "Specify values for each component. Ex: A 3-component array would be 6;8;12 and every tuple would have these same component values", "1;1;1"));

  params.insert(std::make_unique<StringParameter>(
      k_StartingFillValue_Key, "Starting Value [Seperated with ;]",
      "The value to start incrementing from. Ex: 6;8;12 would increment a 3-component array starting at 6 for the first component, 8 for the 2nd, and 12 for the 3rd.", "0;1;2"));
  params.insert(std::make_unique<ChoicesParameter>(k_StepOperation_Key, "Step Operation", "The type of step operation to perform", static_cast<ChoicesParameter::ValueType>(0),
                                                   ChoicesParameter::Choices{"Addition", "Subtraction"}));
  params.insert(std::make_unique<StringParameter>(k_StepValue_Key, "Step Value [Seperated with ;]", "The number to add/subtract the fill value by", "1;1;1"));

  params.insert(std::make_unique<BoolParameter>(k_UseSeed_Key, "Use Seed for Random Generation", "When true, the Seed Value will be used to seed the generator", false));
  params.insert(std::make_unique<NumberParameter<uint64>>(k_SeedValue_Key, "Seed Value", "The seed fed into the random generator", std::mt19937::default_seed));
  params.insert(std::make_unique<DataObjectNameParameter>(k_SeedArrayName_Key, "Stored Seed Value Array Name", "Name of the array holding the seed value", "InitializeDataFilter SeedValue"));
  params.insert(std::make_unique<BoolParameter>(k_StandardizeSeed_Key, "Use the Same Seed for Each Component",
                                                "When true the same seed will be used for each component's generator in a multi-component array", false));

  params.insert(
      std::make_unique<StringParameter>(k_InitStartRange_Key, "Initialization Start Range [Seperated with ;]", "[Inclusive] The lower bound initialization range for random values", "0;0;0"));
  params.insert(std::make_unique<StringParameter>(k_InitEndRange_Key, "Initialization End Range [Seperated with ;]", "[Inclusive] The upper bound initialization range for random values", "1;1;1"));

  // Associate the Linkable Parameter(s) to the children parameters that they control
  params.linkParameters(k_AdvancedOptions_Key, k_TupleDims_Key, true);

  // Associate the Linkable Parameter(s) to the children parameters that they control
  /* Using Fill Value */
  params.linkParameters(k_InitType_Key, k_InitValue_Key, static_cast<ChoicesParameter::ValueType>(0));

  /* Using Incremental */
  params.linkParameters(k_InitType_Key, k_StartingFillValue_Key, static_cast<ChoicesParameter::ValueType>(1));
  params.linkParameters(k_InitType_Key, k_StepOperation_Key, static_cast<ChoicesParameter::ValueType>(1));
  params.linkParameters(k_InitType_Key, k_StepValue_Key, static_cast<ChoicesParameter::ValueType>(1));

  /* Random - Using Random */
  params.linkParameters(k_InitType_Key, k_UseSeed_Key, static_cast<ChoicesParameter::ValueType>(2));
  params.linkParameters(k_InitType_Key, k_SeedValue_Key, static_cast<ChoicesParameter::ValueType>(2));
  params.linkParameters(k_InitType_Key, k_SeedArrayName_Key, static_cast<ChoicesParameter::ValueType>(2));
  params.linkParameters(k_InitType_Key, k_StandardizeSeed_Key, static_cast<ChoicesParameter::ValueType>(2));

  /* Random - Using Random With Range */
  params.linkParameters(k_InitType_Key, k_UseSeed_Key, static_cast<ChoicesParameter::ValueType>(3));
  params.linkParameters(k_InitType_Key, k_SeedValue_Key, static_cast<ChoicesParameter::ValueType>(3));
  params.linkParameters(k_InitType_Key, k_SeedArrayName_Key, static_cast<ChoicesParameter::ValueType>(3));
  params.linkParameters(k_InitType_Key, k_StandardizeSeed_Key, static_cast<ChoicesParameter::ValueType>(3));
  params.linkParameters(k_InitType_Key, k_InitStartRange_Key, static_cast<ChoicesParameter::ValueType>(3));
  params.linkParameters(k_InitType_Key, k_InitEndRange_Key, static_cast<ChoicesParameter::ValueType>(3));
  return params;
}

//------------------------------------------------------------------------------
IFilter::VersionType CreateDataArrayAdvancedFilter::parametersVersion() const
{
  return 1;
}

//------------------------------------------------------------------------------
IFilter::UniquePointer CreateDataArrayAdvancedFilter::clone() const
{
  return std::make_unique<CreateDataArrayAdvancedFilter>();
}

//------------------------------------------------------------------------------
IFilter::PreflightResult CreateDataArrayAdvancedFilter::preflightImpl(const DataStructure& dataStructure, const Arguments& filterArgs, const MessageHandler& messageHandler,
                                                                      const std::atomic_bool& shouldCancel) const
{
  auto useDims = filterArgs.value<bool>(k_AdvancedOptions_Key);
  auto numericType = filterArgs.value<NumericType>(k_NumericType_Key);
  auto compDimsData = filterArgs.value<DynamicTableParameter::ValueType>(k_CompDims_Key);
  auto dataArrayPath = filterArgs.value<DataPath>(k_DataPath_Key);
  auto tableData = filterArgs.value<DynamicTableParameter::ValueType>(k_TupleDims_Key);
  auto dataFormat = filterArgs.value<std::string>(k_DataFormat_Key);
  auto initFillValue = filterArgs.value<std::string>(k_InitValue_Key);
  auto initIncFillValue = filterArgs.value<std::string>(k_StartingFillValue_Key);
  auto stepValue = filterArgs.value<std::string>(k_StepValue_Key);
  auto stepOperation = filterArgs.value<ChoicesParameter::ValueType>(k_StepOperation_Key);
  auto initStartRange = filterArgs.value<std::string>(k_InitStartRange_Key);
  auto initEndRange = filterArgs.value<std::string>(k_InitEndRange_Key);
  auto standardizeSeed = filterArgs.value<bool>(k_StandardizeSeed_Key);

  nx::core::Result<OutputActions> resultOutputActions;

  std::vector<usize> compDims(compDimsData[0].size());
  std::transform(compDimsData[0].begin(), compDimsData[0].end(), compDims.begin(), [](double val) { return static_cast<usize>(val); });
  usize numComponents = std::accumulate(compDims.begin(), compDims.end(), static_cast<usize>(1), std::multiplies<>());
  if(numComponents <= 0)
  {
    std::string compDimsStr = std::accumulate(compDims.begin() + 1ULL, compDims.end(), std::to_string(compDims[0]), [](const std::string& a, int b) { return a + " x " + std::to_string(b); });
    return MakePreflightErrorResult(
        -78601,
        fmt::format("The chosen component dimensions ({}) results in 0 total components.  Please choose component dimensions that result in a positive number of total components.", compDimsStr));
  }

  std::vector<usize> tupleDims = {};

  auto* parentAM = dataStructure.getDataAs<AttributeMatrix>(dataArrayPath.getParent());
  if(parentAM == nullptr)
  {
    if(!useDims)
    {
      return MakePreflightErrorResult(
          -78602, fmt::format("The DataArray to be created '{}'is not within an AttributeMatrix, so the dimensions cannot be determined implicitly. Check Set Tuple Dimensions to set the dimensions",
                              dataArrayPath.toString()));
    }
    else
    {
      const auto& rowData = tableData.at(0);
      tupleDims.reserve(rowData.size());
      for(auto floatValue : rowData)
      {
        if(floatValue == 0)
        {
          return MakePreflightErrorResult(-78603, "Tuple dimension cannot be zero");
        }

        tupleDims.push_back(static_cast<usize>(floatValue));
      }
    }
  }
  else
  {
    tupleDims = parentAM->getShape();
    if(useDims)
    {
      resultOutputActions.warnings().push_back(
          Warning{-78604, "You checked Set Tuple Dimensions, but selected a DataPath that has an Attribute Matrix as the parent. The Attribute Matrix tuples will override your "
                          "custom dimensions. It is recommended to uncheck Set Tuple Dimensions for the sake of clarity."});
    }
  }

  usize numTuples = std::accumulate(tupleDims.begin(), tupleDims.end(), static_cast<usize>(1), std::multiplies<>());

  auto arrayDataType = ConvertNumericTypeToDataType(numericType);
  auto action = std::make_unique<CreateArrayAction>(ConvertNumericTypeToDataType(numericType), tupleDims, compDims, dataArrayPath, dataFormat);

  resultOutputActions.value().appendAction(std::move(action));

  auto seedArrayNameValue = filterArgs.value<std::string>(k_SeedArrayName_Key);
  auto initializeTypeValue = static_cast<InitializeType>(filterArgs.value<uint64>(k_InitType_Key));

  //  nx::core::Result<OutputActions> resultOutputActions;
  std::vector<PreflightValue> preflightUpdatedValues;

  switch(initializeTypeValue)
  {
  case InitializeType::FillValue: {
    auto result = ExecuteDataFunction(::ValidateMultiInputFunctor{}, arrayDataType, numComponents, filterArgs.value<std::string>(k_InitValue_Key), 1);
    if(result.outputActions.invalid())
    {
      return {MergeResults(result.outputActions, std::move(resultOutputActions)), std::move(preflightUpdatedValues)};
    }

    // Create the Fill Values preflight updated values
    CreateFillPreflightVals(initFillValue, numComponents, preflightUpdatedValues);

    break;
  }
  case InitializeType::Incremental: {
    auto result = ExecuteDataFunction(::ValidateMultiInputFunctor{}, arrayDataType, numComponents, filterArgs.value<std::string>(k_StartingFillValue_Key), 1);
    if(result.outputActions.invalid())
    {
      return {MergeResults(result.outputActions, std::move(resultOutputActions)), std::move(preflightUpdatedValues)};
    }

    result = ExecuteDataFunction(::ValidateMultiInputFunctor{}, arrayDataType, numComponents, filterArgs.value<std::string>(k_StepValue_Key), 1);
    if(result.outputActions.invalid())
    {
      return {MergeResults(result.outputActions, std::move(resultOutputActions)), std::move(preflightUpdatedValues)};
    }

    // Create the Incremental/Decremental Values preflight updated values
    CreateIncrementalPreflightVals(initIncFillValue, stepOperation, stepValue, numTuples, numComponents, preflightUpdatedValues);

    break;
  }
  case InitializeType::RangedRandom: {
    auto result = ExecuteDataFunction(::ValidateMultiInputFunctor{}, arrayDataType, numComponents, filterArgs.value<std::string>(k_InitStartRange_Key), 1);
    if(result.outputActions.invalid())
    {
      return {MergeResults(result.outputActions, std::move(resultOutputActions)), std::move(preflightUpdatedValues)};
    }

    result = ExecuteDataFunction(::ValidateMultiInputFunctor{}, arrayDataType, numComponents, filterArgs.value<std::string>(k_InitEndRange_Key), 1);
    if(result.outputActions.invalid())
    {
      return {MergeResults(result.outputActions, std::move(resultOutputActions)), std::move(preflightUpdatedValues)};
    }

    [[fallthrough]];
  }
  case InitializeType::Random: {
    auto createAction = std::make_unique<CreateArrayAction>(DataType::uint64, std::vector<usize>{1}, std::vector<usize>{1}, DataPath({seedArrayNameValue}));
    resultOutputActions.value().appendAction(std::move(createAction));

    // Create the Random Values preflight updated values
    CreateRandomPreflightVals(standardizeSeed, initializeTypeValue, initStartRange, initEndRange, numTuples, numComponents, preflightUpdatedValues);

    break;
  }
  }

  return {std::move(resultOutputActions), std::move(preflightUpdatedValues)};
}

//------------------------------------------------------------------------------
Result<> CreateDataArrayAdvancedFilter::executeImpl(DataStructure& dataStructure, const Arguments& filterArgs, const PipelineFilter* pipelineNode, const MessageHandler& messageHandler,
                                                    const std::atomic_bool& shouldCancel) const
{
  auto path = filterArgs.value<DataPath>(k_DataPath_Key);

  ExecuteNeighborFunction(CreateAndInitArrayFunctor{}, ConvertNumericTypeToDataType(filterArgs.value<NumericType>(k_NumericType_Key)), dataStructure.getDataAs<IDataArray>(path), "0");

  auto initType = static_cast<InitializeType>(filterArgs.value<uint64>(k_InitType_Key));

  auto seed = filterArgs.value<std::mt19937_64::result_type>(k_SeedValue_Key);
  if(!filterArgs.value<bool>(k_UseSeed_Key))
  {
    seed = static_cast<std::mt19937_64::result_type>(std::chrono::steady_clock::now().time_since_epoch().count());
  }

  if(initType == InitializeType::Random || initType == InitializeType::RangedRandom)
  {
    // Store Seed Value in Top Level Array
    dataStructure.getDataRefAs<UInt64Array>(DataPath({filterArgs.value<std::string>(k_SeedArrayName_Key)}))[0] = seed;
  }

  InitializeDataInputValues inputValues;
  inputValues.InputArrayPath = filterArgs.value<DataPath>(k_DataPath_Key);
  inputValues.initType = initType;
  inputValues.stepType = static_cast<StepType>(filterArgs.value<uint64>(k_StepOperation_Key));
  inputValues.stringValues = StringUtilities::split(StringUtilities::trimmed(filterArgs.value<std::string>(k_InitValue_Key)), k_DelimiterChar);
  inputValues.startValues = StringUtilities::split(StringUtilities::trimmed(filterArgs.value<std::string>(k_StartingFillValue_Key)), k_DelimiterChar);
  inputValues.stepValues = StringUtilities::split(StringUtilities::trimmed(filterArgs.value<std::string>(k_StepValue_Key)), k_DelimiterChar);
  inputValues.seed = seed;
  inputValues.randBegin = StringUtilities::split(StringUtilities::trimmed(filterArgs.value<std::string>(k_InitStartRange_Key)), k_DelimiterChar);
  inputValues.randEnd = StringUtilities::split(StringUtilities::trimmed(filterArgs.value<std::string>(k_InitEndRange_Key)), k_DelimiterChar);
  inputValues.standardizeSeed = filterArgs.value<bool>(k_StandardizeSeed_Key);

  return InitializeData(dataStructure, messageHandler, shouldCancel, &inputValues)();
}

namespace
{
namespace SIMPL
{
constexpr StringLiteral k_ScalarTypeKey = "ScalarType";
constexpr StringLiteral k_NumberOfComponentsKey = "NumberOfComponents";
constexpr StringLiteral k_InitializationValueKey = "InitializationValue";
constexpr StringLiteral k_NewArrayKey = "NewArray";
} // namespace SIMPL
} // namespace

Result<Arguments> CreateDataArrayAdvancedFilter::FromSIMPLJson(const nlohmann::json& json)
{
  Arguments args = CreateDataArrayAdvancedFilter().getDefaultArguments();

  args.insertOrAssign(k_AdvancedOptions_Key, false);

  std::vector<Result<>> results;

  results.push_back(SIMPLConversion::ConvertParameter<SIMPLConversion::ScalarTypeParameterToNumericTypeConverter>(args, json, SIMPL::k_ScalarTypeKey, k_NumericType_Key));
  // Initialize Type parameter is not applicable in NX
  results.push_back(SIMPLConversion::ConvertParameter<SIMPLConversion::StringFilterParameterConverter>(args, json, SIMPL::k_InitializationValueKey, k_InitValue_Key));
  // Initialization Range parameter is not applicable in NX
  // Starting Index value parameter is not applicable in NX
  results.push_back(SIMPLConversion::ConvertParameter<SIMPLConversion::DataArrayCreationFilterParameterConverter>(args, json, SIMPL::k_NewArrayKey, k_DataPath_Key));

  Result<> conversionResult = MergeResults(std::move(results));

  return ConvertResultTo<Arguments>(std::move(conversionResult), std::move(args));
}
} // namespace nx::core
