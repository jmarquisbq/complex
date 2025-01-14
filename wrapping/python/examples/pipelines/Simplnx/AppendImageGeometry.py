import simplnx as nx

import itkimageprocessing as cxitk
import orientationanalysis as cxor
import simplnx_test_dirs as nxtest

import numpy as np

# Create a Data Structure
data_structure = nx.DataStructure()

# Filter 1
# Instantiate Filter
nx_filter = nx.CreateImageGeometryFilter()
# Execute Filter with Parameters
result = nx_filter.execute(
    data_structure=data_structure,
    cell_data_name="Cell Data",
    dimensions=[60, 80, 100],
    output_image_geometry_path=nx.DataPath("[Image Geometry]"),
    origin=[100.0, 100.0, 0.0],
    spacing=[1.0, 1.0, 1.0]
)
nxtest.check_filter_result(nx_filter, result)

# Filter 2
# Instantiate Filter
nx_filter = nx.ReadTextDataArrayFilter()
# Execute Filter with Parameters
result = nx_filter.execute(
    data_structure=data_structure,
    set_tuple_dimensions=True,
    data_format="",
    delimiter_index=0,
    input_file=nxtest.get_data_directory() / "ASCIIData/ConfidenceIndex.csv",
    number_comp=1,
    skip_line_count=0,
    number_tuples=[[480000.0]],
    output_data_array_path=nx.DataPath("Confidence Index"),
    scalar_type_index=nx.NumericType.float32
)
nxtest.check_filter_result(nx_filter, result)

# Filter 3
# Instantiate Filter
nx_filter = nx.ReadTextDataArrayFilter()
# Execute Filter with Parameters
result = nx_filter.execute(
    data_structure=data_structure,
    set_tuple_dimensions=True,
    data_format="",
    delimiter_index=0,
    input_file=nxtest.get_data_directory() / "ASCIIData/FeatureIds.csv",
    number_comp=1,
    skip_line_count=0,
    number_tuples=[[480000.0]],
    output_data_array_path=nx.DataPath("[Image Geometry]/Cell Data/FeatureIds"),
    scalar_type_index=nx.NumericType.int32
)
nxtest.check_filter_result(nx_filter, result)


# Filter 4
# Instantiate Filter
nx_filter = nx.ReadTextDataArrayFilter()
# Execute Filter with Parameters
result = nx_filter.execute(
    data_structure=data_structure,
    set_tuple_dimensions=True,
    data_format="",
    delimiter_index=0,
    input_file=nxtest.get_data_directory() / "ASCIIData/ImageQuality.csv",
    number_comp=1,
    skip_line_count=0,
    number_tuples=[[480000.0]],
    output_data_array_path=nx.DataPath("[Image Geometry]/Cell Data/Image Quality"),
    scalar_type_index=nx.NumericType.float32
)
nxtest.check_filter_result(nx_filter, result)
# Filter 5
# Instantiate Filter
nx_filter = nx.ReadTextDataArrayFilter()
# Execute Filter with Parameters
result = nx_filter.execute(
    data_structure=data_structure,
    set_tuple_dimensions=True,
    data_format="",
    delimiter_index=0,
    input_file=nxtest.get_data_directory() / "ASCIIData/IPFColor.csv",
    number_comp=3,
    skip_line_count=0,
    number_tuples=[[480000.0]],
    output_data_array_path=nx.DataPath("[Image Geometry]/Cell Data/IPFColors"),
    scalar_type_index=nx.NumericType.uint8
)
nxtest.check_filter_result(nx_filter, result)
# Filter 6
# Instantiate Filter
nx_filter = nx.CropImageGeometryFilter()
# Execute Filter with Parameters
result = nx_filter.execute(
    data_structure=data_structure,
    output_image_geometry_path=nx.DataPath("CroppedBottomHalf"),
    max_voxel=[59, 79, 50],
    min_voxel=[0, 0, 0],
    remove_original_geometry=False,
    renumber_features=False,
    input_image_geometry_path=nx.DataPath("[Image Geometry]")
    #update_origin=False
    # cell_feature_attribute_matrix: DataPath = ...,  # Not currently part of the code
    # feature_ids: DataPath = ...,  # Not currently part of the code
)
nxtest.check_filter_result(nx_filter, result)


# Filter 7
# Instantiate Filter
nx_filter = nx.CropImageGeometryFilter()
# Execute Filter with Parameters
result = nx_filter.execute(
    data_structure=data_structure,
    output_image_geometry_path=nx.DataPath("CroppedTopHalf"),
    max_voxel=[59, 79, 99],
    min_voxel=[0, 0, 51],
    remove_original_geometry=False,
    renumber_features=False,
    input_image_geometry_path=nx.DataPath("[Image Geometry]")
    # update_origin=True
    # cell_feature_attribute_matrix: DataPath = ...,  # Not currently part of the code
    # feature_ids: DataPath = ...,  # Not currently part of the code
)
nxtest.check_filter_result(nx_filter, result)

# Filter 8
# Instantiate Filter
nx_filter = nx.AppendImageGeometryFilter()
# Execute Filter with Parameters
result = nx_filter.execute(
    data_structure=data_structure,
    check_resolution=True,
    destination_image_geometry_path=nx.DataPath("CroppedBottomHalf"),
    input_image_geometries_paths=[nx.DataPath("CroppedTopHalf")],
    output_image_geometry_path=nx.DataPath("AppendedImageGeom"),
    save_as_new_geometry=True
)
nxtest.check_filter_result(nx_filter, result)

# Filter 9
# Instantiate Filter
nx_filter = nx.WriteDREAM3DFilter()
# Set Output File Path
output_file_path = nxtest.get_data_directory() / "Output/Examples/AppendImageGeometry.dream3d"
# Execute Filter with Parameters
result = nx_filter.execute(
    data_structure=data_structure,
    export_file_path=output_file_path,
    write_xdmf_file=True
)
nxtest.check_filter_result(nx_filter, result)

# *****************************************************************************
# THIS SECTION IS ONLY HERE FOR CLEANING UP THE CI Machines
# If you are using this code, you should COMMENT out the next line
nxtest.cleanup_test_file(output_file_path)
# *****************************************************************************


print("===> Pipeline Complete")
