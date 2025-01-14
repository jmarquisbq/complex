from typing import List

# This is required to set matplotlib in a non-interactive mode so that
# it can be used in a DREAM3DNX pipeline that runs on a non-GUI thread
import matplotlib
matplotlib.use('Agg')

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import io
import simplnx as nx

class CalculateHistogramFilter:
  INPUT_ARRAY_PATH = 'input_array_path'
  HISTOGRAM_IMAGE_SIZE = 'histogram_image_size'
  HISTOGRAM_IMAGE_DPI = 'histogram_image_dpi'
  NUMBER_OF_BINS = 'number_of_bins'
  GENERATE_HISTOGRAM_IMAGE = 'generate_histogram_image'
  HISTOGRAM_TITLE = 'histogram_title'
  HISTOGRAM_X_LABEL = 'histogram_x_label'
  HISTOGRAM_Y_LABEL = 'histogram_y_label'
  HISTOGRAM_GEOM_PATH = 'histogram_geom_path'
  HISTOGRAM_STATS_ATTRMAT_PATH = 'histogram_stats_attrmat_path'
  HISTOGRAM_CELLATTRMAT_NAME = 'histogram_cellattrmat_name'
  HISTOGRAM_DATA_NAME = 'histogram_data_name'
  COUNTS = 'counts'
  BIN_BOUNDS = 'bin_bounds'

  def uuid(self) -> nx.Uuid:
    return nx.Uuid('5b4d8eba-ea79-4239-9e88-3d35c89b9da7')

  def human_name(self) -> str:
    return 'Matplotlib: Calculate Histogram (Python)'

  def class_name(self) -> str:
    return 'CalculateHistogramFilter'

  def name(self) -> str:
    return 'CalculateHistogramFilter'

  def default_tags(self) -> List[str]:
    return ['python']

  def clone(self):
    return CalculateHistogramFilter()

  def parameters(self) -> nx.Parameters:
    params = nx.Parameters()

    params.insert(nx.Parameters.Separator("General Parameters"))
    params.insert(nx.UInt64Parameter(CalculateHistogramFilter.NUMBER_OF_BINS, 'Number of Bins', 'The number of bins in the histogram.', 10))
    params.insert_linkable_parameter(nx.BoolParameter(CalculateHistogramFilter.GENERATE_HISTOGRAM_IMAGE, 'Generate Histogram Image', 'Determines whether or not to generate an output histogram image.', True))

    params.insert(nx.Parameters.Separator("Histogram Image Parameters"))
    params.insert(nx.VectorUInt64Parameter(CalculateHistogramFilter.HISTOGRAM_IMAGE_SIZE, 'Histogram Image Size', 'The width and height of the output histogram image.', [800, 600], ['Width', 'Height']))
    params.insert(nx.UInt64Parameter(CalculateHistogramFilter.HISTOGRAM_IMAGE_DPI, 'Histogram Image DPI (dots per inch)', 'The dots per inch of the output histogram image.', 100))
    params.insert(nx.StringParameter(CalculateHistogramFilter.HISTOGRAM_TITLE, 'Histogram Title', 'The title of the histogram.', 'Histogram'))
    params.insert(nx.StringParameter(CalculateHistogramFilter.HISTOGRAM_X_LABEL, 'Histogram X Label', 'The label shown along the X axis of the histogram.', 'Value'))
    params.insert(nx.StringParameter(CalculateHistogramFilter.HISTOGRAM_Y_LABEL, 'Histogram Y Label', 'The label shown along the Y axis of the histogram.', 'Frequency'))
    params.insert(nx.Parameters.Separator("Required Data Objects"))
    params.insert(nx.ArraySelectionParameter(CalculateHistogramFilter.INPUT_ARRAY_PATH, 'Input Array', 'The dataset that will be used to calculate the histogram.', nx.DataPath(), nx.get_all_data_types()))
    params.insert(nx.Parameters.Separator("Created Data Objects"))
    params.insert(nx.DataGroupCreationParameter(CalculateHistogramFilter.HISTOGRAM_STATS_ATTRMAT_PATH, 'Histogram Stats Attribute Matrix', 'The path to the newly created histogram statistics attribute matrix.', nx.DataPath("Histogram Statistics")))
    params.insert(nx.DataObjectNameParameter(CalculateHistogramFilter.COUNTS, 'Histogram Counts', 'The name of the array that will contain the number of values that appear in each bin.', "Counts"))
    params.insert(nx.DataObjectNameParameter(CalculateHistogramFilter.BIN_BOUNDS, 'Histogram Bin Bounds', 'The name of the array that will contain the minimum and maximum values for each bin.', "Bin Bounds"))
    params.insert(nx.DataGroupCreationParameter(CalculateHistogramFilter.HISTOGRAM_GEOM_PATH, 'Histogram Image Geometry', 'The path to the newly created histogram image geometry.', nx.DataPath("Histogram Image Geometry")))
    params.insert(nx.DataObjectNameParameter(CalculateHistogramFilter.HISTOGRAM_CELLATTRMAT_NAME, 'Histogram Cell Attribute Matrix Name', 'The name of the histogram cell attribute matrix.', "Cell Data"))
    params.insert(nx.DataObjectNameParameter(CalculateHistogramFilter.HISTOGRAM_DATA_NAME, 'Histogram Array Name', 'The name of the histogram data array.', "Histogram Image Data"))

    params.link_parameters(CalculateHistogramFilter.GENERATE_HISTOGRAM_IMAGE, CalculateHistogramFilter.HISTOGRAM_IMAGE_SIZE, True)
    params.link_parameters(CalculateHistogramFilter.GENERATE_HISTOGRAM_IMAGE, CalculateHistogramFilter.HISTOGRAM_IMAGE_DPI, True)
    params.link_parameters(CalculateHistogramFilter.GENERATE_HISTOGRAM_IMAGE, CalculateHistogramFilter.HISTOGRAM_TITLE, True)
    params.link_parameters(CalculateHistogramFilter.GENERATE_HISTOGRAM_IMAGE, CalculateHistogramFilter.HISTOGRAM_X_LABEL, True)
    params.link_parameters(CalculateHistogramFilter.GENERATE_HISTOGRAM_IMAGE, CalculateHistogramFilter.HISTOGRAM_Y_LABEL, True)
    params.link_parameters(CalculateHistogramFilter.GENERATE_HISTOGRAM_IMAGE, CalculateHistogramFilter.HISTOGRAM_GEOM_PATH, True)
    params.link_parameters(CalculateHistogramFilter.GENERATE_HISTOGRAM_IMAGE, CalculateHistogramFilter.HISTOGRAM_CELLATTRMAT_NAME, True)
    params.link_parameters(CalculateHistogramFilter.GENERATE_HISTOGRAM_IMAGE, CalculateHistogramFilter.HISTOGRAM_DATA_NAME, True)

    return params

  def parameters_version(self) -> int:
    return 1

  def preflight_impl(self, data_structure: nx.DataStructure, args: dict, message_handler: nx.IFilter.MessageHandler, should_cancel: nx.AtomicBoolProxy) -> nx.IFilter.PreflightResult:
    histogram_image_size: list = args[CalculateHistogramFilter.HISTOGRAM_IMAGE_SIZE]
    num_of_bins: int = args[CalculateHistogramFilter.NUMBER_OF_BINS]
    generate_histogram_image: bool = args[CalculateHistogramFilter.GENERATE_HISTOGRAM_IMAGE]
    histogram_geom_path: nx.DataPath = args[CalculateHistogramFilter.HISTOGRAM_GEOM_PATH]
    histogram_stats_attrmat_path = args[CalculateHistogramFilter.HISTOGRAM_STATS_ATTRMAT_PATH]
    histogram_cellattrmat_name: str = args[CalculateHistogramFilter.HISTOGRAM_CELLATTRMAT_NAME]
    histogram_data_array_name: str = args[CalculateHistogramFilter.HISTOGRAM_DATA_NAME]
    counts_name: str = args[CalculateHistogramFilter.COUNTS]
    bin_bounds_name: str = args[CalculateHistogramFilter.BIN_BOUNDS]

    if num_of_bins <= 0:
      return nx.IFilter.PreflightResult(nx.OutputActions(), [nx.Error(-1000, "The number of bins should be at least 1.")])
    
    output_actions = nx.OutputActions()

    if generate_histogram_image == True:
      geom_dims = [histogram_image_size[0], histogram_image_size[1], 1]
      geom_origin = [0, 0, 0]
      geom_spacing = [1, 1, 1]

      output_actions.append_action(nx.CreateImageGeometryAction(histogram_geom_path, geom_dims, geom_origin, geom_spacing, histogram_cellattrmat_name))
      histogram_data_array_path = histogram_geom_path.create_child_path(histogram_cellattrmat_name).create_child_path(histogram_data_array_name)
      output_actions.append_action(nx.CreateArrayAction(nx.DataType.uint8, list(reversed(geom_dims)), [3], histogram_data_array_path))

    output_actions.append_action(nx.CreateAttributeMatrixAction(histogram_stats_attrmat_path, [num_of_bins]))

    counts_path = histogram_stats_attrmat_path.create_child_path(counts_name)
    output_actions.append_action(nx.CreateArrayAction(nx.DataType.uint64, [num_of_bins], [1], counts_path))

    bin_bounds_path = histogram_stats_attrmat_path.create_child_path(bin_bounds_name)
    output_actions.append_action(nx.CreateArrayAction(nx.DataType.float32, [num_of_bins], [2], bin_bounds_path))

    return nx.IFilter.PreflightResult(output_actions)

  def execute_impl(self, data_structure: nx.DataStructure, args: dict, message_handler: nx.IFilter.MessageHandler, should_cancel: nx.AtomicBoolProxy) -> nx.IFilter.ExecuteResult:
    input_data_path: nx.DataPath = args[CalculateHistogramFilter.INPUT_ARRAY_PATH]
    histogram_image_size: list = args[CalculateHistogramFilter.HISTOGRAM_IMAGE_SIZE]
    histogram_image_dpi: int = args[CalculateHistogramFilter.HISTOGRAM_IMAGE_DPI]
    num_of_bins: int = args[CalculateHistogramFilter.NUMBER_OF_BINS]
    generate_histogram_image: bool = args[CalculateHistogramFilter.GENERATE_HISTOGRAM_IMAGE]
    histogram_title: str = args[CalculateHistogramFilter.HISTOGRAM_TITLE]
    histogram_x_label: str = args[CalculateHistogramFilter.HISTOGRAM_X_LABEL]
    histogram_y_label: str = args[CalculateHistogramFilter.HISTOGRAM_Y_LABEL]
    histogram_geom_path: nx.DataPath = args[CalculateHistogramFilter.HISTOGRAM_GEOM_PATH]
    histogram_stats_attrmat_path = args[CalculateHistogramFilter.HISTOGRAM_STATS_ATTRMAT_PATH]
    histogram_cellattrmat_name: str = args[CalculateHistogramFilter.HISTOGRAM_CELLATTRMAT_NAME]
    histogram_data_array_name: str = args[CalculateHistogramFilter.HISTOGRAM_DATA_NAME]
    counts_name: str = args[CalculateHistogramFilter.COUNTS]
    bin_bounds_name: str = args[CalculateHistogramFilter.BIN_BOUNDS]

    message_handler(nx.IFilter.Message(nx.IFilter.Message.Type.Info, f'Calculating Histogram Counts and Bin Bounds...'))

    input_data_array = data_structure[input_data_path]
    input_data_store = input_data_array.store
    input_npdata = input_data_store.npview().copy()

    counts_path = histogram_stats_attrmat_path.create_child_path(counts_name)
    bin_bounds_path = histogram_stats_attrmat_path.create_child_path(bin_bounds_name)
    counts, bin_edges = np.histogram(input_npdata, num_of_bins)
    bin_mins = bin_edges[:-1]  # Minimum values of each bin
    bin_maxs = bin_edges[1:]   # Maximum values of each bin
    bin_min_max_pairs = np.array(list(zip(bin_mins, bin_maxs)))

    # Store the Counts data
    counts_array = data_structure[counts_path]
    counts_data_store = counts_array.store
    counts_npdata = counts_data_store.npview()
    counts_npdata[:] = counts.reshape(len(counts), 1)

    # Store the Bin Bounds data
    bin_bounds_array = data_structure[bin_bounds_path]
    bin_bounds_data_store = bin_bounds_array.store
    bin_bounds_npdata = bin_bounds_data_store.npview()
    bin_bounds_npdata[:] = bin_min_max_pairs

    if generate_histogram_image == True:
      message_handler(nx.IFilter.Message(nx.IFilter.Message.Type.Info, f'Generating Histogram Image...'))

      # Calculate figure size in inches
      figure_width_in = histogram_image_size[0] / histogram_image_dpi
      figure_height_in = histogram_image_size[1] / histogram_image_dpi

      # Create a histogram
      plt.figure(figsize=(figure_width_in, figure_height_in), dpi=histogram_image_dpi)
      plt.hist(np.ravel(input_npdata), bins=num_of_bins, edgecolor='black')
      plt.title(histogram_title)
      plt.xlabel(histogram_x_label)
      plt.ylabel(histogram_y_label)
      plt.tight_layout()

      # Save the plot to a BytesIO object
      buf = io.BytesIO()
      plt.savefig(buf, format='tiff')
      buf.seek(0)

      # Open the image and convert to a NumPy array
      img = Image.open(buf)
      img = img.convert('RGB')
      img_array = np.array(img)
      buf.close()

      # Reshape the array to the desired dimensions
      img_array = img_array.reshape((1, img_array.shape[0], img_array.shape[1], img_array.shape[2]))

      # Store the histogram image data
      histogram_data_array_path = histogram_geom_path.create_child_path(histogram_cellattrmat_name).create_child_path(histogram_data_array_name)
      data_array = data_structure[histogram_data_array_path]
      data_store = data_array.store
      npdata = data_store.npview()
      npdata[:] = img_array

    return nx.Result()
