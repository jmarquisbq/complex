#include "GeometryUtilities.hpp"

#include "simplnx/Common/Array.hpp"
#include "simplnx/Common/Result.hpp"
#include "simplnx/Utilities/Math/MatrixMath.hpp"

using namespace nx::core;

namespace
{
constexpr float32 k_PartitionEdgePadding = 0.000001;
const Point3Df k_Padding(k_PartitionEdgePadding, k_PartitionEdgePadding, k_PartitionEdgePadding);
} // namespace

GeometryUtilities::FindUniqueIdsImpl::FindUniqueIdsImpl(VertexStore& vertexStore, const std::vector<std::vector<size_t>>& nodesInBin, nx::core::Int64DataStore& uniqueIds)
: m_VertexStore(vertexStore)
, m_NodesInBin(nodesInBin)
, m_UniqueIds(uniqueIds)
{
}

// -----------------------------------------------------------------------------
void GeometryUtilities::FindUniqueIdsImpl::convert(size_t start, size_t end) const
{
  int64* uniqueIdsPtr = m_UniqueIds.data();
  for(size_t i = start; i < end; i++)
  {
    for(size_t j = 0; j < m_NodesInBin[i].size(); j++)
    {
      size_t node1 = m_NodesInBin[i][j];
      if(uniqueIdsPtr[node1] == static_cast<int64_t>(node1))
      {
        for(size_t k = j + 1; k < m_NodesInBin[i].size(); k++)
        {
          size_t node2 = m_NodesInBin[i][k];
          if(m_VertexStore[node1 * 3] == m_VertexStore[node2 * 3] && m_VertexStore[node1 * 3 + 1] == m_VertexStore[node2 * 3 + 1] && m_VertexStore[node1 * 3 + 2] == m_VertexStore[node2 * 3 + 2])
          {
            uniqueIdsPtr[node2] = node1;
          }
        }
      }
    }
  }
}

// -----------------------------------------------------------------------------
void GeometryUtilities::FindUniqueIdsImpl::operator()(const Range& range) const
{
  convert(range.min(), range.max());
}

Result<FloatVec3> GeometryUtilities::CalculatePartitionLengthsByPartitionCount(const INodeGeometry0D& geometry, const SizeVec3& numberOfPartitionsPerAxis)
{
  BoundingBox3Df boundingBox = geometry.getBoundingBox();
  if(!boundingBox.isValid())
  {
    return {};
  }
  return GeometryUtilities::CalculatePartitionLengthsOfBoundingBox({(boundingBox.getMinPoint() - k_Padding), (boundingBox.getMaxPoint() + k_Padding)}, numberOfPartitionsPerAxis);
}

Result<FloatVec3> GeometryUtilities::CalculatePartitionLengthsByPartitionCount(const ImageGeom& geometry, const SizeVec3& numberOfPartitionsPerAxis)
{
  SizeVec3 dims = geometry.getDimensions();
  FloatVec3 spacing = geometry.getSpacing();
  float32 lengthX = static_cast<float32>(dims.getX()) / static_cast<float32>(numberOfPartitionsPerAxis.getX()) * spacing[0];
  float32 lengthY = static_cast<float32>(dims.getY()) / static_cast<float32>(numberOfPartitionsPerAxis.getY()) * spacing[1];
  float32 lengthZ = static_cast<float32>(dims.getZ()) / static_cast<float32>(numberOfPartitionsPerAxis.getZ()) * spacing[2];
  return Result<FloatVec3>{FloatVec3(lengthX, lengthY, lengthZ)};
}

Result<FloatVec3> GeometryUtilities::CalculatePartitionLengthsByPartitionCount(const RectGridGeom& geometry, const SizeVec3& numberOfPartitionsPerAxis)
{
  const Float32Array* xBounds = geometry.getXBounds();
  const Float32Array* yBounds = geometry.getYBounds();
  const Float32Array* zBounds = geometry.getZBounds();

  if(xBounds == nullptr)
  {
    return MakeErrorResult<FloatVec3>(-4000, "Unable to calculate partition lengths using the partition count - X Bounds array is not available.");
  }

  if(yBounds == nullptr)
  {
    return MakeErrorResult<FloatVec3>(-4001, "Unable to calculate partition lengths using the partition count - Y Bounds array is not available.");
  }

  if(zBounds == nullptr)
  {
    return MakeErrorResult<FloatVec3>(-4002, "Unable to calculate partition lengths using the partition count - Z Bounds array is not available.");
  }

  if(xBounds->getSize() == 0)
  {
    return MakeErrorResult<FloatVec3>(-4003, "Unable to calculate partition lengths using the partition count - X Bounds array is empty.");
  }

  if(yBounds->getSize() == 0)
  {
    return MakeErrorResult<FloatVec3>(-4004, "Unable to calculate partition lengths using the partition count - Y Bounds array is empty.");
  }

  if(zBounds->getSize() == 0)
  {
    return MakeErrorResult<FloatVec3>(-4005, "Unable to calculate partition lengths using the partition count - Z Bounds array is empty.");
  }

  FloatVec3 lengthPerPartition = {0.0f, 0.0f, 0.0f};

  const AbstractDataStore<float32>& xStore = xBounds->getDataStoreRef();
  float32 maxX = xStore.getValue(xBounds->getNumberOfTuples() - 1);
  lengthPerPartition.setX(maxX / static_cast<float32>(numberOfPartitionsPerAxis.getX()));

  const AbstractDataStore<float32>& yStore = yBounds->getDataStoreRef();
  float32 maxY = yStore.getValue(yBounds->getNumberOfTuples() - 1);
  lengthPerPartition.setY(maxY / static_cast<float32>(numberOfPartitionsPerAxis.getY()));

  const AbstractDataStore<float32>& zStore = yBounds->getDataStoreRef();
  float32 maxZ = zStore.getValue(zBounds->getNumberOfTuples() - 1);
  lengthPerPartition.setZ(maxZ / static_cast<float32>(numberOfPartitionsPerAxis.getZ()));

  return Result<FloatVec3>{lengthPerPartition};
}

Result<FloatVec3> GeometryUtilities::CalculateNodeBasedPartitionSchemeOrigin(const INodeGeometry0D& geometry)
{
  BoundingBox3Df boundingBox = geometry.getBoundingBox();
  if(!boundingBox.isValid())
  {
    return {};
  }
  return Result<FloatVec3>{FloatVec3(boundingBox.getMinPoint() - k_Padding)};
}

Result<FloatVec3> GeometryUtilities::CalculatePartitionLengthsOfBoundingBox(const BoundingBox3Df& boundingBox, const SizeVec3& numberOfPartitionsPerAxis)
{
  auto min = boundingBox.getMinPoint();
  auto max = boundingBox.getMaxPoint();
  // Calculate the length per partition for each dimension, and set it into the partitioning scheme image geometry
  float32 lengthX = ((max[0] - min[0]) / static_cast<float32>(numberOfPartitionsPerAxis.getX()));
  float32 lengthY = ((max[1] - min[1]) / static_cast<float32>(numberOfPartitionsPerAxis.getY()));
  float32 lengthZ = ((max[2] - min[2]) / static_cast<float32>(numberOfPartitionsPerAxis.getZ()));
  FloatVec3 lengthPerPartition = {lengthX, lengthY, lengthZ};
  return Result<FloatVec3>{lengthPerPartition};
}

/**
 * @brief The ComputeTriangleAreasImpl class implements a threaded algorithm that computes the area of each
 * triangle for a set of triangles
 */
class ComputeTriangleAreasImpl
{
public:
  ComputeTriangleAreasImpl(const TriangleGeom* triangleGeom, Float64AbstractDataStore& areas, const std::atomic_bool& shouldCancel)
  : m_TriangleGeom(triangleGeom)
  , m_Areas(areas)
  , m_ShouldCancel(shouldCancel)
  {
  }
  virtual ~ComputeTriangleAreasImpl() = default;

  void convert(size_t start, size_t end) const
  {
    std::array<float, 3> cross = {0.0f, 0.0f, 0.0f};
    for(size_t triangleIndex = start; triangleIndex < end; triangleIndex++)
    {
      if(m_ShouldCancel)
      {
        break;
      }
      std::array<Point3Df, 3> vertCoords;
      m_TriangleGeom->getFaceCoordinates(triangleIndex, vertCoords);
      m_Areas[triangleIndex] = 0.5F * (vertCoords[0] - vertCoords[1]).cross(vertCoords[0] - vertCoords[2]).magnitude();
    }
  }

  void operator()(const Range& range) const
  {
    convert(range.min(), range.max());
  }

private:
  const TriangleGeom* m_TriangleGeom = nullptr;
  Float64AbstractDataStore& m_Areas;
  const std::atomic_bool& m_ShouldCancel;
};

Result<> GeometryUtilities::ComputeTriangleAreas(const nx::core::TriangleGeom* triangleGeom, Float64AbstractDataStore& faceAreas, const std::atomic_bool& shouldCancel)
{
  // Parallel algorithm to find duplicate nodes
  ParallelDataAlgorithm dataAlg;
  dataAlg.setRange(0ULL, static_cast<size_t>(triangleGeom->getNumberOfFaces()));
  dataAlg.execute(ComputeTriangleAreasImpl(triangleGeom, faceAreas, shouldCancel));

  return {};
}

/**
 * @brief The CalculateAreasImpl class implements a threaded algorithm that computes the normal of each
 * triangle for a set of triangles
 */
class CalculateNormalsImpl
{
public:
  CalculateNormalsImpl(const TriangleGeom* triangleGeom, Float64AbstractDataStore& normals, const std::atomic_bool& shouldCancel)
  : m_TriangleGeom(triangleGeom)
  , m_Normals(normals)
  , m_ShouldCancel(shouldCancel)
  {
  }
  virtual ~CalculateNormalsImpl() = default;

  void generate(size_t start, size_t end) const
  {
    for(size_t triangleIndex = start; triangleIndex < end; triangleIndex++)
    {
      if(m_ShouldCancel)
      {
        break;
      }
      std::array<Point3Df, 3> vertCoords;
      m_TriangleGeom->getFaceCoordinates(triangleIndex, vertCoords);

      auto normal = (vertCoords[1] - vertCoords[0]).cross(vertCoords[2] - vertCoords[0]);
      normal = normal / normal.magnitude();

      m_Normals[triangleIndex * 3] = static_cast<float64>(normal[0]);
      m_Normals[triangleIndex * 3 + 1] = static_cast<float64>(normal[1]);
      m_Normals[triangleIndex * 3 + 2] = static_cast<float64>(normal[2]);
    }
  }

  void operator()(const Range& range) const
  {
    generate(range.min(), range.max());
  }

private:
  const TriangleGeom* m_TriangleGeom = nullptr;
  Float64AbstractDataStore& m_Normals;
  const std::atomic_bool& m_ShouldCancel;
};

Result<> GeometryUtilities::ComputeTriangleNormals(const nx::core::TriangleGeom* triangleGeom, Float64AbstractDataStore& normals, const std::atomic_bool& shouldCancel)
{
  // Parallel algorithm to find duplicate nodes
  ParallelDataAlgorithm dataAlg;
  dataAlg.setRange(0ULL, static_cast<size_t>(triangleGeom->getNumberOfFaces()));
  dataAlg.execute(CalculateNormalsImpl(triangleGeom, normals, shouldCancel));

  return {};
}
