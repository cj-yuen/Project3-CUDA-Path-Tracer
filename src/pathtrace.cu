#include "pathtrace.h"

#include <cstdio>
#include <cuda.h>
#include <cmath>
#include <thrust/execution_policy.h>
#include <thrust/random.h>
#include <thrust/remove.h>
#include <thrust/partition.h>
#include <thrust/sort.h>
#include <thrust/transform.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/tuple.h>

#include "sceneStructs.h"
#include "scene.h"
#include "glm/glm.hpp"
#include "glm/gtx/norm.hpp"
#include "utilities.h"
#include "intersections.h"
#include "interactions.h"
#include <iostream>

#define ERRORCHECK 0
#define SORT_BY_MATERIAL 0  // toggle to make same material contiguous in mem. before shading (1 = sort, 0 = no sort)
#define BVH_BBOX_CULLING 1  // 1 = cull w/mesh bbox, 0 = test every triangle

#define FILENAME (strrchr(__FILE__, '/') ? strrchr(__FILE__, '/') + 1 : __FILE__)
#define checkCUDAError(msg) checkCUDAErrorFn(msg, FILENAME, __LINE__)
void checkCUDAErrorFn(const char* msg, const char* file, int line)
{
#if ERRORCHECK
    cudaDeviceSynchronize();
    cudaError_t err = cudaGetLastError();
    if (cudaSuccess == err)
    {
        return;
    }

    fprintf(stderr, "CUDA error");
    if (file)
    {
        fprintf(stderr, " (%s:%d)", file, line);
    }
    fprintf(stderr, ": %s: %s\n", msg, cudaGetErrorString(err));
#ifdef _WIN32
    getchar();
#endif // _WIN32
    exit(EXIT_FAILURE);
#endif // ERRORCHECK
}

__host__ __device__
thrust::default_random_engine makeSeededRandomEngine(int iter, int index, int depth)
{
    int h = utilhash((1 << 31) | (depth << 22) | iter) ^ utilhash(index);
    return thrust::default_random_engine(h);
}

//Kernel that writes the image to the OpenGL PBO directly.
__global__ void sendImageToPBO(uchar4* pbo, glm::ivec2 resolution, int iter, glm::vec3* image)
{
    int x = (blockIdx.x * blockDim.x) + threadIdx.x;
    int y = (blockIdx.y * blockDim.y) + threadIdx.y;

    if (x < resolution.x && y < resolution.y)
    {
        int index = x + (y * resolution.x);
        glm::vec3 pix = image[index] / (float)iter;

        // pix = glm::pow(glm::max(pix, glm::vec3(0.0f)), glm::vec3(1.0f / 2.2f)); // gamma-correction

        glm::ivec3 color;
        color.x = glm::clamp((int)(pix.x * 255.0), 0, 255);
        color.y = glm::clamp((int)(pix.y * 255.0), 0, 255);
        color.z = glm::clamp((int)(pix.z * 255.0), 0, 255);

        // Each thread writes one pixel location in the texture (textel)
        pbo[index].w = 0;
        pbo[index].x = color.x;
        pbo[index].y = color.y;
        pbo[index].z = color.z;
    }
}

static Scene* hst_scene = NULL;
static GuiDataContainer* guiData = NULL;
static glm::vec3* dev_image = NULL;
static Geom* dev_geoms = NULL;
static Material* dev_materials = NULL;
static PathSegment* dev_paths = NULL;
static ShadeableIntersection* dev_intersections = NULL;
// TODO: static variables for device memory, any extra info you need, etc
// ...
static int* dev_materialIds = NULL; // material sorting
static Triangle* dev_triangles = NULL;
static MeshInfo* dev_meshInfos = NULL;
static int hst_num_meshes = 0;
static BVHNode* dev_bvhNodes = NULL;

void InitDataContainer(GuiDataContainer* imGuiData)
{
    guiData = imGuiData;
}

void pathtraceInit(Scene* scene)
{
    hst_scene = scene;

    const Camera& cam = hst_scene->state.camera;
    const int pixelcount = cam.resolution.x * cam.resolution.y;

    cudaMalloc(&dev_image, pixelcount * sizeof(glm::vec3));
    cudaMemset(dev_image, 0, pixelcount * sizeof(glm::vec3));

    cudaMalloc(&dev_paths, pixelcount * sizeof(PathSegment));

    cudaMalloc(&dev_geoms, scene->geoms.size() * sizeof(Geom));
    cudaMemcpy(dev_geoms, scene->geoms.data(), scene->geoms.size() * sizeof(Geom), cudaMemcpyHostToDevice);

    cudaMalloc(&dev_materials, scene->materials.size() * sizeof(Material));
    cudaMemcpy(dev_materials, scene->materials.data(), scene->materials.size() * sizeof(Material), cudaMemcpyHostToDevice);

    cudaMalloc(&dev_intersections, pixelcount * sizeof(ShadeableIntersection));
    cudaMemset(dev_intersections, 0, pixelcount * sizeof(ShadeableIntersection));

    // TODO: initialize any extra device memeory you need
    cudaMalloc(&dev_materialIds, pixelcount * sizeof(int));

    // flatten all meshes --> triangle array
    int totalTris = 0;
    int totalBVH = 0;
    for (const auto& m : scene->meshes) {
        totalTris += (int)m.triangles.size();
        totalBVH += (int)m.bvhNodes.size();
    }
    hst_num_meshes = (int)scene->meshes.size();

    if (totalTris > 0) {
        std::vector<Triangle> flatTris;
        std::vector<BVHNode>  flatBVH;
        std::vector<MeshInfo> meshInfos;
        flatTris.reserve(totalTris);
        flatBVH.reserve(totalBVH);

        int triOffset = 0;
        int bvhOffset = 0;

        for (const auto& m : scene->meshes) {
            MeshInfo info;
            info.triStart = triOffset;
            info.triCount = (int)m.triangles.size();
            info.bvhStart = bvhOffset;
            info.bvhCount = (int)m.bvhNodes.size();
            info.materialid = m.materialid;
            info.bboxMin = m.bboxMin;
            info.bboxMax = m.bboxMax;
            meshInfos.push_back(info);

            for (const auto& t : m.triangles) {
                flatTris.push_back(t);
            }

            for (const auto& n : m.bvhNodes) {
                flatBVH.push_back(n);
            }

            triOffset += (int)m.triangles.size();
            bvhOffset += (int)m.bvhNodes.size();
        }

        cudaMalloc(&dev_triangles, totalTris * sizeof(Triangle));
        cudaMemcpy(dev_triangles, flatTris.data(),
            totalTris * sizeof(Triangle), cudaMemcpyHostToDevice);

        cudaMalloc(&dev_bvhNodes, totalBVH * sizeof(BVHNode));
        cudaMemcpy(dev_bvhNodes, flatBVH.data(),
            totalBVH * sizeof(BVHNode), cudaMemcpyHostToDevice);

        cudaMalloc(&dev_meshInfos, hst_num_meshes * sizeof(MeshInfo));
        cudaMemcpy(dev_meshInfos, meshInfos.data(),
            hst_num_meshes * sizeof(MeshInfo), cudaMemcpyHostToDevice);

        std::cout << "Uploaded " << totalTris << " triangles and "
            << totalBVH << " BVH nodes from "
            << hst_num_meshes << " mesh(es) to GPU." << std::endl;
    }

    checkCUDAError("pathtraceInit");
}

void pathtraceFree()
{
    cudaFree(dev_image);  // no-op if dev_image is null
    cudaFree(dev_paths);
    cudaFree(dev_geoms);
    cudaFree(dev_materials);
    cudaFree(dev_intersections);
    // TODO: clean up any extra device memory you created
    cudaFree(dev_materialIds);
    
    if (dev_triangles) {
        cudaFree(dev_triangles);
    }
    if (dev_meshInfos) {
        cudaFree(dev_meshInfos);
    }
    if (dev_bvhNodes) {
        cudaFree(dev_bvhNodes);
    }
    dev_triangles = NULL;
    dev_meshInfos = NULL;
    dev_bvhNodes = NULL;
    hst_num_meshes = 0;

    checkCUDAError("pathtraceFree");
}

/**
* Generate PathSegments with rays from the camera through the screen into the
* scene, which is the first bounce of rays.
*
* Antialiasing - add rays for sub-pixel sampling
* motion blur - jitter rays "in time"
* lens effect - jitter ray origin positions based on a lens
*/
__global__ void generateRayFromCamera(Camera cam, int iter, int traceDepth, PathSegment* pathSegments)
{
    int x = (blockIdx.x * blockDim.x) + threadIdx.x;
    int y = (blockIdx.y * blockDim.y) + threadIdx.y;

    if (x < cam.resolution.x && y < cam.resolution.y) {
        int index = x + (y * cam.resolution.x);
        PathSegment& segment = pathSegments[index];

        segment.ray.origin = cam.position;
        segment.color = glm::vec3(1.0f, 1.0f, 1.0f);

        // TODO: implement antialiasing by jittering the ray
		thrust::default_random_engine rng = makeSeededRandomEngine(iter, index, 0);
        thrust::uniform_real_distribution<float> u01(0, 1);
        float jitterX = u01(rng) - 0.5f;
        float jitterY = u01(rng) - 0.5f;

        // pinhole ray dir 
        glm::vec3 pinholeDir = glm::normalize(cam.view 
            - cam.right * cam.pixelLength.x * (((float)x + jitterX) - (float)cam.resolution.x * 0.5f) 
            - cam.up * cam.pixelLength.y * (((float)y + jitterY) - (float)cam.resolution.y * 0.5f));

        // pt on focal plane
        float t = cam.focalDistance / glm::dot(pinholeDir, cam.view);
        glm::vec3 focalPoint = cam.position + t * pinholeDir;

        // sample pt on aperture lens disck (uniform area)
        float r = cam.aperture * sqrtf(u01(rng));
        float theta = TWO_PI * u01(rng);
        glm::vec3 lensOffset = cam.right * (r * cosf(theta)) 
            + cam.up * (r * sinf(theta));

        // fire from lens
        segment.ray.origin = cam.position + lensOffset;
        segment.ray.direction = glm::normalize(focalPoint - segment.ray.origin);

        segment.pixelIndex = index;
        segment.remainingBounces = traceDepth;
        segment.hitLight = 0;
    }
}

// TODO:
// computeIntersections handles generating ray intersections ONLY.
// Generating new rays is handled in your shader(s).
// Feel free to modify the code below.
__global__ void computeIntersections(
    int depth,
    int num_paths,
    PathSegment* pathSegments,
    Geom* geoms,
    int geoms_size,
    Triangle* triangles,
    BVHNode* bvhNodes,
    MeshInfo* meshes,
    int num_meshes,
    ShadeableIntersection* intersections)
{
    int path_index = blockIdx.x * blockDim.x + threadIdx.x;

    if (path_index < num_paths)
    {
        PathSegment pathSegment = pathSegments[path_index];

        float t;
        glm::vec3 intersect_point;
        glm::vec3 normal;
        float t_min = FLT_MAX;
        int hit_geom_index = -1;    // sphere/cube
        int hit_mesh_index = -1;    // meshes
        bool outside = true;
        bool hitOutside = true;

        glm::vec3 tmp_intersect;
        glm::vec3 tmp_normal;

        // naive parse through global geoms

        for (int i = 0; i < geoms_size; i++)
        {
            Geom& geom = geoms[i];

            if (geom.type == CUBE)
            {
                t = boxIntersectionTest(geom, pathSegment.ray, tmp_intersect, tmp_normal, outside);
            }
            else if (geom.type == SPHERE)
            {
                t = sphereIntersectionTest(geom, pathSegment.ray, tmp_intersect, tmp_normal, outside);
            }
            // TODO: add more intersection tests here... triangle? metaball? CSG?

            // Compute the minimum t from the intersection tests to determine what
            // scene geometry object was hit first.
            if (t > 0.0f && t_min > t)
            {
                t_min = t;
                hit_geom_index = i;
                intersect_point = tmp_intersect;
                normal = tmp_normal;
                hitOutside = outside;
            }
        }

        // mesh loop (naive iterate all triangles per mesh)
        for (int m = 0; m < num_meshes; m++) {
            MeshInfo& mesh = meshes[m];

#if BVH_BBOX_CULLING
            // bbox rejection 
            if (!bboxIntersectionTest(mesh.bboxMin, mesh.bboxMax, pathSegment.ray)) {
                continue;
            }
#endif

            glm::vec3 meshNormal;
            bool meshOutside;
            float t = intersectMeshBVH(
                bvhNodes + mesh.bvhStart,
                triangles + mesh.triStart,
                pathSegment.ray,
                meshNormal, meshOutside);

            if (t > 0.0f && t < t_min) {
                t_min = t;
                hit_mesh_index = m;
                hit_geom_index = -1;
                intersect_point = pathSegment.ray.origin + t * pathSegment.ray.direction;
                normal = meshNormal;
                hitOutside = meshOutside;
            }
        }

        if (hit_geom_index == -1 && hit_mesh_index == -1)
        {
            intersections[path_index].t = -1.0f;
        }
        else
        {
            // The ray hits something
            intersections[path_index].t = t_min;
            intersections[path_index].surfaceNormal = normal;
            intersections[path_index].outside = hitOutside;

            if (hit_mesh_index >= 0) {
                intersections[path_index].materialId = meshes[hit_mesh_index].materialid;
            }
            else {
                intersections[path_index].materialId = geoms[hit_geom_index].materialid;
            }
        }
    }
}

__global__ void shadeMaterial(
    int iter,
    int depth,
    int num_paths,
	ShadeableIntersection* shadeableIntersections,
	PathSegment* pathSegments,
	Material* materials) 
{ 
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_paths) {
        PathSegment& pathSegment = pathSegments[idx];
        if (pathSegment.remainingBounces <= 0) {
            return;
        }

        pathSegment.hitLight = 0;

		ShadeableIntersection intersection = shadeableIntersections[idx];
        if (intersection.t > 0.0f) {
			thrust::default_random_engine rng = makeSeededRandomEngine(iter, pathSegment.pixelIndex, depth);

            Material material = materials[intersection.materialId];
			glm::vec3 materialColor = material.color;

            if (material.emittance > 0.0f) {
				pathSegment.color *= (materialColor * material.emittance);
                pathSegment.remainingBounces = 0;
                pathSegment.hitLight = 1;
            }
            else {
                glm::vec3 intersectPoint = pathSegment.ray.origin + intersection.t * pathSegment.ray.direction; 

				scatterRay(pathSegment, intersectPoint, intersection.surfaceNormal, material, intersection.outside, rng);
            }
        } else {
            pathSegment.color = glm::vec3(0.0f);
            pathSegment.remainingBounces = 0;
		}
    }
}

// LOOK: "fake" shader demonstrating what you might do with the info in
// a ShadeableIntersection, as well as how to use thrust's random number
// generator. Observe that since the thrust random number generator basically
// adds "noise" to the iteration, the image should start off noisy and get
// cleaner as more iterations are computed.
//
// Note that this shader does NOT do a BSDF evaluation!
// Your shaders should handle that - this can allow techniques such as
// bump mapping.
__global__ void shadeFakeMaterial(
    int iter,
    int num_paths,
    ShadeableIntersection* shadeableIntersections,
    PathSegment* pathSegments,
    Material* materials)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_paths)
    {
        ShadeableIntersection intersection = shadeableIntersections[idx];
        if (intersection.t > 0.0f) // if the intersection exists...
        {
          // Set up the RNG
          // LOOK: this is how you use thrust's RNG! Please look at
          // makeSeededRandomEngine as well.
            thrust::default_random_engine rng = makeSeededRandomEngine(iter, idx, 0);
            thrust::uniform_real_distribution<float> u01(0, 1);

            Material material = materials[intersection.materialId];
            glm::vec3 materialColor = material.color;

            // If the material indicates that the object was a light, "light" the ray
            if (material.emittance > 0.0f) {
                pathSegments[idx].color *= (materialColor * material.emittance);
            }
            // Otherwise, do some pseudo-lighting computation. This is actually more
            // like what you would expect from shading in a rasterizer like OpenGL.
            // TODO: replace this! you should be able to start with basically a one-liner
            else {
                float lightTerm = glm::dot(intersection.surfaceNormal, glm::vec3(0.0f, 1.0f, 0.0f));
                pathSegments[idx].color *= (materialColor * lightTerm) * 0.3f + ((1.0f - intersection.t * 0.02f) * materialColor) * 0.7f;
                pathSegments[idx].color *= u01(rng); // apply some noise because why not
            }
            // If there was no intersection, color the ray black.
            // Lots of renderers use 4 channel color, RGBA, where A = alpha, often
            // used for opacity, in which case they can indicate "no opacity".
            // This can be useful for post-processing and image compositing.
        }
        else {
            pathSegments[idx].color = glm::vec3(0.0f);
        }
    }
}

// Add the current iteration's output to the overall image
__global__ void finalGather(int nPaths, glm::vec3* image, PathSegment* iterationPaths)
{
    int index = (blockIdx.x * blockDim.x) + threadIdx.x;

    if (index < nPaths)
    {
        PathSegment iterationPath = iterationPaths[index];
        if (iterationPath.hitLight) {
            image[iterationPath.pixelIndex] += iterationPath.color;
        }
    }
}

struct PathIsActive {
    __host__ __device__ bool operator()(const PathSegment& path) const {
        return path.remainingBounces > 0;
    }
};

struct GetMaterialId {
    __host__ __device__ int operator()(const ShadeableIntersection& intersection) const {
        return intersection.materialId;
    }
};

/**
 * Wrapper for the __global__ call that sets up the kernel calls and does a ton
 * of memory management
 */
void pathtrace(uchar4* pbo, int frame, int iter)
{
    const int traceDepth = hst_scene->state.traceDepth;
    const Camera& cam = hst_scene->state.camera;
    const int pixelcount = cam.resolution.x * cam.resolution.y;

    // 2D block for generating ray from camera
    const dim3 blockSize2d(8, 8);
    const dim3 blocksPerGrid2d(
        (cam.resolution.x + blockSize2d.x - 1) / blockSize2d.x,
        (cam.resolution.y + blockSize2d.y - 1) / blockSize2d.y);

    // 1D block for path tracing
    const int blockSize1d = 128;

    ///////////////////////////////////////////////////////////////////////////

    // Recap:
    // * Initialize array of path rays (using rays that come out of the camera)
    //   * You can pass the Camera object to that kernel.
    //   * Each path ray must carry at minimum a (ray, color) pair,
    //   * where color starts as the multiplicative identity, white = (1, 1, 1).
    //   * This has already been done for you.
    // * For each depth:
    //   * Compute an intersection in the scene for each path ray.
    //     A very naive version of this has been implemented for you, but feel
    //     free to add more primitives and/or a better algorithm.
    //     Currently, intersection distance is recorded as a parametric distance,
    //     t, or a "distance along the ray." t = -1.0 indicates no intersection.
    //     * Color is attenuated (multiplied) by reflections off of any object
    //   * TODO: Stream compact away all of the terminated paths.
    //     You may use either your implementation or `thrust::remove_if` or its
    //     cousins.
    //     * Note that you can't really use a 2D kernel launch any more - switch
    //       to 1D.
    //   * TODO: Shade the rays that intersected something or didn't bottom out.
    //     That is, color the ray by performing a color computation according
    //     to the shader, then generate a new ray to continue the ray path.
    //     We recommend just updating the ray's PathSegment in place.
    //     Note that this step may come before or after stream compaction,
    //     since some shaders you write may also cause a path to terminate.
    // * Finally, add this iteration's results to the image. This has been done
    //   for you.

    // TODO: perform one iteration of path tracing

    generateRayFromCamera<<<blocksPerGrid2d, blockSize2d>>>(cam, iter, traceDepth, dev_paths);
    checkCUDAError("generate camera ray");

    int depth = 0;
    PathSegment* dev_path_end = dev_paths + pixelcount;
    int num_paths = dev_path_end - dev_paths;

    // --- PathSegment Tracing Stage ---
    // Shoot ray into scene, bounce between objects, push shading chunks

    bool iterationComplete = false;
    while (!iterationComplete)
    {
        // clean shading chunks
        cudaMemset(dev_intersections, 0, pixelcount * sizeof(ShadeableIntersection));

        // tracing
        dim3 numblocksPathSegmentTracing = (num_paths + blockSize1d - 1) / blockSize1d;
        computeIntersections << <numblocksPathSegmentTracing, blockSize1d >> > (
            depth,
            num_paths,
            dev_paths,
            dev_geoms,
            hst_scene->geoms.size(),
            dev_triangles,
            dev_bvhNodes,
            dev_meshInfos,
            hst_num_meshes,
            dev_intersections
        );
        checkCUDAError("trace one bounce");
        cudaDeviceSynchronize();
        depth++;

        // TODO:
        // --- Shading Stage ---
        // Shade path segments based on intersections and generate new rays by
        // evaluating the BSDF.
        // Start off with just a big kernel that handles all the different
        // materials you have in the scenefile.
        // TODO: compare between directly shading the path segments and shading
        // path segments that have been reshuffled to be contiguous in memory.

#if SORT_BY_MATERIAL
        thrust::transform(
            thrust::device,
            dev_intersections,
            dev_intersections + num_paths,
            dev_materialIds,
            GetMaterialId());
        
        thrust::sort_by_key(
            thrust::device,
            dev_materialIds,
            dev_materialIds + num_paths,
            thrust::make_zip_iterator(thrust::make_tuple(dev_paths, dev_intersections)));
#endif

        shadeMaterial<<<numblocksPathSegmentTracing, blockSize1d>>>(
            iter,
            depth, 
            num_paths,
            dev_intersections,
            dev_paths,
            dev_materials
        );

        // stream compaction
        PathSegment* dev_active_end = thrust::stable_partition(
            thrust::device,
            dev_paths,
            dev_paths + num_paths,
            PathIsActive());
        num_paths = dev_active_end - dev_paths;

        iterationComplete = (num_paths == 0) || (depth >= traceDepth); // TODO: should be based off stream compaction results.

        if (guiData != NULL)
        {
            guiData->TracedDepth = depth;
        }
    }

    // Assemble this iteration and apply it to the image
    dim3 numBlocksPixels = (pixelcount + blockSize1d - 1) / blockSize1d;
    finalGather<<<numBlocksPixels, blockSize1d>>>(pixelcount, dev_image, dev_paths);

    ///////////////////////////////////////////////////////////////////////////

    // Send results to OpenGL buffer for rendering
    sendImageToPBO<<<blocksPerGrid2d, blockSize2d>>>(pbo, cam.resolution, iter, dev_image);

    // Retrieve image from GPU
    cudaMemcpy(hst_scene->state.image.data(), dev_image,
        pixelcount * sizeof(glm::vec3), cudaMemcpyDeviceToHost);

    checkCUDAError("pathtrace");
}
