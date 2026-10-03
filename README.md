# CUDA Path Tracer

**University of Pennsylvania, CIS 565: GPU Programming and Architecture, Project 3**

* Christopher Yuen
  * [LinkedIn](https://www.linkedin.com/in/christopher-yuen-16b5a8221/)
* Tested on: Windows 11, 14th Gen Intel(R) Core(TM) i9-14900HX @ 2.22GHz 32GB, RTX 4090 Laptop GPU 16GB (Personal Laptop)

![final_showcase](img/)

## Overview
This is a CUDA path tracer that renders globally-illuminated scenes on the GPU. Rays are fired from a virtual camera, scattered off surfaces according to their BSDF, and accumulated over many iterations to converge on a noise-free image. The renderer pipeline is as follows: ray generation, intersection, shading, stream compaction, and final gather. Rendered results are averaged across iterations. 

Rendering is iterative with each frame producing one sample per pixel. The renderer supports arbitrary OBJ meshes, acceleration structures for cheap ray-scene intersection, dielectric materials with Fresnel, and a thin-lens camera.

## Outline
The rendered is structured as a per-bounce pipeline:
1. **Ray generation:**  one camera ray per pixel, jittered within the pixel for antialiasing along with optional redirection through a thin lens for depth of field.
2. **Intersection:** each active ray tested against analytic primitives (spheres, cubs) and against one or more imported triangle meshes via BVH traversal.
3. **Shading & scattering:** the hit material evaluates its BSDF, the path throughput is updated, and a new ray is produced for the next bounce.
4. **Stream compaction:** terminated and escaped paths are removed from the active path array so later passes operate on smaller working set.
5. **Final gather:** paths that reach an emissive surface contribute their accumulated throughput to the framebuffer.

## Part 1 - Core Features
### <ins>Diffuse BSDF</ins>

### <ins>Specular BSDF</ins>

### <ins>Material-Coherent Ray Reordering</ins>

### <ins>Stream Compaction</ins>

### <ins>Stochastic Sampled Antialiasing</ins>

### <ins>Radiance/Throughput Separation</ins>

## Part 2 - Extra Features 
### <ins>OBJ Mesh Loading</ins>

### <ins>Bounding Volume Hierarchy</ins>

### <ins>Refraction and Fresnel</ins>

### <ins>Physically-Based Depth of Field</ins>

## Performance Analysis
All measurements below are from the Release build on the RTX 4090 Laptop listed at the top of this README, with `ERRORCHECK` set to `0` and V-Sync off. The measurements are application-level (`ms/frame` from the ImGui overlay).

### <ins>Stream Compaction</ins>

### <ins>Material Sorting</ins>

### <ins>BVH Traversal</ins>

### <ins>Refraction and Fresnel</ins>

### <ins>Depth of Field</ins> 

## Build and Run
The project is built with CMake and Visual Studio 2022 on Windows 11. 

**Prerequisites:**
* CUDA toolkit (v13.3)
* Visual Studio 2022 with CUDA workload
* `tiny_obj_loader.h` in `external/include`

**Run:** 
```
.\build\bin\Release\cis565_path_tracer.exe .\scenes\cornell.json
```

**Controls:**
* `ESC` — save & exit
* `S` — save image
* `Space` — re-center camera to original lookAt point
* Left mouse — orbit camera
* Right mouse — zoom
* Middle mouse — pan lookAt point in X/Z plane

**Object Transforms:** Imported meshes use the same `TRANS` / `ROTAT` / `SCALE` block as primitives:
```
"TYPE": "mesh",
"MATERIAL": "specular_white",
"FILE": "../scenes/mesh.obj",
"TRANS": [0.0, 4.0, 0.0],
"ROTAT": [90.0, 180.0, 0.0],
"SCALE": [0.5, 0.5, 0.5]
```

**Camera Extensions:** camera block accepts 2 new fields `APERTURE` and `FOCAL_DISTANCE`
```
"Camera": {
    ...
    "APERTURE": 0.1,
    "FOCAL_DISTANCE": 10.5
}
```
`APERTURE` defaults to `0.0` (pinhole) & `FOCAL_DISTANCE` defaults to `length(lookAt - position)`

**Refractive Materials:** `Refractive` material type with `IOR` field
```
"glass": {
    "TYPE": "Refractive",
    "RGB": [1.0, 1.0, 1.0],
    "IOR": 1.5
}
```

## Third-Party Assets
* `tiny_obj_loader.h` — single-header [Wavefront OBJ parser](https://github.com/tinyobjloader/tinyobjloader)

