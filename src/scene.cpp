#include "scene.h"

#include "utilities.h"

#include <tiny_obj_loader.h>
#include <glm/gtc/matrix_transform.hpp>

#include <glm/gtc/matrix_inverse.hpp>
#include <glm/gtx/string_cast.hpp>
#include "json.hpp"

#include <fstream>
#include <iostream>
#include <string>
#include <unordered_map>

using namespace std;
using json = nlohmann::json;

Scene::Scene(string filename)
{
    cout << "Reading scene from " << filename << " ..." << endl;
    cout << " " << endl;
    auto ext = filename.substr(filename.find_last_of('.'));
    if (ext == ".json")
    {
        loadFromJSON(filename);
        return;
    }
    else
    {
        cout << "Couldn't read from " << filename << endl;
        exit(-1);
    }
}

void Scene::loadFromJSON(const std::string& jsonName)
{
    std::ifstream f(jsonName);
    json data = json::parse(f);
    const auto& materialsData = data["Materials"];
    std::unordered_map<std::string, uint32_t> MatNameToID;
    for (const auto& item : materialsData.items())
    {
        const auto& name = item.key();
        const auto& p = item.value();
        Material newMaterial{};
        // TODO: handle materials loading differently
        if (p["TYPE"] == "Diffuse")
        {
            const auto& col = p["RGB"];
            newMaterial.color = glm::vec3(col[0], col[1], col[2]);
        }
        else if (p["TYPE"] == "Emitting")
        {
            const auto& col = p["RGB"];
            newMaterial.color = glm::vec3(col[0], col[1], col[2]);
            newMaterial.emittance = p["EMITTANCE"];
        }
        else if (p["TYPE"] == "Specular")
        {
            const auto& col = p["RGB"];
            newMaterial.color = glm::vec3(col[0], col[1], col[2]);
            newMaterial.hasReflective = 1.f;
            newMaterial.specular.color = newMaterial.color;
            newMaterial.specular.exponent = p["ROUGHNESS"];
        }
        MatNameToID[name] = materials.size();
        materials.emplace_back(newMaterial);
    }
    const auto& objectsData = data["Objects"];
    for (const auto& p : objectsData)
    {
        const auto& type = p["TYPE"];

        if (type == "mesh") {
            tinyobj::attrib_t attrib;
            std::vector<tinyobj::shape_t> shapes;
            std::vector<tinyobj::material_t> objMaterials;
            std::string warn, err;

            std::string objPath = p["FILE"].get<std::string>();
            bool ret = tinyobj::LoadObj(&attrib, &shapes, &objMaterials,
                &warn, &err, objPath.c_str());

            if (!ret) {
                std::cerr << "Failed to load OBJ " << objPath << ": " << err << std::endl;
                exit(-1);
            }
            
            if (!warn.empty()) {
                std::cout << "OBJ warning: " << warn << std::endl;
            }

            const auto& trans = p["TRANS"];
            const auto& rotat = p["ROTAT"];
            const auto& scale = p["SCALE"];

            glm::mat4 xform = utilityCore::buildTransformationMatrix(
                glm::vec3(trans[0], trans[1], trans[2]),
                glm::vec3(rotat[0], rotat[1], rotat[2]),
                glm::vec3(scale[0], scale[1], scale[2]));

            // normal transform
            glm::mat3 normalXform = glm::transpose(glm::inverse(glm::mat3(xform)));

            TriangleMesh mesh;
            mesh.materialid = MatNameToID[p["MATERIAL"]];

            glm::vec3 bmin(FLT_MAX);
            glm::vec3 bmax(-FLT_MAX);

            for (const auto& shape : shapes) {
                size_t index_offset = 0;
                for (size_t f = 0; f < shape.mesh.num_face_vertices.size(); f++) {
                    size_t fv = shape.mesh.num_face_vertices[f];
                    
                    if (fv != 3) { // triangles only
                        index_offset += fv; continue; 
                    } 

                    Triangle tri{};
                    for (size_t v = 0; v < 3; v++) {
                        tinyobj::index_t idx = shape.mesh.indices[index_offset + v];

                        glm::vec3 pos(
                            attrib.vertices[3 * idx.vertex_index + 0],
                            attrib.vertices[3 * idx.vertex_index + 1],
                            attrib.vertices[3 * idx.vertex_index + 2]);

                        glm::vec3 pWorld = glm::vec3(xform * glm::vec4(pos, 1.0f));

                        glm::vec3 nrm(0.0f);
                        if (idx.normal_index >= 0) {
                            nrm = glm::vec3(
                                attrib.normals[3 * idx.normal_index + 0],
                                attrib.normals[3 * idx.normal_index + 1],
                                attrib.normals[3 * idx.normal_index + 2]);

                            nrm = glm::normalize(normalXform * nrm);
                        }

                        if (v == 0) { 
                            tri.v0 = pWorld; 
                            tri.n0 = nrm; 
                        }

                        if (v == 1) { 
                            tri.v1 = pWorld; 
                            tri.n1 = nrm; 
                        }

                        if (v == 2) { 
                            tri.v2 = pWorld; 
                            tri.n2 = nrm; 
                        }

                        bmin = glm::min(bmin, pWorld);
                        bmax = glm::max(bmax, pWorld);
                    }

                    // if OBJ has no normals --> use face normal
                    if (glm::length(tri.n0) < 1e-6f &&
                        glm::length(tri.n1) < 1e-6f &&
                        glm::length(tri.n2) < 1e-6f) {

                        glm::vec3 fn = glm::normalize(glm::cross(tri.v1 - tri.v0, tri.v2 - tri.v0));
                        tri.n0 = tri.n1 = tri.n2 = fn;
                    }

                    mesh.triangles.push_back(tri);
                    index_offset += fv;
                }
            }

            mesh.bboxMin = bmin;
            mesh.bboxMax = bmax;
            meshes.push_back(mesh);

            std::cout << "Loaded mesh " << objPath << " (" << mesh.triangles.size() << " triangles)" << std::endl;
            continue;   // skip the sphere/cube branch
        }

        Geom newGeom;
        if (type == "cube")
        {
            newGeom.type = CUBE;
        }
        else
        {
            newGeom.type = SPHERE;
        }
        newGeom.materialid = MatNameToID[p["MATERIAL"]];
        const auto& trans = p["TRANS"];
        const auto& rotat = p["ROTAT"];
        const auto& scale = p["SCALE"];
        newGeom.translation = glm::vec3(trans[0], trans[1], trans[2]);
        newGeom.rotation = glm::vec3(rotat[0], rotat[1], rotat[2]);
        newGeom.scale = glm::vec3(scale[0], scale[1], scale[2]);
        newGeom.transform = utilityCore::buildTransformationMatrix(
            newGeom.translation, newGeom.rotation, newGeom.scale);
        newGeom.inverseTransform = glm::inverse(newGeom.transform);
        newGeom.invTranspose = glm::inverseTranspose(newGeom.transform);

        geoms.push_back(newGeom);
    }
    const auto& cameraData = data["Camera"];
    Camera& camera = state.camera;
    RenderState& state = this->state;
    camera.resolution.x = cameraData["RES"][0];
    camera.resolution.y = cameraData["RES"][1];
    float fovy = cameraData["FOVY"];
    state.iterations = cameraData["ITERATIONS"];
    state.traceDepth = cameraData["DEPTH"];
    state.imageName = cameraData["FILE"];
    const auto& pos = cameraData["EYE"];
    const auto& lookat = cameraData["LOOKAT"];
    const auto& up = cameraData["UP"];
    camera.position = glm::vec3(pos[0], pos[1], pos[2]);
    camera.lookAt = glm::vec3(lookat[0], lookat[1], lookat[2]);
    camera.up = glm::vec3(up[0], up[1], up[2]);

    // depth of field (both optional)
    if (cameraData.contains("APERTURE")) {
        camera.aperture = cameraData["APERTURE"];
    }
    else {
        camera.aperture = 0.0f; // pinhole (default)
    }

    if (cameraData.contains("FOCAL_DISTANCE")) {
        camera.focalDistance = cameraData["FOCAL_DISTANCE"];
    }
    else {
        camera.focalDistance = glm::length(camera.lookAt - camera.position);
    }

    //calculate fov based on resolution
    float yscaled = tan(fovy * (PI / 180));
    float xscaled = (yscaled * camera.resolution.x) / camera.resolution.y;
    float fovx = (atan(xscaled) * 180) / PI;
    camera.fov = glm::vec2(fovx, fovy);

    camera.right = glm::normalize(glm::cross(camera.view, camera.up));
    camera.pixelLength = glm::vec2(2 * xscaled / (float)camera.resolution.x,
        2 * yscaled / (float)camera.resolution.y);

    camera.view = glm::normalize(camera.lookAt - camera.position);

    //set up render camera stuff
    int arraylen = camera.resolution.x * camera.resolution.y;
    state.image.resize(arraylen);
    std::fill(state.image.begin(), state.image.end(), glm::vec3());
}
