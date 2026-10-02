#include "intersections.h"

__host__ __device__ float boxIntersectionTest(
    Geom box,
    Ray r,
    glm::vec3 &intersectionPoint,
    glm::vec3 &normal,
    bool &outside)
{
    Ray q;
    q.origin    =                multiplyMV(box.inverseTransform, glm::vec4(r.origin   , 1.0f));
    q.direction = glm::normalize(multiplyMV(box.inverseTransform, glm::vec4(r.direction, 0.0f)));

    float tmin = -1e38f;
    float tmax = 1e38f;
    glm::vec3 tmin_n;
    glm::vec3 tmax_n;
    for (int xyz = 0; xyz < 3; ++xyz)
    {
        float qdxyz = q.direction[xyz];
        /*if (glm::abs(qdxyz) > 0.00001f)*/
        {
            float t1 = (-0.5f - q.origin[xyz]) / qdxyz;
            float t2 = (+0.5f - q.origin[xyz]) / qdxyz;
            float ta = glm::min(t1, t2);
            float tb = glm::max(t1, t2);
            glm::vec3 n;
            n[xyz] = t2 < t1 ? +1 : -1;
            if (ta > 0 && ta > tmin)
            {
                tmin = ta;
                tmin_n = n;
            }
            if (tb < tmax)
            {
                tmax = tb;
                tmax_n = n;
            }
        }
    }

    if (tmax >= tmin && tmax > 0)
    {
        outside = true;
        if (tmin <= 0)
        {
            tmin = tmax;
            tmin_n = tmax_n;
            outside = false;
        }
        intersectionPoint = multiplyMV(box.transform, glm::vec4(getPointOnRay(q, tmin), 1.0f));
        normal = glm::normalize(multiplyMV(box.invTranspose, glm::vec4(tmin_n, 0.0f)));
        return glm::length(r.origin - intersectionPoint);
    }

    return -1;
}

__host__ __device__ float sphereIntersectionTest(
    Geom sphere,
    Ray r,
    glm::vec3 &intersectionPoint,
    glm::vec3 &normal,
    bool &outside)
{
    float radius = .5;

    glm::vec3 ro = multiplyMV(sphere.inverseTransform, glm::vec4(r.origin, 1.0f));
    glm::vec3 rd = glm::normalize(multiplyMV(sphere.inverseTransform, glm::vec4(r.direction, 0.0f)));

    Ray rt;
    rt.origin = ro;
    rt.direction = rd;

    float vDotDirection = glm::dot(rt.origin, rt.direction);
    float radicand = vDotDirection * vDotDirection - (glm::dot(rt.origin, rt.origin) - powf(radius, 2));
    if (radicand < 0)
    {
        return -1;
    }

    float squareRoot = sqrt(radicand);
    float firstTerm = -vDotDirection;
    float t1 = firstTerm + squareRoot;
    float t2 = firstTerm - squareRoot;

    float t = 0;
    if (t1 < 0 && t2 < 0)
    {
        return -1;
    }
    else if (t1 > 0 && t2 > 0)
    {
        t = min(t1, t2);
        outside = true;
    }
    else
    {
        t = max(t1, t2);
        outside = false;
    }

    glm::vec3 objspaceIntersection = getPointOnRay(rt, t);

    intersectionPoint = multiplyMV(sphere.transform, glm::vec4(objspaceIntersection, 1.f));
    normal = glm::normalize(multiplyMV(sphere.invTranspose, glm::vec4(objspaceIntersection, 0.f)));
    if (!outside)
    {
        normal = -normal;
    }

    return glm::length(r.origin - intersectionPoint);
}

__host__ __device__ bool bboxIntersectionTest(
    const glm::vec3& bmin,
    const glm::vec3& bmax,
    const Ray& r) 
{
    float tmin = 0.0f;
    float tmax = FLT_MAX;

    for (int i = 0; i < 3; i++) {
        float invD = 1.0f / r.direction[i];
        float t1 = (bmin[i] - r.origin[i]) * invD;
        float t2 = (bmax[i] - r.origin[i]) * invD;

        if (t1 > t2) { 
            float tmp = t1; 
            t1 = t2; 
            t2 = tmp; 
        }

        tmin = glm::max(tmin, t1);
        tmax = glm::min(tmax, t2);

        if (tmin > tmax) {
            return false; 
        }
    }

    return true;
}

__host__ __device__ float triangleIntersectionTest(
    const Triangle& tri,
    const Ray& r,
    glm::vec3& intersectionPoint,
    glm::vec3& normal,
    bool& outside)
{
    const float EPS = 1e-7f;

    glm::vec3 edge1 = tri.v1 - tri.v0;
    glm::vec3 edge2 = tri.v2 - tri.v0;

    // determinant
    glm::vec3 pvec = glm::cross(r.direction, edge2);
    float det = glm::dot(edge1, pvec);

    // ray parallel to triangle
    if (fabsf(det) < EPS) {
        return -1.0f;
    }

    float invDet = 1.0f / det;

    // dist from v0 --> ray origin
    glm::vec3 tvec = r.origin - tri.v0;

    // u parameter
    float u = glm::dot(tvec, pvec) * invDet;
    if (u < 0.0f || u > 1.0f) {
        return -1.0f;
    }

    // v parameter
    glm::vec3 qvec = glm::cross(tvec, edge1);
    float v = glm::dot(r.direction, qvec) * invDet;
    if (v < 0.0f || u + v > 1.0f) {
        return -1.0f;
    }

    // t along the ray
    float t = glm::dot(edge2, qvec) * invDet;
    if (t < EPS) {
        return -1.0f;
    }

    intersectionPoint = r.origin + t * r.direction;

    // barycentric interpolation of vertex normals
    float w = 1.0f - u - v;
    glm::vec3 n = w * tri.n0 + u * tri.n1 + v * tri.n2;

    // fallback to geometric normal if vertex normals are missing
    if (glm::dot(n, n) < 1e-12f) {
        n = glm::cross(edge1, edge2);
    }
    normal = glm::normalize(n);

    // flip normal if ray hit back face (closed meshes)
    if (glm::dot(normal, r.direction) > 0.0f) {
        normal = -normal;
        outside = false;
    } else {
        outside = true;
    }

    return t;
}

__host__ __device__ float intersectMeshBVH(
    const BVHNode* __restrict__ nodes,
    const Triangle* __restrict__ tris,
    const Ray& r,
    glm::vec3& outNormal,
    bool& outOutside)
{
    float t_min = FLT_MAX;

    // 64 entries for balanced trees 
    int stack[64];
    int sp = 0;
    stack[sp++] = 0;   // root

    while (sp > 0) {
        int nodeIdx = stack[--sp];
        const BVHNode& node = nodes[nodeIdx];

        // Ray-AABB reject
        if (!bboxIntersectionTest(node.bboxMin, node.bboxMax, r)) {
            continue;
        }

        if (node.triCount == 0) {
            // interior --> push both children
            stack[sp++] = node.leftFirst;
            stack[sp++] = node.leftFirst + 1;
        } else {
            // leaf --> test all triangles in range
            for (int i = 0; i < node.triCount; ++i) {
                const Triangle& tri = tris[node.leftFirst + i];
                glm::vec3 tmpP, tmpN;
                bool tmpO;
                float t = triangleIntersectionTest(tri, r, tmpP, tmpN, tmpO);

                if (t > 0.0f && t < t_min) {
                    t_min = t;
                    outNormal = tmpN;
                    outOutside = tmpO;
                }
            }
        }
    }

    return (t_min < FLT_MAX) ? t_min : -1.0f;
}