#include "interactions.h"

#include "utilities.h"

#include <thrust/random.h>

__host__ __device__ glm::vec3 calculateRandomDirectionInHemisphere(
    glm::vec3 normal,
    thrust::default_random_engine &rng)
{
    thrust::uniform_real_distribution<float> u01(0, 1);

    float up = sqrt(u01(rng)); // cos(theta)
    float over = sqrt(1 - up * up); // sin(theta)
    float around = u01(rng) * TWO_PI;

    // Find a direction that is not the normal based off of whether or not the
    // normal's components are all equal to sqrt(1/3) or whether or not at
    // least one component is less than sqrt(1/3). Learned this trick from
    // Peter Kutz.

    glm::vec3 directionNotNormal;
    if (abs(normal.x) < SQRT_OF_ONE_THIRD)
    {
        directionNotNormal = glm::vec3(1, 0, 0);
    }
    else if (abs(normal.y) < SQRT_OF_ONE_THIRD)
    {
        directionNotNormal = glm::vec3(0, 1, 0);
    }
    else
    {
        directionNotNormal = glm::vec3(0, 0, 1);
    }

    // Use not-normal direction to generate two perpendicular directions
    glm::vec3 perpendicularDirection1 =
        glm::normalize(glm::cross(normal, directionNotNormal));
    glm::vec3 perpendicularDirection2 =
        glm::normalize(glm::cross(normal, perpendicularDirection1));

    return up * normal
        + cos(around) * over * perpendicularDirection1
        + sin(around) * over * perpendicularDirection2;
}

__host__ __device__ void scatterRay(
    PathSegment & pathSegment,
    glm::vec3 intersect,
    glm::vec3 normal,
    const Material &m,
    bool outside,
    thrust::default_random_engine &rng)
{
    // TODO: implement this.
    // A basic implementation of pure-diffuse shading will just call the
    // calculateRandomDirectionInHemisphere defined above.
    
    const float EPS = 0.001f;

    if (m.hasRefractive > 0.f) {    // refractive (fresnel-split reflection/refraction)
        float ior = m.indexOfRefraction;

        // eta = n1 / n2
        float eta = outside ? (1.0f / ior) : ior;

        glm::vec3 incident = glm::normalize(pathSegment.ray.direction);
        glm::vec3 refracted = glm::refract(incident, normal, eta);

        // total internal reflection
        bool tir = (glm::dot(refracted, refracted) < 1e-6f);

        // schilck approx. (fresnel reflectance)
        float cosTheta = glm::clamp(glm::dot(-incident, normal), 0.0f, 1.0f);
        float F0 = (1.0f - ior) / (1.0f + ior);
        F0 = F0 * F0;
        float fresnel = F0 + (1.0f - F0) * powf(1.0f - cosTheta, 5.0f);

        thrust::uniform_real_distribution<float> u01(0, 1);
        float xi = u01(rng);

        const float MAX_MULT = 2.0f;    // tune 2.0 - 8.0

        if (tir || xi < fresnel) {  // reflect
            // reflect 
            glm::vec3 reflected = glm::reflect(incident, normal);
            pathSegment.ray.origin = intersect + normal * EPS;
            pathSegment.ray.direction = glm::normalize(reflected);

            float p = tir ? 1.0f : fresnel;
            float mult = glm::min(1.0f / p, MAX_MULT);
            pathSegment.color *= mult;
        }
        else {  // refract
            refracted = glm::normalize(refracted);
            pathSegment.ray.origin = intersect + refracted * EPS;
            pathSegment.ray.direction = refracted;

            float p = 1.0f - fresnel;
            float mult = glm::min(1.0f / p, MAX_MULT);
			pathSegment.color *= m.color * mult;
        }
    }
    else if (m.hasReflective > 0.f) {   // specular (perfect mirror)
        glm::vec3 incident = glm::normalize(pathSegment.ray.direction);
        glm::vec3 reflected = glm::reflect(incident, normal);


        glm::vec3 specColor = (m.specular.color.x + m.specular.color.y + m.specular.color.z > 0.0f)
            ? m.specular.color
            : m.color;

        pathSegment.ray.origin = intersect + normal * EPS;
        pathSegment.ray.direction = glm::normalize(reflected);
        pathSegment.color *= specColor;

    } else {  // diffuse (lambertian)
        glm::vec3 newDir = calculateRandomDirectionInHemisphere(normal, rng);

        pathSegment.ray.origin = intersect + normal * EPS;
        pathSegment.ray.direction = normalize(newDir);
        pathSegment.color *= m.color;
    }
    

    pathSegment.remainingBounces--;
}
