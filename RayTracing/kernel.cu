#include <cuda_runtime.h>
#include <curand_kernel.h>

#define MAX_SPHERES 10
#define MAX_LIGHTS 2
#define MAX_DEPTH 5
#define EPSILON 0.0001f

struct Sphere {
    float3 center;
    float radius;
    float3 color;
    float shininess;
};

struct Light {
    float3 position;
    float3 color;
    float intensity;
};

struct Ray {
    float3 origin;
    float3 direction;
};

__device__ float3 operator+(const float3& a, const float3& b) {
    return make_float3(a.x + b.x, a.y + b.y, a.z + b.z);
}

__device__ float3 operator-(const float3& a, const float3& b) {
    return make_float3(a.x - b.x, a.y - b.y, a.z - b.z);
}

__device__ float3 operator*(const float3& a, float b) {
    return make_float3(a.x * b, a.y * b, a.z * b);
}

__device__ float3 operator*(const float3& a, const float3& b) {
    return make_float3(a.x * b.x, a.y * b.y, a.z * b.z);
}

__device__ float dot(const float3& a, const float3& b) {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

__device__ float3 normalize(const float3& v) {
    float inv_len = rsqrtf(dot(v, v));
    return v * inv_len;
}

__device__ float3 reflect(const float3& v, const float3& n) {
    return v - n * (2.0f * dot(v, n));
}

__device__ bool intersectSphere(const Ray& ray, const Sphere& sphere, float& t) {
    float3 oc = ray.origin - sphere.center;
    float a = dot(ray.direction, ray.direction);
    float b = 2.0f * dot(oc, ray.direction);
    float c = dot(oc, oc) - sphere.radius * sphere.radius;
    float discriminant = b * b - 4 * a * c;
    
    if (discriminant < 0) return false;
    
    float t0 = (-b - sqrtf(discriminant)) / (2.0f * a);
    float t1 = (-b + sqrtf(discriminant)) / (2.0f * a);
    
    if (t0 > EPSILON) {
        t = t0;
        return true;
    }
    if (t1 > EPSILON) {
        t = t1;
        return true;
    }
    return false;
}

__device__ float3 trace(Ray ray, Sphere* spheres, int num_spheres, Light* lights, int num_lights, int depth) {
    if (depth >= MAX_DEPTH) return make_float3(0.0f, 0.0f, 0.0f);
    
    float closest_t = INFINITY;
    int closest_sphere = -1;
    
    for (int i = 0; i < num_spheres; i++) {
        float t;
        if (intersectSphere(ray, spheres[i], t)) {
            if (t < closest_t) {
                closest_t = t;
                closest_sphere = i;
            }
        }
    }
    
    if (closest_sphere == -1) return make_float3(0.2f, 0.3f, 0.5f); // Sky color
    
    Sphere sphere = spheres[closest_sphere];
    float3 hit_point = ray.origin + ray.direction * closest_t;
    float3 normal = normalize(hit_point - sphere.center);
    
    float3 color = make_float3(0.0f, 0.0f, 0.0f);
    
    for (int i = 0; i < num_lights; i++) {
        float3 light_dir = normalize(lights[i].position - hit_point);
        
        // Shadow check
        Ray shadow_ray;
        shadow_ray.origin = hit_point + normal * EPSILON;
        shadow_ray.direction = light_dir;
        bool in_shadow = false;
        
        for (int j = 0; j < num_spheres; j++) {
            float t;
            if (j != closest_sphere && intersectSphere(shadow_ray, spheres[j], t)) {
                in_shadow = true;
                break;
            }
        }
        
        if (!in_shadow) {
            float diff = fmaxf(dot(normal, light_dir), 0.0f);
            float3 reflect_dir = reflect(-light_dir, normal);
            float spec = powf(fmaxf(dot(-ray.direction, reflect_dir), 0.0f), sphere.shininess);
            
            color = color + sphere.color * lights[i].color * (diff * 0.8f + spec * 0.2f) * lights[i].intensity;
        }
    }
    
    // Reflection
    if (depth < MAX_DEPTH) {
        Ray reflect_ray;
        reflect_ray.origin = hit_point + normal * EPSILON;
        reflect_ray.direction = reflect(ray.direction, normal);
        float3 reflect_color = trace(reflect_ray, spheres, num_spheres, lights, num_lights, depth + 1);
        color = color * 0.8f + reflect_color * 0.2f;
    }
    
    return color;
}

extern "C" __global__ void setupRandom(curandState *state) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    curand_init(1234, idx, 0, &state[idx]);
}

extern "C" __global__ void render(uchar4* output, int width, int height, 
                                Sphere* spheres, int num_spheres,
                                Light* lights, int num_lights) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (x >= width || y >= height) return;
    
    float u = (float)x / (float)width;
    float v = (float)y / (float)height;
    
    Ray ray;
    ray.origin = make_float3(0.0f, 0.0f, -5.0f);
    ray.direction = normalize(make_float3(
        (2.0f * u - 1.0f) * ((float)width / (float)height),
        1.0f - 2.0f * v,
        1.0f
    ));
    
    float3 color = trace(ray, spheres, num_spheres, lights, num_lights, 0);
    
    // Tone mapping and gamma correction
    color.x = fminf(color.x, 1.0f);
    color.y = fminf(color.y, 1.0f);
    color.z = fminf(color.z, 1.0f);
    
    color.x = powf(color.x, 1.0f/2.2f);
    color.y = powf(color.y, 1.0f/2.2f);
    color.z = powf(color.z, 1.0f/2.2f);
    
    output[y * width + x] = make_uchar4(
        (unsigned char)(color.x * 255.0f),
        (unsigned char)(color.y * 255.0f),
        (unsigned char)(color.z * 255.0f),
        255
    );
}