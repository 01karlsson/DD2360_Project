#ifndef MATERIALH
#define MATERIALH

struct hit_record;

#include "ray.h"
#include "hitable.h"


__device__ float schlick(float cosine, float ref_idx) {
    float r0 = (1.0f-ref_idx) / (1.0f+ref_idx);
    r0 = r0*r0;
    return r0 + (1.0f-r0)*pow((1.0f - cosine),5.0f);
}

__device__ bool refract(const vec3& v, const vec3& n, float ni_over_nt, vec3& refracted) {
    vec3 uv = unit_vector(v);
    float dt = dot(uv, n);
    float discriminant = 1.0f - ni_over_nt*ni_over_nt*(1.0f-dt*dt);
    if (discriminant > 0.0f) {
        refracted = ni_over_nt*(uv - n*dt) - n*sqrtf(discriminant);
        return true;
    }
    else
        return false;
}

#define RANDVEC3 vec3(curand_uniform(local_rand_state),curand_uniform(local_rand_state),curand_uniform(local_rand_state))

__device__ vec3 random_in_unit_sphere(curandState *local_rand_state) {
    vec3 p;
    do {
        float r1 = curand_uniform(local_rand_state);
        float r2 = curand_uniform(local_rand_state);
        float r3 = curand_uniform(local_rand_state);
        p = 2.0f*vec3(r1, r2, r3) - vec3(1.0f, 1.0f, 1.0f);
    } while (__half2float(p.squared_length()) >= 1.0f);
    return p;
}

__device__ vec3 reflect(const vec3& v, const vec3& n) {
    float dot_vn = dot(v, n);
    return v - 2.0f*dot_vn*n;
}

class material  {
    public:
        __device__ virtual bool scatter(const ray& r_in, const hit_record& rec, vec3& attenuation, ray& scattered, curandState *local_rand_state) const = 0;
};

class lambertian : public material {
    public:
        __device__ lambertian(const vec3& a) : albedo(a) {}
        __device__ virtual bool scatter(const ray& r_in, const hit_record& rec, vec3& attenuation, ray& scattered, curandState *local_rand_state) const  {
             vec3 target = rec.p + rec.normal + random_in_unit_sphere(local_rand_state);
             scattered = ray(rec.p, target-rec.p);
             attenuation = albedo;
             return true;
        }

        vec3 albedo;
};

class metal : public material {
    public:
        __device__ metal(const vec3& a, float f) : albedo(a) { if (f < 1) fuzz = f; else fuzz = 1; }
        __device__ virtual bool scatter(const ray& r_in, const hit_record& rec, vec3& attenuation, ray& scattered, curandState *local_rand_state) const  {
            vec3 reflected = reflect(unit_vector(r_in.direction()), rec.normal);
            scattered = ray(rec.p, reflected + fuzz*random_in_unit_sphere(local_rand_state));
            attenuation = albedo;
            return (dot(scattered.direction(), rec.normal) > 0.0f);
        }
        vec3 albedo;
        float fuzz;
};

class dielectric : public material {
public:
    __device__ dielectric(float ri) : ref_idx(ri) {}
    __device__ virtual bool scatter(const ray& r_in,
                         const hit_record& rec,
                         vec3& attenuation,
                         ray& scattered,
                         curandState *local_rand_state) const  {
        vec3 outward_normal;
        vec3 reflected = reflect(r_in.direction(), rec.normal);
        float ni_over_nt;
        attenuation = vec3(1.0f, 1.0f, 1.0f);
        vec3 refracted;
        float reflect_prob;
        float cosine;
        float dot_dir_normal = dot(r_in.direction(), rec.normal);
        if (dot_dir_normal > 0.0f) {
            outward_normal = -rec.normal;
            ni_over_nt = ref_idx;
            float dir_len = __half2float(r_in.direction().length());
            cosine = dot_dir_normal / dir_len;
            cosine = sqrtf(1.0f - ref_idx*ref_idx*(1.0f-cosine*cosine));
        }
        else {
            outward_normal = rec.normal;
            ni_over_nt = 1.0f / ref_idx;
            float dir_len = __half2float(r_in.direction().length());
            cosine = -dot_dir_normal / dir_len;
        }
        if (refract(r_in.direction(), outward_normal, ni_over_nt, refracted))
            reflect_prob = schlick(cosine, ref_idx);
        else
            reflect_prob = 1.0f;
        if (curand_uniform(local_rand_state) < reflect_prob)
            scattered = ray(rec.p, reflected);
        else
            scattered = ray(rec.p, refracted);
        return true;
    }

    float ref_idx;
};
#endif
