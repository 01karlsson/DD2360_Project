#ifndef VEC3H
#define VEC3H

#include <math.h>
#include <stdlib.h>
#include <iostream>

#define dataType float

class vec3  {


public:
    __host__ __device__ vec3() {}
    __host__ __device__ vec3(dataType e0, dataType e1, dataType e2) { e[0] = e0; e[1] = e1; e[2] = e2; }
    __host__ __device__ inline dataType x() const { return e[0]; }
    __host__ __device__ inline dataType y() const { return e[1]; }
    __host__ __device__ inline dataType z() const { return e[2]; }
    __host__ __device__ inline dataType r() const { return e[0]; }
    __host__ __device__ inline dataType g() const { return e[1]; }
    __host__ __device__ inline dataType b() const { return e[2]; }

    __host__ __device__ inline const vec3& operator+() const { return *this; }
    __host__ __device__ inline vec3 operator-() const { return vec3(-e[0], -e[1], -e[2]); }
    __host__ __device__ inline dataType operator[](int i) const { return e[i]; }
    __host__ __device__ inline dataType& operator[](int i) { return e[i]; };

    __host__ __device__ inline vec3& operator+=(const vec3 &v2);
    __host__ __device__ inline vec3& operator-=(const vec3 &v2);
    __host__ __device__ inline vec3& operator*=(const vec3 &v2);
    __host__ __device__ inline vec3& operator/=(const vec3 &v2);
    __host__ __device__ inline vec3& operator*=(const dataType t);
    __host__ __device__ inline vec3& operator/=(const dataType t);

    __host__ __device__ inline dataType length() const { return sqrt(e[0]*e[0] + e[1]*e[1] + e[2]*e[2]); }
    __host__ __device__ inline dataType squared_length() const { return e[0]*e[0] + e[1]*e[1] + e[2]*e[2]; }
    __host__ __device__ inline void make_unit_vector();


    dataType e[3];
};



inline std::istream& operator>>(std::istream &is, vec3 &t) {
    if(abs(t.e[0])<65504||abs(t.e[1])<65504||abs(t.e[2])<65504){//notera vänt tecken
      is >> t.e[0] >> t.e[1] >> t.e[2];
    }

    return is;
}

inline std::ostream& operator<<(std::ostream &os, const vec3 &t) {
    if(abs(t.e[0])>65504||abs(t.e[1])>65504||abs(t.e[2])>65504){
      os<<"00 00 00";
    } else{    os << t.e[0] << " " << t.e[1] << " " << t.e[2];}
    return os;
}

__host__ __device__ inline void vec3::make_unit_vector() {
    if(abs(e[0])>65504||abs(e[1])>65504||abs(e[2])>65504){
      vec3 e= vec3(0,0,0);
    }
    float k = 1.0 / sqrt(e[0]*e[0] + e[1]*e[1] + e[2]*e[2]);
    e[0] *= k; e[1] *= k; e[2] *= k;
}

__host__ __device__ inline vec3 operator+(const vec3 &v1, const vec3 &v2) {
    vec3 e = vec3(v1.e[0] + v2.e[0], v1.e[1] + v2.e[1], v1.e[2] + v2.e[2]);
    if(abs(e[0])>65504||abs(e[1])>65504||abs(e[2])>65504){
      vec3 e= vec3(0,0,0);
    }
    return e;
}

__host__ __device__ inline vec3 operator-(const vec3 &v1, const vec3 &v2) {
    vec3 e = vec3(v1.e[0] - v2.e[0], v1.e[1] - v2.e[1], v1.e[2] - v2.e[2]);
    if(abs(e[0])>65504||abs(e[1])>65504||abs(e[2])>65504){
      vec3 e= vec3(0,0,0);
    }
    return e;
}

__host__ __device__ inline vec3 operator*(const vec3 &v1, const vec3 &v2) {
    vec3 e = vec3(v1.e[0] * v2.e[0], v1.e[1] * v2.e[1], v1.e[2] * v2.e[2]);
    if(abs(e[0])>65504||abs(e[1])>65504||abs(e[2])>65504){
      vec3 e= vec3(0,0,0);
    }
    return e;
}

__host__ __device__ inline vec3 operator/(const vec3 &v1, const vec3 &v2) {
    vec3 e = vec3(v1.e[0] / v2.e[0], v1.e[1] / v2.e[1], v1.e[2] / v2.e[2]);
    if(abs(e[0])>65504||abs(e[1])>65504||abs(e[2])>65504){
      vec3 e= vec3(0,0,0);
    }
    return e;
}

__host__ __device__ inline vec3 operator*(dataType t, const vec3 &v) {
    vec3 e = vec3(t*v.e[0], t*v.e[1], t*v.e[2]);
    if(abs(e[0])>65504||abs(e[1])>65504||abs(e[2])>65504){
      vec3 e= vec3(0,0,0);
    }
    return e;
}

__host__ __device__ inline vec3 operator/(vec3 v, dataType t) {
    vec3 e = vec3(v.e[0]/t, v.e[1]/t, v.e[2]/t);
    if(abs(e[0])>65504||abs(e[1])>65504||abs(e[2])>65504){
      vec3 e= vec3(0,0,0);
    }
    return e;
}

__host__ __device__ inline vec3 operator*(const vec3 &v, dataType t) {
    vec3 e = vec3(t*v.e[0], t*v.e[1], t*v.e[2]);
    if(abs(e[0])>65504||abs(e[1])>65504||abs(e[2])>65504){
      vec3 e= vec3(0,0,0);
    }
    return e;
}

__host__ __device__ inline dataType dot(const vec3 &v1, const vec3 &v2) {
    dataType e = v1.e[0] *v2.e[0] + v1.e[1] *v2.e[1]  + v1.e[2] *v2.e[2];
    if(abs(e)>65504){
      e= 0.0f;
    }
    return e;
}

__host__ __device__ inline vec3 cross(const vec3 &v1, const vec3 &v2) {
    vec3 e = vec3( (v1.e[1]*v2.e[2] - v1.e[2]*v2.e[1]),
                (-(v1.e[0]*v2.e[2] - v1.e[2]*v2.e[0])),
                (v1.e[0]*v2.e[1] - v1.e[1]*v2.e[0]));
    if(abs(e[0])>65504||abs(e[1])>65504||abs(e[2])>65504){
      vec3 e= vec3(0,0,0);
    }
    return e;
}


__host__ __device__ inline vec3& vec3::operator+=(const vec3 &v){
    e[0]  += v.e[0];
    e[1]  += v.e[1];
    e[2]  += v.e[2];
    if(abs(e[0])>65504||abs(e[1])>65504||abs(e[2])>65504){
      vec3 e= vec3(0,0,0);
    }
    return *this;
}

__host__ __device__ inline vec3& vec3::operator*=(const vec3 &v){
    e[0]  *= v.e[0];
    e[1]  *= v.e[1];
    e[2]  *= v.e[2];
    if(abs(e[0])>65504||abs(e[1])>65504||abs(e[2])>65504){
      vec3 e= vec3(0,0,0);
    }
    return *this;
}

__host__ __device__ inline vec3& vec3::operator/=(const vec3 &v){
    e[0]  /= v.e[0];
    e[1]  /= v.e[1];
    e[2]  /= v.e[2];
    if(abs(e[0])>65504||abs(e[1])>65504||abs(e[2])>65504){
      vec3 e= vec3(0,0,0);
    }
    return *this;
}

__host__ __device__ inline vec3& vec3::operator-=(const vec3& v) {
    e[0]  -= v.e[0];
    e[1]  -= v.e[1];
    e[2]  -= v.e[2];
    if(abs(e[0])>65504||abs(e[1])>65504||abs(e[2])>65504){
      vec3 e= vec3(0,0,0);
    }
    return *this;
}

__host__ __device__ inline vec3& vec3::operator*=(const dataType t) {
    e[0]  *= t;
    e[1]  *= t;
    e[2]  *= t;
    if(abs(e[0])>65504||abs(e[1])>65504||abs(e[2])>65504){
      vec3 e= vec3(0,0,0);
    }
    return *this;
}

__host__ __device__ inline vec3& vec3::operator/=(const dataType t) {
    dataType k = 1.0/t;

    e[0]  *= k;
    e[1]  *= k;
    e[2]  *= k;
    if(abs(e[0])>65504||abs(e[1])>65504||abs(e[2])>65504){
      vec3 e= vec3(0,0,0);
    }
    return *this;
}

__host__ __device__ inline vec3 unit_vector(vec3 v) {
    return v / v.length();
}

#endif
