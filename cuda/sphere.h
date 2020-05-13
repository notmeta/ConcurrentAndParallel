#ifndef SPHEREH
#define SPHEREH

#include <random>
#include "hitable.h"

#define MIN_X -15
#define MAX_X 15

#define MIN_Y -18
#define MAX_Y 12

#define MIN_Z -50
#define MAX_Z -5

float random(double minBound, double maxBound) {
    std::random_device rd;
    std::mt19937 mt(rd());
    std::uniform_real_distribution<double> dist(minBound, maxBound);

    return dist(mt);
}

class sphere : public hitable {
public:
    sphere() {
        auto vr = 0.1 * sqrt(random(0, 1)) + 0.05;
        auto vphi = 2 * M_PI * random(0, 1);

        position = vec3(random(MIN_X, MAX_X), random(MIN_Y, MAX_Y), random(MIN_Z, MAX_Z));
        velocity = vec3(vr * cos(vphi), vr * sin(vphi), vr * cos(vphi));
        radius = random(0.5, 2);
        mass = pow(radius, 2);
    };

    __device__ sphere(vec3 pos, vec3 velocity, float r) : position(pos), velocity(velocity), radius(r),
                                                          mass(pow(r, 2)) {

    };

    __device__ virtual bool hit(const ray &r, float tmin, float tmax, hit_record &rec) const;

    __device__ void move(float dt);

    __device__ void boundaryCheck();

    vec3 position;
    vec3 velocity;
    float radius;
    float mass;

};

__device__ bool sphere::hit(const ray &r, float t_min,
                            float t_max, hit_record &rec) const {
    vec3 oc = r.origin() - position;
    float a = dot(r.direction(), r.direction());
    float b = dot(oc, r.direction());
    float c = dot(oc, oc) - radius * radius;
    float discriminant = b * b - a * c;
    if (discriminant > 0) {
        float temp = (-b - sqrt(discriminant)) / a;
        if (temp < t_max && temp > t_min) {
            rec.t = temp;
            rec.p = r.point_at_parameter(rec.t);
            rec.normal = (rec.p - position) / radius;
            return true;
        }
        temp = (-b + sqrt(discriminant)) / a;
        if (temp < t_max && temp > t_min) {
            rec.t = temp;
            rec.p = r.point_at_parameter(rec.t);
            rec.normal = (rec.p - position) / radius;
            return true;
        }
    }
    return false;
}

__device__ void sphere::move(const float dt) {
    this->position += velocity * dt;
    boundaryCheck();
}

__device__ void sphere::boundaryCheck() {
    if (position.x() - radius < MIN_X) {
        position.setX(radius + MIN_X);
        velocity.setX(-velocity.x());
    }
    if (position.x() + radius > MAX_X) {
        position.setX(MAX_X - radius);
        velocity.setX(-velocity.x());
    }
    if (position.y() - radius < MIN_Y) {
        position.setY(radius + MIN_Y);
        velocity.setY(-velocity.y());
    }
    if (position.y() + radius > MAX_Y) {
        position.setY(MAX_Y - radius);
        velocity.setY(-velocity.y());
    }
    if (position.z() - radius < MIN_Z) {
        position.setZ(radius + MIN_Z);
        velocity.setZ(-velocity.z());
    }
    if (position.z() + radius > MAX_Z) {
        position.setZ(MAX_Z - radius);
        velocity.setZ(-velocity.z());
    }
}


#endif