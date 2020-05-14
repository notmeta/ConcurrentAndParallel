#ifndef SPHEREH
#define SPHEREH

#include <random>
#include "hitable.h"

#define MIN_X -20
#define MAX_X 20

#define MIN_Y -25
#define MAX_Y 17

#define MIN_Z -40
#define MAX_Z -10

#define FIRE_MODE true

#define CENTER_OF_MASS vec3((MIN_X + MAX_X) / 2, (MIN_Y + MAX_Y) / 2, (MIN_Z + MAX_Z) / 2)

float random(double minBound, double maxBound) {
    std::random_device rd;
    std::mt19937 mt(rd());
    std::uniform_real_distribution<double> dist(minBound, maxBound);

    return dist(mt);
}

enum ColourMode {
    SOLID,
    VELOCITY,
    DISTANCE
};

class sphere {
public:
    sphere() {
        auto vr = 0.1 * sqrt(random(0, 3)) + 0.05;
        auto vphi = 2 * M_PI * random(0, 3);

        position = vec3(random(MIN_X, MAX_X), random(MIN_Y, MAX_Y), random(MIN_Z, MAX_Z));
        velocity = vec3(vr * cos(vphi), vr * sin(vphi), vr * cos(vphi));
        colour = make_uchar4(random(0, 255), random(0, 255), random(0, 255), 0);
        radius = random(0.1, 1.3);
//        radius = 0.1;
        mass = pow(radius, 2);
    };

    __device__ bool hit(const ray &r, float tmin, float tmax, hit_record &rec) const;

    __device__ void move(float dt);

    __device__ void boundaryCheck();

    __device__ bool overlaps(sphere *other) const;

    __device__ void changeVelocities(sphere *other);

    __device__ uchar4 getColour(ColourMode mode) const;

    vec3 position;
    vec3 velocity;
    bool updated = false;
    uchar4 colour{};
    float radius;
    float mass;

};

__device__ bool sphere::hit(const ray &r, float t_min, float t_max, hit_record &rec) const {
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
    updated = false;
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

__device__ void sphere::changeVelocities(sphere *other) {
    auto combinedMass = mass + other->mass;
    auto norm = linAlgNorm(position - other->position);
    auto d = norm * norm;

    auto u1 = velocity -
              vec3(2, 2, 2) * other->mass / combinedMass * dot(velocity - other->velocity, position - other->position) /
              d * (position - other->position);
    auto u2 = other->velocity -
              vec3(2, 2, 2) * mass / combinedMass * dot(other->velocity - velocity, other->position - position) / d *
              (other->position - position);

    velocity = u1;
    other->velocity = u2;

    updated = true;
    other->updated = true;
}

__device__ bool sphere::overlaps(sphere *other) const {
    auto rSquared = radius + other->radius;
    rSquared *= rSquared;

    auto delta = other->position - position;

    return dot(delta, delta) < rSquared;
}

__device__ uchar4 sphere::getColour(ColourMode mode) const {
    auto vCol = min((abs(velocity.x() * 2) + abs(velocity.y() * 2) + abs(velocity.z() * 2)) * 250, 255.0);
    vCol = max(vCol, 10.0);

    auto dist = min(255, max((int) distance(position, CENTER_OF_MASS) * 11, 0));
    auto distanceColour = abs(dist - 255);

    switch (mode) {
        case VELOCITY:
            return make_uchar4(vCol, vCol, vCol, 0);
        case DISTANCE:
            if (distanceColour >= 150 && FIRE_MODE) {
                return make_uchar4(0, 265 - pow((255 - distanceColour), 1.15), 255, 0);
            } else {
                return make_uchar4(distanceColour, distanceColour, distanceColour, 0);
            }
        case SOLID:
        default:
            return colour;
    }
}


#endif