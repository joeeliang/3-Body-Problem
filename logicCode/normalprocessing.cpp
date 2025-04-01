#define FMT_HEADER_ONLY
#include <vector>
#include <cmath>
#include <array>
#include <limits>
#include <sstream>
#include <algorithm>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <chrono>
#include <fmt/core.h>

class Body {
public:
    double mass;
    std::array<double, 2> position;
    std::array<double, 2> velocity;

    Body(double m, std::array<double, 2> pos, std::array<double, 2> vel)
        : mass(m), position(pos), velocity(vel) {}

    std::array<double, 2> compute_acceleration(const std::vector<Body>& other_bodies, double G = 1.0, double softening = 0.2) const {
        std::array<double, 2> acceleration = {0.0, 0.0};
        for (const auto& other : other_bodies) {
            if (&other != this) {
                std::array<double, 2> r = {other.position[0] - position[0], other.position[1] - position[1]};
                double distance = std::sqrt(r[0]*r[0] + r[1]*r[1]);
                double factor = G * other.mass / std::max(std::pow(distance, 3), std::pow(softening, 3));
                acceleration[0] += factor * r[0];
                acceleration[1] += factor * r[1];
            }
        }
        return acceleration;
    }

    std::array<double, 4> get_state() const {
        return {position[0], position[1], velocity[0], velocity[1]};
    }
};

class System {
private:
    double G;
    std::vector<Body> bodies;

public:
    System(double g = 1.0) : G(g) {}

    void set_state(const std::vector<double>& state) {
        bodies.clear();
        for (size_t i = 0; i < state.size(); i += 4) {
            bodies.emplace_back(1.0, 
                std::array<double, 2>{state[i], state[i+1]},
                std::array<double, 2>{state[i+2], state[i+3]});
        }
    }

    std::vector<std::array<double, 2>> compute_accelerations() {
        std::vector<std::array<double, 2>> accelerations;
        accelerations.reserve(bodies.size());
        for (const auto& body : bodies) {
            accelerations.push_back(body.compute_acceleration(bodies, G));
        }
        return accelerations;
    }

    double compute_total_energy() const {
        double kinetic_energy = 0.0;
        double potential_energy = 0.0;

        for (const auto& body : bodies) {
            kinetic_energy += 0.5 * body.mass * (body.velocity[0]*body.velocity[0] + body.velocity[1]*body.velocity[1]);
        }

        for (size_t i = 0; i < bodies.size(); ++i) {
            for (size_t j = i + 1; j < bodies.size(); ++j) {
                double dx = bodies[j].position[0] - bodies[i].position[0];
                double dy = bodies[j].position[1] - bodies[i].position[1];
                double distance = std::sqrt(dx*dx + dy*dy);
                potential_energy -= G * bodies[i].mass * bodies[j].mass / distance;
            }
        }

        return kinetic_energy + potential_energy;
    }

    std::vector<double> get_state() const {
        std::vector<double> state;
        state.reserve(bodies.size() * 4);
        for (const auto& body : bodies) {
            auto body_state = body.get_state();
            state.insert(state.end(), body_state.begin(), body_state.end());
        }
        return state;
    }

    std::vector<double> integrate(double dt, int num_steps, bool save_positions = false) {
        std::vector<std::vector<std::array<double, 2>>> positions;
        std::vector<double> delta_energy;

        if (save_positions) {
            positions.resize(num_steps, std::vector<std::array<double, 2>>(bodies.size()));
            delta_energy.resize(num_steps);
            double initial_energy = compute_total_energy();

            for (int step = 0; step < num_steps; ++step) {
                auto accelerations = compute_accelerations();

                for (size_t i = 0; i < bodies.size(); ++i) {
                    for (int j = 0; j < 2; ++j) {
                        bodies[i].position[j] += bodies[i].velocity[j] * dt + 0.5 * accelerations[i][j] * dt * dt;
                    }
                }

                auto new_accelerations = compute_accelerations();

                for (size_t i = 0; i < bodies.size(); ++i) {
                    for (int j = 0; j < 2; ++j) {
                        bodies[i].velocity[j] += 0.5 * (accelerations[i][j] + new_accelerations[i][j]) * dt;
                    }
                    positions[step][i] = bodies[i].position;
                }

                delta_energy[step] = initial_energy - compute_total_energy();
            }

            return delta_energy;
        } else {
            for (int step = 0; step < num_steps; ++step) {
                auto accelerations = compute_accelerations();

                for (size_t i = 0; i < bodies.size(); ++i) {
                    for (int j = 0; j < 2; ++j) {
                        bodies[i].position[j] += bodies[i].velocity[j] * dt + 0.5 * accelerations[i][j] * dt * dt;
                    }
                }

                auto new_accelerations = compute_accelerations();

                for (size_t i = 0; i < bodies.size(); ++i) {
                    for (int j = 0; j < 2; ++j) {
                        bodies[i].velocity[j] += 0.5 * (accelerations[i][j] + new_accelerations[i][j]) * dt;
                    }
                }
            }

            return get_state();
        }
    }
    double proximity(const std::vector<double>& stan_state, double total_time = 30, double delta_t = 0.01) {
        int num_steps = static_cast<int>(total_time / delta_t);
        std::vector<double> initial_state = stan_state;
        set_state(stan_state);

        double min_distance = std::numeric_limits<double>::infinity();
        double di = 0.0; // initial distance

        for (int step = 0; step < num_steps; ++step) {
            std::vector<double> new_state = integrate(delta_t, 1);
            
            // Calculate Euclidean distance
            double df = 0.0;
            for (size_t i = 0; i < new_state.size(); ++i) {
                double diff = new_state[i] - initial_state[i];
                df += diff * diff;
            }
            df = std::sqrt(df);

            if (df < di && df < min_distance) {
                min_distance = df;
            }
            di = df;
        }

        if (std::isinf(min_distance)) {
            min_distance = di;
        }

        return min_distance;
    }
};


std::vector<double> make_state(double d, double vx, double vy) {
    return {0, 0, vx, vy, d, 0, -vx/2, -vy/2, -d, 0, -vx/2, -vy/2};
}

void heatmap_batch(System& system, double (*quantifier)(System&, const std::vector<double>&, double, double),
                   int start_j, int end_j, int grid_size, double vx_start, double vy_start, double vx_step, double vy_step, const std::string& outfile_path) {
    std::ofstream outfile(outfile_path, std::ios::app);
    
    outfile << std::fixed << std::setprecision(16);

    int total_iterations = (end_j - start_j) * grid_size;
    int current_iteration = 0;
    
    auto start_time = std::chrono::high_resolution_clock::now();
    
    for (int j = start_j; j < end_j; ++j) {
        double vy = vy_start + j * vy_step;
        for (int i = 0; i < grid_size; ++i) {
            double vx = vx_start + i * vx_step;
            
            std::vector<double> initial_state = make_state(1, vx, vy);
            double proximity = quantifier(system, initial_state, 30, 0.01);
            
            outfile << "1," << vy << "," << vx << "," << proximity << "\n";
            
            ++current_iteration;
            if (current_iteration % 10 == 0) {
                auto current_time = std::chrono::high_resolution_clock::now();
                auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(current_time - start_time).count();
                double progress = static_cast<double>(current_iteration) / total_iterations;
                int eta = static_cast<int>((elapsed / progress) - elapsed);
                
                std::cout << "\rBatch Progress: " << std::fixed << std::setprecision(2) << (progress * 100)
                          << "% | ETA: " << eta << "s" << std::flush;
            }
        }
    }
    
    outfile.close();
}

void heatmap(System& system, double (*quantifier)(System&, const std::vector<double>&, double, double),
             double vx_start, double vx_end, double vy_start, double vy_end, int dimension) {
    
    const int grid_size = dimension;
    const double vx_step = (vx_end - vx_start) / (grid_size - 1);
    const double vy_step = (vy_end - vy_start) / (grid_size - 1);
    
    // Create the CSV file and write the header
    std::string outfile_path = fmt::format("enhanced/proximity{}_{}.csv", vx_start, vx_end);
    std::ofstream outfile(outfile_path);
    outfile << "d,vy,vx,proximity\n";
    outfile.close();
    
    const int batch_size = 20; // Process 20 rows at a time
    int num_batches = (grid_size + batch_size - 1) / batch_size;
    
    for (int batch = 0; batch < num_batches; ++batch) {
        int start_j = batch * batch_size;
        int end_j = std::min((batch + 1) * batch_size, grid_size);
        
        std::cout << "Processing batch " << (batch + 1) << " of " << num_batches << std::endl;
        heatmap_batch(system, quantifier, start_j, end_j, grid_size, vx_start, vy_start, vx_step, vy_step, outfile_path);
        std::cout << "\nBatch " << (batch + 1) << " complete." << std::endl;
    }
    
    std::cout << "\nAll results saved to " << outfile_path << std::endl;
}

// Wrapper function for proximity to match the function pointer signature
double proximity_wrapper(System& system, const std::vector<double>& initial_state, double total_time, double delta_t) {
    return system.proximity(initial_state, total_time, delta_t);
}

int main() {
    System system;
    double vx_start, vx_end, vy_start, vy_end;
    int dimension;
    
    // Get the input string from the user
    std::string input;
    std::cout << "Enter vx start, vx end, vy start, vy end: ";
    std::getline(std::cin, input);
    std::cout << "Enter the dimension of grid (1 second per 100 pixels): ";
    std::cin >> dimension;

    
    // Use istringstream to parse the values
    std::istringstream iss(input);
    if (iss >> vx_start >> vx_end >> vy_start >> vy_end) {
        // Use the parsed values
        std::cout << "vx start: " << vx_start << "\n";
        std::cout << "vx end: " << vx_end << "\n";
        std::cout << "vy start: " << vy_start << "\n";
        std::cout << "vy end: " << vy_end << "\n";
    } else {
        std::cerr << "Error: Invalid input format." << std::endl;
    }
    
    heatmap(system, proximity_wrapper, vx_start, vx_end, vy_start, vy_end, dimension);
    return 0;
}

