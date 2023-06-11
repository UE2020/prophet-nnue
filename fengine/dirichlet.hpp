#pragma once

#include <random>
#include <vector>

template <typename RealType = double>
class DirichletDistribution {
public:
    DirichletDistribution(const std::vector<RealType>& alpha) {
        set_params(alpha);
    }

    void set_params(const std::vector<RealType>& new_params) {
        alpha = new_params;
        std::vector<std::gamma_distribution<RealType>> new_gamma(alpha.size());
        for (size_t i = 0; i < alpha.size(); ++i) {
            std::gamma_distribution<RealType> temp(alpha[i], 1);
            new_gamma[i] = temp;
        }
        gamma = new_gamma;
    }

    std::vector<RealType> get_params() const {
        return alpha;
    }

    template <typename RNG>
    std::vector<RealType> operator()(RNG& generator) {
        std::vector<RealType> x(alpha.size());
        RealType sum = 0;
        for (size_t i = 0; i < alpha.size(); ++i) {
            x[i] = gamma[i](generator);
            sum += x[i];
        }
        for (RealType& xi : x) xi = xi / sum;
        return x;
    }

private:
    std::vector<RealType> alpha;
    std::vector<std::gamma_distribution<RealType>> gamma;
};
