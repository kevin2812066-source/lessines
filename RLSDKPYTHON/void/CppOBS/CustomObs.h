#pragma once
#include "ObsBuilder.h"

namespace RLGC {
    // CustomObs ultra optimisé 1v1 avec ball prediction simple
    // Observation complète 147 features: 9+6+18+7+6+8+4+2+34+8+21+38
    // Reproduction EXACTE du Python obs_cpp_exact.py
    class CustomObs : public ObsBuilder {
    public:
        // Coefficients de normalisation
        constexpr static float
            POS_COEF = 1.f / 5000.f,
            VEL_COEF = 1.f / 2300.f,
            ANG_VEL_COEF = 1.f / 3.f;

        CustomObs() = default;
        virtual ~CustomObs() = default;

        virtual FList BuildObs(const Player& player, const GameState& state) override;
    };
}

