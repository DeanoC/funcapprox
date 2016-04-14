//
// Created by Dean Calver on 12/04/2016.
//

#pragma once


class RealFunc {
public:

    Core::real operator()(const Core::real x);

private:
    Core::real sine(const Core::real x);
};


