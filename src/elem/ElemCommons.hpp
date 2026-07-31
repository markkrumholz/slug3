/**
 * @file ElemCommons.hpp
 * @author Mark Krumholz
 * @brief Common definitions for the elem namespace
 * @date 2026-07-31
 */

#ifndef ELEMCOMMONS_HPP
#define ELEMCOMMONS_HPP

namespace elem
{
    /**
     * @brief Atomic symbols for all 118 elements, ordered by atomic number
     */
    enum class Symbols
    {
        H,                                                      // Z=1
        He,                                                     // Z=2
        Li, Be,  B,  C,  N,  O,  F, Ne,                       // Z=3-10
        Na, Mg, Al, Si,  P,  S, Cl, Ar,                       // Z=11-18
         K, Ca, Sc, Ti,  V, Cr, Mn, Fe, Co, Ni, Cu, Zn,       // Z=19-30
        Ga, Ge, As, Se, Br, Kr,                                // Z=31-36
        Rb, Sr,  Y, Zr, Nb, Mo, Tc, Ru, Rh, Pd, Ag, Cd,       // Z=37-48
        In, Sn, Sb, Te,  I, Xe,                                // Z=49-54
        Cs, Ba,                                                 // Z=55-56
        La, Ce, Pr, Nd, Pm, Sm, Eu, Gd, Tb, Dy, Ho, Er, Tm, Yb, Lu, // Z=57-71
        Hf, Ta,  W, Re, Os, Ir, Pt, Au, Hg,                   // Z=72-80
        Tl, Pb, Bi, Po, At, Rn,                                // Z=81-86
        Fr, Ra,                                                 // Z=87-88
        Ac, Th, Pa,  U, Np, Pu, Am, Cm, Bk, Cf, Es, Fm, Md, No, Lr, // Z=89-103
        Rf, Db, Sg, Bh, Hs, Mt, Ds, Rg, Cn, Nh, Fl, Mc, Lv, Ts, Og, // Z=104-118
        nElem
    };

} // namespace elem

#endif // ELEMCOMMONS_HPP
