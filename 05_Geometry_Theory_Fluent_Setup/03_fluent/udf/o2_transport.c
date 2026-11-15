/* ==========================================================================
 * o2_transport.c  -  ANSYS Fluent UDF for dissolved-oxygen transport and
 *                    cellular consumption in the Hele-Shaw channel
 * --------------------------------------------------------------------------
 * Transported quantity : User-Defined Scalar 0 (uds-0) = C, dissolved O2 [mol m^-3]
 * Fluent UDS equation  : d(rho*phi)/dt + div(rho*u*phi - Gamma grad phi) = S_phi
 *   With phi = C, Gamma = rho*D and constant rho, dividing by rho gives
 *          dC/dt + u.grad C = D lap C + S_phi/rho
 *   => every source / boundary FLUX handed to Fluent must be multiplied by rho.
 *      (S_phi = -rho*R_vol   ;   wall flux = rho * N, N in mol m^-2 s^-1)
 *
 * Hooks provided
 *   DEFINE_DIFFUSIVITY  o2_diffusivity          -> Materials > medium > UDS Diffusivity
 *   DEFINE_PROFILE      o2_flux_bottom_mm       -> wall_bottom : Michaelis-Menten flux (Model 3, recommended)
 *   DEFINE_PROFILE      o2_flux_bottom_zero     -> wall_bottom : constant flux (Model 1, zero order)
 *   DEFINE_SOURCE       o2_sink_cell_layer      -> fluid zone  : volumetric MM sink for z < delta (Model 2)
 *   DEFINE_PROFILE      o2_flux_pdms            -> wall_top / wall_side : Robin supply through PDMS
 *   DEFINE_ON_DEMAND    o2_report               -> integral checks on the bottom wall
 *
 * Parameters are read at run time from Fluent RP variables (load o2_params.scm),
 * so sensitivity runs do not need recompilation. Defaults (used if a variable is
 * missing) are the BASELINE values of the project manual; see the manual for the
 * literature sources and which values are ASSUMED / ILLUSTRATIVE.
 *
 * Sign convention: Fluent "Specified Flux" for a UDS at a wall is treated here as
 * POSITIVE INTO THE FLUID. The run-time switch o2/flux-sign (+1/-1) lets you flip
 * it if the mass-balance test in the manual (Phase 8, step 8.6) shows the opposite.
 *
 * Compile: Define > User-Defined > Functions > Compiled > Add o2_transport.c > Build > Load
 *          (Windows: tick "Use Built-In Compiler")
 * ========================================================================== */
#include "udf.h"

#define O2 0                                /* UDS index */

/* ---------------- baseline defaults (SI) ---------------------------------- */
#define DEF_D        3.0e-9    /* O2 diffusivity in medium, 37 C   [m2/s]      */
#define DEF_CSAT     0.178     /* air-saturated medium, 37 C, 5% CO2 [mol/m3]  */
#define DEF_QMAX     4.0e-8    /* areal max uptake = rho_cell*OCR [mol/m2/s]   */
                               /* = 2e9 cells/m2 * 20e-18 mol/cell/s (ILLUSTRATIVE) */
#define DEF_KM       6.3e-4    /* Michaelis-Menten constant [mol/m3] (0.5 mmHg) */
#define DEF_DELTA    1.0e-5    /* cell-layer thickness for Model 2 [m]          */
#define DEF_KPDMS    5.65e-6   /* PDMS mass-transfer coeff. P/(t*alpha) [m/s]  */
#define DEF_CCRIT    0.0178    /* hypoxia threshold for reporting [mol/m3]      */
#define DEF_SIGN     1.0
#define DEF_BOTID    -1        /* zone ID of wall_bottom for o2_report          */

static real rp_real(const char *name, real def)
{
    if (RP_Variable_Exists_P(name)) return RP_Get_Real(name);
    return def;
}
static int rp_int(const char *name, int def)
{
    if (RP_Variable_Exists_P(name)) return RP_Get_Integer(name);
    return def;
}

/* wall-face concentration: face value if stored, otherwise adjacent cell value */
static real wall_conc(face_t f, Thread *t)
{
    real c;
    if (NNULLP(THREAD_STORAGE(t, SV_UDS_I(O2))))
        c = F_UDSI(f, t, O2);
    else
        c = C_UDSI(F_C0(f, t), THREAD_T0(t), O2);
    return (c > 0.0) ? c : 0.0;
}

static real wall_rho(face_t f, Thread *t)
{
    return C_R(F_C0(f, t), THREAD_T0(t));
}

/* ------------------------------------------------------------------------ */
DEFINE_DIFFUSIVITY(o2_diffusivity, c, t, i)
{
    return C_R(c, t) * rp_real("o2/diff", DEF_D);      /* Gamma = rho * D */
}

/* Model 3 (recommended): Michaelis-Menten consumption by the monolayer on z = 0
 *   N_cell = q_max * C_w / (K_m + C_w)        [mol m^-2 s^-1], leaves the fluid */
DEFINE_PROFILE(o2_flux_bottom_mm, t, i)
{
    face_t f;
    real qmax = rp_real("o2/qmax", DEF_QMAX);
    real km   = rp_real("o2/km", DEF_KM);
    real sgn  = rp_real("o2/flux-sign", DEF_SIGN);
    real cw, q;
    begin_f_loop(f, t)
    {
        cw = wall_conc(f, t);
        q  = qmax * cw / (km + cw);
        F_PROFILE(f, t, i) = sgn * (-q) * wall_rho(f, t);
    }
    end_f_loop(f, t)
}

/* Model 1: zero-order (constant) consumption. Can drive C below zero where
 * supply is insufficient -> use only for comparison / mass-balance test.     */
DEFINE_PROFILE(o2_flux_bottom_zero, t, i)
{
    face_t f;
    real qmax = rp_real("o2/qmax", DEF_QMAX);
    real sgn  = rp_real("o2/flux-sign", DEF_SIGN);
    begin_f_loop(f, t)
    {
        F_PROFILE(f, t, i) = sgn * (-qmax) * wall_rho(f, t);
    }
    end_f_loop(f, t)
}

/* Model 2: volumetric MM sink in the cells whose centroid lies below z = delta.
 *   R_vol = (q_max/delta) * C/(K_m + C)      [mol m^-3 s^-1]
 * Approximation: the fluid in the layer still flows (the monolayer is not a
 * separate solid/porous zone). Hook in Cell Zone Conditions > fluid > Source Terms > uds-0. */
DEFINE_SOURCE(o2_sink_cell_layer, c, t, dS, eqn)
{
    real x[ND_ND];
    real delta = rp_real("o2/delta", DEF_DELTA);
    real km    = rp_real("o2/km", DEF_KM);
    real vmax, conc, rho;
    C_CENTROID(x, c, t);
    if (x[2] > delta)
    {
        dS[eqn] = 0.0;
        return 0.0;
    }
    vmax = rp_real("o2/qmax", DEF_QMAX) / delta;
    conc = C_UDSI(c, t, O2);
    if (conc < 0.0) conc = 0.0;
    rho  = C_R(c, t);
    dS[eqn] = -rho * vmax * km / ((km + conc) * (km + conc));   /* implicit part, <= 0 */
    return -rho * vmax * conc / (km + conc);
}

/* Robin condition for oxygen entering through the gas-permeable PDMS:
 *   N = k_pdms (C_sat - C_w),  k_pdms = P_O2 / (t_pdms * alpha_medium)
 * k_pdms = 0 -> impermeable wall ; very large k_pdms -> C_w = C_sat (Dirichlet). */
DEFINE_PROFILE(o2_flux_pdms, t, i)
{
    face_t f;
    real k    = rp_real("o2/kpdms", DEF_KPDMS);
    real csat = rp_real("o2/csat", DEF_CSAT);
    real sgn  = rp_real("o2/flux-sign", DEF_SIGN);
    begin_f_loop(f, t)
    {
        F_PROFILE(f, t, i) = sgn * k * (csat - wall_conc(f, t)) * wall_rho(f, t);
    }
    end_f_loop(f, t)
}

/* Integral diagnostics on the cell-covered wall (set o2/bottom-id first):
 *   area, area-averaged C_w, min C_w, total MM uptake [mol/s],
 *   hypoxic area fraction (C_w < o2/ccrit)                                     */
DEFINE_ON_DEMAND(o2_report)
{
    real A = 0.0, CA = 0.0, U = 0.0, Ahyp = 0.0, cmin = 1.0e30;
#if !RP_HOST
    Domain *d = Get_Domain(1);
    int id = rp_int("o2/bottom-id", DEF_BOTID);
    Thread *t;
    face_t f;
    real av[ND_ND], a, cw;
    real qmax = rp_real("o2/qmax", DEF_QMAX);
    real km   = rp_real("o2/km", DEF_KM);
    real ccr  = rp_real("o2/ccrit", DEF_CCRIT);
    if (id < 0)
    {
        Message0("o2_report: set (rpsetvar 'o2/bottom-id <zone id of wall_bottom>) first\n");
        return;
    }
    t = Lookup_Thread(d, id);
    begin_f_loop(f, t)
    {
        if (PRINCIPAL_FACE_P(f, t))
        {
            F_AREA(av, f, t);
            a  = NV_MAG(av);
            cw = wall_conc(f, t);
            A  += a;
            CA += cw * a;
            U  += qmax * cw / (km + cw) * a;
            if (cw < ccr) Ahyp += a;
            if (cw < cmin) cmin = cw;
        }
    }
    end_f_loop(f, t)
    A = PRF_GRSUM1(A); CA = PRF_GRSUM1(CA); U = PRF_GRSUM1(U); Ahyp = PRF_GRSUM1(Ahyp);
    cmin = PRF_GRLOW1(cmin);
    Message0("\n--- o2_report (wall zone %d) ---\n", id);
    Message0("bottom area                 = %g m2\n", A);
    Message0("area-avg C_w                = %g mol/m3\n", (A > 0 ? CA / A : 0.0));
    Message0("min C_w                     = %g mol/m3\n", cmin);
    Message0("total MM uptake (Model 3)   = %g mol/s\n", U);
    Message0("hypoxic area fraction       = %g  (C_w < %g mol/m3)\n", (A > 0 ? Ahyp / A : 0.0), ccr);
#endif
}
