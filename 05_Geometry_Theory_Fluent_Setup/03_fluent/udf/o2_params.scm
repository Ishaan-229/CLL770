; o2_params.scm  -  run-time parameters for o2_transport.c
; Load in Fluent:  File > Read > Scheme...  (or TUI:  (load "o2_params.scm") )
; Change a value later in the TUI, e.g.   (rpsetvar 'o2/qmax 1.0e-7)
; All values SI. See the manual, Phase 7-8, for sources and status of each value.

(define (o2-def name val type)
  (if (rp-var-object name) (rpsetvar name val) (rp-var-define name val type #f)))

(o2-def 'o2/diff      3.0e-9  'real)    ; D_O2 in medium at 37 C [m2/s]
(o2-def 'o2/csat      0.178   'real)    ; inlet / saturation concentration [mol/m3]
(o2-def 'o2/qmax      4.0e-8  'real)    ; areal max uptake rho_cell*OCR [mol/m2/s]  (ILLUSTRATIVE)
(o2-def 'o2/km        6.3e-4  'real)    ; Michaelis-Menten constant [mol/m3] (0.5 mmHg)
(o2-def 'o2/delta     1.0e-5  'real)    ; cell-layer thickness, Model 2 [m]
(o2-def 'o2/kpdms     5.65e-6 'real)    ; PDMS Robin coefficient [m/s]; 0 = impermeable
(o2-def 'o2/ccrit     0.0178  'real)    ; reporting threshold (10% of C_sat) [mol/m3]
(o2-def 'o2/flux-sign 1.0     'real)    ; +1: specified wall flux positive INTO fluid
(o2-def 'o2/bottom-id -1      'int)     ; set to the zone ID of wall_bottom
