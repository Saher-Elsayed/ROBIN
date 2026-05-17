################################################################################
# ROBIN — Vivado canonical strategy bundles (referenced by directive.tcl deltas)
################################################################################

# Default (baseline)
set_property STRATEGY                "Vivado Synthesis Defaults" [current_run -synthesis]
set_property STRATEGY                "Vivado Implementation Defaults" [current_run -implementation]

# Aggressive — used by ROBIN as a high-effort starting point
# (overridden per-run by the directive bundle the policy emits)
# set_property STRATEGY                "Performance_ExploreWithRemap" [current_run -implementation]

# Common BUFG / clock-domain hints
set_property CONFIG_VOLTAGE          1.0  [current_design]
set_property CFGBVS                  GND  [current_design]
