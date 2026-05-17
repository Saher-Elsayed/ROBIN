################################################################################
# ROBIN — Quartus STA-only re-run.
################################################################################
load_package flow
load_package sta

create_timing_netlist
read_sdc
update_timing_netlist
report_clocks
report_timing -setup -npaths 50
report_timing -hold  -npaths 50
report_min_pulse_width
delete_timing_netlist
