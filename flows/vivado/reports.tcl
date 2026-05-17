################################################################################
# ROBIN — Vivado reporting driver (post-route).
#   Usage: vivado -mode batch -source reports.tcl -tclargs --dcp <file> --out <dir>
################################################################################
package require cmdline
set parameters {
    {dcp.arg "routed.dcp" "routed checkpoint"}
    {out.arg "reports"    "output directory"}
}
array set opts [::cmdline::getoptions argv $parameters]
file mkdir $opts(out)
open_checkpoint $opts(dcp)

report_timing_summary  -file [file join $opts(out) report_timing_summary.rpt] -warn_on_violation
report_timing          -file [file join $opts(out) report_timing.rpt] -nworst 20 -sort_by group
report_utilization     -file [file join $opts(out) report_utilization.rpt]
report_power           -file [file join $opts(out) report_power.rpt]
report_clock_utilization -file [file join $opts(out) report_clock_util.rpt]
report_methodology     -file [file join $opts(out) report_methodology.rpt]

exit 0
