################################################################################
# ROBIN — Vivado place & route driver
#   Usage: vivado -mode batch -source place_route.tcl \
#                 -tclargs --directive <tcl-fragment> --seed <int> \
#                          --corner <name> --out <dir>
#
# Reads the post-synth .dcp produced by synth.tcl, applies the directive
# bundle emitted by the ROBIN policy, runs place_design / phys_opt /
# route_design, and dumps the canonical report set for parsing.
################################################################################
package require cmdline

set parameters {
    {directive.arg "" "directive fragment file"}
    {seed.arg      "0" "P&R seed"}
    {corner.arg    "TT25" "PVT corner label"}
    {out.arg       "run_out" "output directory"}
    {dcp.arg       "post_synth.dcp" "post-synth checkpoint"}
}
array set opts [::cmdline::getoptions argv $parameters]

file mkdir $opts(out)
set seed $opts(seed)
set corner $opts(corner)

open_checkpoint $opts(dcp)
if {[file exists $opts(directive)]} { source $opts(directive) }

# Pull pblock variant from environment (set by directive)
if {[info exists ROBIN_pblock] && $ROBIN_pblock ne "none"} {
    create_pblock pblock_$ROBIN_pblock
    if {$ROBIN_pblock eq "central"} {
        resize_pblock pblock_central -add {SLICE_X10Y10:SLICE_X40Y90}
    } elseif {$ROBIN_pblock eq "top_corner"} {
        resize_pblock pblock_top_corner -add {SLICE_X0Y80:SLICE_X20Y120}
    } elseif {$ROBIN_pblock eq "edge_band"} {
        resize_pblock pblock_edge_band -add {SLICE_X0Y0:SLICE_X5Y120}
    }
}

# Place
place_design -directive [expr {[info exists STRATEGY] ? $STRATEGY : "Default"}] -seed $seed

# Phys-opt (optional)
if {[info exists ROBIN_phys_opt] && $ROBIN_phys_opt ne "off"} {
    if {$ROBIN_phys_opt eq "aggressive"} {
        phys_opt_design -directive AggressiveExplore -retime
    } else {
        phys_opt_design
    }
}

# Route
if {[info exists ROBIN_route_effort] && $ROBIN_route_effort eq "high"} {
    route_design -directive AggressiveExplore -tns_cleanup
} else {
    route_design -tns_cleanup
}

# Reports
report_timing_summary -file [file join $opts(out) report_timing_summary.rpt] -warn_on_violation
report_utilization -file [file join $opts(out) report_utilization.rpt]
report_power -file [file join $opts(out) report_power.rpt]
write_checkpoint -force [file join $opts(out) routed.dcp]

# Audit manifest with seed/corner/directive hash
set manifest [file join $opts(out) manifest_partial.json]
set fh [open $manifest "w"]
puts $fh "{"
puts $fh "  \"seed\": $seed,"
puts $fh "  \"corner\": \"$corner\","
puts $fh "  \"tool_version\": \"[version -short]\""
puts $fh "}"
close $fh

exit 0
