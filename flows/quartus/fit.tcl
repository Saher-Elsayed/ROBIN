################################################################################
# ROBIN — Quartus Prime Pro Fitter driver
#   Usage: quartus_sh -t fit.tcl --directive <file> --seed <int> --corner <name> --out <dir>
################################################################################
load_package flow

array set opts {directive "" seed "0" corner "TT25" out "run_out"}
for {set i 0} {$i < $argc} {incr i} {
    set arg [lindex $argv $i]
    switch -- $arg {
        --directive { set opts(directive) [lindex $argv [incr i]] }
        --seed      { set opts(seed)      [lindex $argv [incr i]] }
        --corner    { set opts(corner)    [lindex $argv [incr i]] }
        --out       { set opts(out)       [lindex $argv [incr i]] }
    }
}

if {[file exists $opts(directive)]} { source $opts(directive) }

set_global_assignment -name SEED $opts(seed)
set_global_assignment -name HYPER_RETIMER_ENABLE ON

execute_module -tool fit
execute_module -tool sta -args "--multicorner"
execute_module -tool pow

# Reports
qexec "cp output_files/*.sta.rpt $opts(out)/report_timing_summary.rpt"
qexec "cp output_files/*.fit.summary $opts(out)/report_utilization.rpt"
qexec "cp output_files/*.pow.summary $opts(out)/report_power.rpt"
