################################################################################
# ROBIN — Vivado synthesis driver
#   Usage: vivado -mode batch -source synth.tcl \
#                 -tclargs --design <name> --rtl <dir> --xdc <file> \
#                          --device <part> --period <ns> --out <dir>
################################################################################
package require cmdline

set parameters {
    {design.arg "" "design name"}
    {rtl.arg    "" "rtl source dir"}
    {xdc.arg    "" "constraints file"}
    {device.arg "xcve2302-sfva784-2MP-e-S" "FPGA part"}
    {period.arg "5.0" "target period in ns"}
    {out.arg    "post_synth" "output directory"}
}
array set opts [::cmdline::getoptions argv $parameters]

file mkdir $opts(out)
set proj_dir $opts(out)

create_project -in_memory -part $opts(device)

# Add RTL
foreach f [glob -nocomplain $opts(rtl)/*.v $opts(rtl)/*.sv $opts(rtl)/*.vhd] {
    if {[string match "*.vhd" $f]} {
        read_vhdl $f
    } else {
        read_verilog $f
    }
}
read_xdc $opts(xdc)

# Set clock period
set period_ns $opts(period)
create_clock -name clk -period $period_ns [get_ports clk] -quiet

# Synthesise
synth_design -top $opts(design) -part $opts(device)
opt_design -directive ExploreSequentialArea

# Persist the post-synth checkpoint and reports
write_checkpoint -force [file join $proj_dir post_synth.dcp]
report_utilization -file [file join $proj_dir post_synth_util.rpt]
report_timing_summary -file [file join $proj_dir post_synth_timing.rpt] -warn_on_violation

# Emit timing graph + tabular features in JSON for the encoder
# (Implementation note: the timing graph is extracted via the Tcl property
# accessor on each cell; the resulting JSON is consumed by the ROBIN
# encoder. See robin/environment.py for the parser.)
source [file join [file dirname [info script]] extract_features.tcl]

exit 0
