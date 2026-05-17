################################################################################
# ROBIN — Quartus Prime Pro synthesis driver
#   Usage: quartus_sh -t synthesis.tcl \
#                     --design <name> --rtl <dir> --sdc <file> \
#                     --device <part> --out <dir>
################################################################################
load_package flow
load_package report

# Parse args
array set opts {design "" rtl "" sdc "" device "AGIB027R29A1E2VR0" out "post_synth"}
for {set i 0} {$i < $argc} {incr i} {
    set arg [lindex $argv $i]
    switch -- $arg {
        --design { set opts(design) [lindex $argv [incr i]] }
        --rtl    { set opts(rtl)    [lindex $argv [incr i]] }
        --sdc    { set opts(sdc)    [lindex $argv [incr i]] }
        --device { set opts(device) [lindex $argv [incr i]] }
        --out    { set opts(out)    [lindex $argv [incr i]] }
    }
}
file mkdir $opts(out)

project_new $opts(design) -overwrite
set_global_assignment -name FAMILY     "Agilex 7"
set_global_assignment -name DEVICE     $opts(device)
set_global_assignment -name TOP_LEVEL_ENTITY $opts(design)
set_global_assignment -name SDC_FILE   $opts(sdc)

foreach f [glob -nocomplain $opts(rtl)/*.v $opts(rtl)/*.sv $opts(rtl)/*.vhd] {
    if {[string match "*.vhd" $f]} {
        set_global_assignment -name VHDL_FILE $f
    } else {
        set_global_assignment -name VERILOG_FILE $f
    }
}

execute_module -tool map
execute_module -tool syn

project_close
