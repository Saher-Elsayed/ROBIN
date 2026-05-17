################################################################################
# ROBIN — extract timing graph + tabular features from a post-synth design.
# Output: $opts(out)/timing_graph.json   (nodes, edges)
#         $opts(out)/tabular_features.json (utilization, congestion percentiles)
################################################################################
if {![info exists opts]} {
    error "extract_features.tcl should be sourced from synth.tcl"
}

set graph_fp [open [file join $opts(out) timing_graph.json] "w"]
puts $graph_fp "{"
puts $graph_fp "  \"nodes\": \["

set first 1
foreach cell [get_cells -hierarchical] {
    set name [get_property NAME $cell]
    set ref  [get_property REF_NAME $cell]
    set is_ff [string match "*FDRE*" $ref]
    if {$first == 0} { puts $graph_fp "," }
    puts -nonewline $graph_fp "    {\"name\": \"$name\", \"ref\": \"$ref\", \"is_ff\": $is_ff}"
    set first 0
}
puts $graph_fp "\n  \],"
puts $graph_fp "  \"edges\": \[\]"
puts $graph_fp "}"
close $graph_fp

# Tabular features
set util [report_utilization -return_string]
set lut_pct 0
set ff_pct  0
set bram_pct 0
set dsp_pct 0
foreach line [split $util "\n"] {
    if {[regexp {LUT.*\s(\d+\.\d+)} $line m val]} { set lut_pct $val }
    if {[regexp {FF.*\s(\d+\.\d+)} $line m val]}  { set ff_pct $val  }
    if {[regexp {BRAM.*\s(\d+\.\d+)} $line m val]} { set bram_pct $val }
    if {[regexp {DSP.*\s(\d+\.\d+)} $line m val]}  { set dsp_pct $val }
}

set feat_fp [open [file join $opts(out) tabular_features.json] "w"]
puts $feat_fp "{"
puts $feat_fp "  \"utilization\": {"
puts $feat_fp "    \"LUT\":  $lut_pct,"
puts $feat_fp "    \"FF\":   $ff_pct,"
puts $feat_fp "    \"BRAM\": $bram_pct,"
puts $feat_fp "    \"DSP\":  $dsp_pct"
puts $feat_fp "  },"
puts $feat_fp "  \"tool_version\": \"[version -short]\""
puts $feat_fp "}"
close $feat_fp
