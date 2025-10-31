#! /usr/bin/env gnuplot

set style histogram clustered gap 0
set style fill solid
unset xtics

plot data binary format="%uint32%uint32" using 2:1 notitle lc rgb var with histogram

pause mouse close
