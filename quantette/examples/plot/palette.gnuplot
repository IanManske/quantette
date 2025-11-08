#! /usr/bin/env gnuplot

point_size = 0.5
centroid_size = 2

splot \
  pixels binary format="%3float32%uint32" using 1:2:3:4 notitle with points ps point_size pt 5 lc rgb var, \
  palette using "a":"b":"l":"color" notitle with points ps (centroid_size + 1) pt 7 lc black, \
  palette using "a":"b":"l":"color" notitle with points ps centroid_size pt 7 lc rgb var, \

pause mouse close

