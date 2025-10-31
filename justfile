check:
  typos
  cargo fmt --check
  cargo doc --all-features --no-deps
  cargo hack --rust-version --feature-powerset clippy
  cargo hack --target wasm32-unknown-unknown --rust-version --feature-powerset --exclude-all-features --skip default,std,threads,image clippy

test:
  cargo test --all-features --doc
  cargo test --lib

test-hack:
  cargo test --all-features --doc
  cargo hack --rust-version --feature-powerset test --lib

plot-palette image *args:
  #! /usr/bin/env bash
  set -e
  image="$(realpath "{{image}}")"
  cd '{{justfile_directory()}}/examples/plot'
  data='data/{{file_stem(image)}}'
  mkdir -p "$data"
  pixels="$data/pixels.dat"
  palette="$data/palette.dat"
  cargo r -r --example plot -- palette "$image" --pixels-output "$pixels" {{args}} > "$palette"
  gnuplot -e "pixels='$pixels'; palette='$palette'" palette.gnuplot

plot-freq image *args:
  #! /usr/bin/env bash
  set -e
  image="$(realpath "{{image}}")"
  cd '{{justfile_directory()}}/examples/plot'
  data='data/{{file_stem(image)}}'
  mkdir -p "$data"
  data="$data/freq.dat"
  cargo r -r --example plot -- freq "$image" {{args}} > "$data"
  gnuplot -e "data='$data'" freq.gnuplot
