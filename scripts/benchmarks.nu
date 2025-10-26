#!/bin/nu

const examples = 'target/release/examples'
const cli = ($examples | path join cli)
const accuracy = ($examples | path join accuracy)
const images = 'img/unsplash/img/Original'
const k = 256
const trials = 30

const methods = [
    [name, cli_args, dither_args];
    ['Wu (sRGB)', [quantette -t 4 --srgb], null]
    ['Wu (Oklab)', [quantette -t 4], [--dither]]
    ['k-means', [quantette -t 4 --kmeans], [--dither]]
    [imagequant, [imagequant -t 4 -q 100], [--dither-level 1.0]]
    [color_quant, [neuquant --sample-frac 10], null]
    [exoquant, [exoquant], [--dither]]
]

def main [--dither, --no-dither] {
    let dithers = match [$dither $no_dither] {
        [true true] => [false true]
        [true false] => [true]
        [false true] => [false]
        [false false] => [false true]
    }

    let output = mktemp -t quantette_benchmark_output.XXX --suffix .png

    let table = (
        ls $images
        | select name
        | rename path
        | insert Image { $in.path | path relative-to $images }
    )

    cargo b -r --example accuracy o+e> /dev/null
    RUSTFLAGS='-C target-feature=+avx2' cargo b -r --example cli o+e> /dev/null

    for dither in $dithers {
        print (if $dither { '# With Dithering' } else { '# Without Dithering' })
        print ''

        print '## Time'
        print ''

        let methods = if $dither {
            $methods | where dither_args != null
        } else {
            $methods | update dither_args []
        }

        $methods
        | reduce -f $table {|method, table|
            $table | insert $method.name {|image|
                1..$trials
                | each {
                    ^$cli $image.path -o $output --verbose -k $k ...$method.cli_args ...$method.dither_args
                    | lines
                    | parse 'quantization and remapping took {time}ms'
                    | get time.0
                    | into int
                }
                | math avg
                | math round
                | into int
            }
        }
        | reject path
        | to md --pretty
        | print

        print ''

        print '## Accuracy/DSSIM'
        print ''

        $methods
        | reduce -f $table {|method, table|
            $table | insert $method.name {|image|
                ^$cli $image.path -o $output -k $k ...$method.cli_args ...$method.dither_args
                ^$accuracy compare $image.path $output | into float | math round -p 6
            }
        }
        | reject path
        | to md --pretty
        | print

        print ''
    }

    rm $output
}
