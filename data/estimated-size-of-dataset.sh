#!/usr/bin/env bash

# this script estimates how big the total dataset will be in terms of storage space.
# in hindsight, I might have been better off implementing this in Golang ¯\_(ツ)_/¯

size_of_pair() {
    lang1="$1"; 
    lang2="$2"; 

    curl "https://opus.nlpl.eu/?src=$lang1&trg=$lang2&minsize=all" --silent |
        grep '<i>total</i>' | 
        sed 's/<\/th>/\n/g' | 
        sed -n '8p' | 
        sed 's/.*>//';
}

languages="en zh hi es fr ar bn ru pt ur id de jp yo my pl ln mr te tr ta vi tl ko fa ha sw jv it pa kn gu th am"

total_pairs="0";
pairs_tested="0";
total_size="0";
for lang1 in $languages; do
    for lang2 in $languages; do
        [ "$lang1" = "$lang2" ] && continue;
        total_pairs=$(($total_pairs+1));

        [ "$RANDOM" -lt "750" ] || continue;
        pairs_tested=$(($pairs_tested+1));
        size=$(size_of_pair $lang1 $lang2);

        # we only want to consider pairs which are at least 0.1M in size.
        # pairs with fewer than that are so small that they don't matter much.
        [[ "$size" =~ .*M$ ]] || continue;
        total_size=$(bc <<<"$total_size + ${size::-1}");
    done
done

size_in_GB=$(bc <<<"($total_size * $total_pairs / $pairs_tested) / 1000");
echo "The dataset will be around $size_in_GB gigabytes in total";
