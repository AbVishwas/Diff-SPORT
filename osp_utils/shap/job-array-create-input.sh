# Make this a standalone script or run in a terminal
start=14400
final_stop=17200
step=50
num=50
ncoalitions=(3000)

while [[ $start -lt $final_stop ]]; do
    stop=$((start + num))
    for coalition in "${ncoalitions[@]}"; do
        echo "start=$start stop=$stop step=$step coalition=$coalition"
    done
    start=$((start + num))
done > input_params.txt
