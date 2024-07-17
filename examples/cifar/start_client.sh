# Read argument corresponding to the number of clients

NUM_CLIENTS=$(($1 - 1))

for i in $(seq 0 $NUM_CLIENTS); do
    echo "Starting client $i"
    python client.py --client-id $i --server-ip "localhost" --server-port 8080 --keys-dir keys & 
done

# This will allow you to use CTRL+C to stop all background processes
trap "trap - SIGTERM && kill -- -$$" SIGINT SIGTERM
# Wait for all background processes to complete
wait
