#!/bin/sh
set -eu

PIDS=""

stop() {
    for pid in $PIDS; do
        kill "$pid" 2>/dev/null || true
    done
    for pid in $PIDS; do
        wait "$pid" 2>/dev/null || true
    done
}
trap stop EXIT INT TERM

node_tunnel() {
    node=$1
    agent_port=$2
    model_port=$3
    h3_port=$4
    h3_ref_port=$5
    graph_port=$6
    tts_port=$7
    seedvr3_port=$8
    orchestrator_socket=$9
    qwen38_native_port=${10}
    qwen38_yarn_port=${11}
    retry_delay=10
    retry_delay_max=300
    while true; do
        echo "$(date -Iseconds) node_tunnel_connect node=$node" >&2
        if ! ssh -o BatchMode=yes -o ConnectTimeout=10 "$node" \
            "if [ -S '$orchestrator_socket' ]; then rm -f '$orchestrator_socket'; elif [ -e '$orchestrator_socket' ]; then echo 'refusing to replace non-socket path: $orchestrator_socket' >&2; exit 1; fi"; then
            echo "$(date -Iseconds) node_tunnel_retry node=$node delay_seconds=$retry_delay" >&2
            sleep "$retry_delay"
            retry_delay=$((retry_delay * 2))
            if [ "$retry_delay" -gt "$retry_delay_max" ]; then
                retry_delay=$retry_delay_max
            fi
            continue
        fi
        retry_delay=10
        ssh -NT \
            -o BatchMode=yes \
            -o ConnectTimeout=10 \
            -o ExitOnForwardFailure=yes \
            -o ServerAliveInterval=15 \
            -o ServerAliveCountMax=3 \
            -o StreamLocalBindMask=0177 \
            -o StreamLocalBindUnlink=yes \
            -L "127.0.0.1:${agent_port}:/home/worker/.local/state/gpu-runtime/agent.sock" \
            -L "127.0.0.1:${model_port}:127.0.0.1:30000" \
            -L "127.0.0.1:${qwen38_native_port}:127.0.0.1:30020" \
            -L "127.0.0.1:${qwen38_yarn_port}:127.0.0.1:30021" \
            -L "127.0.0.1:${h3_port}:127.0.0.1:30010" \
            -L "127.0.0.1:${h3_ref_port}:127.0.0.1:30011" \
            -L "127.0.0.1:${graph_port}:/home/worker/.local/state/gpu-runtime/graph.sock" \
            -L "127.0.0.1:${tts_port}:/home/worker/.local/state/gpu-runtime/tts.sock" \
            -L "127.0.0.1:${seedvr3_port}:/home/worker/.local/state/gpu-runtime/seedvr3.sock" \
            -R "${orchestrator_socket}:127.0.0.1:8080" \
            "$node" || true
        sleep 2
    done
}

start_node_tunnel() {
    node_tunnel "$@" &
    PIDS="$PIDS $!"
}

# All host/port arguments come from the same validated snapshot as app.py.
boot_config=$(mktemp)
node_rows=$(python /app/node_manifest.py "${GPU_ORCHESTRATOR_CONFIG:-/app/nodes.json}" --snapshot "$boot_config")
export GPU_ORCHESTRATOR_CONFIG="$boot_config"
while read -r node agent model h3 h3_ref graph tts seedvr3 callback qwen_native qwen_yarn; do
    start_node_tunnel "$node" "$agent" "$model" "$h3" "$h3_ref" "$graph" "$tts" "$seedvr3" "$callback" "$qwen_native" "$qwen_yarn"
done <<TUNNEL_ROWS
$node_rows
TUNNEL_ROWS

exec python -u /app/app.py
