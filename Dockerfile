FROM python:3.12-slim

RUN apt-get update \
    && apt-get install -y --no-install-recommends openssh-client \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --create-home --uid 1000 --shell /bin/bash runner

WORKDIR /app
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt
COPY app.py h3_state.py h3_jobs.py h3_compute.py local_text_wire.py scheduler.py node_manifest.py planner_callback.py entrypoint.sh ./
RUN chmod 0755 /app/entrypoint.sh \
    && install -d -o runner -g runner -m 0700 \
        /home/runner/.ssh \
        /home/runner/.local/bin \
        /home/runner/.cache/gpu-a-connect \
        /home/runner/.cache/gpu-b-connect \
        /var/lib/gpu-orchestrator
USER runner
ENTRYPOINT ["/app/entrypoint.sh"]
